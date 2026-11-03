"""`vde evaluate`: metrics, figures, ablations, lockbox, shift report and previsao.txt."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from vde import shift, splits
from vde.config import load_config
from vde.engine import Engine
from vde.evaluate import (calibration_comparison, evaluate_lockbox, run_ablations, save_json, summarize)
from vde.forecast import forecast, render
from vde.paths import ARTIFACT_DIR, OUTPUT_DIR
from vde.pipeline import MODEL_FILE, dev_arrays, load_oof, publish
from vde.resources import Resources


def evaluate(ablations: bool = False, open_lockbox: bool = False) -> dict:
    config = load_config()
    res = Resources.load(config)
    model = json.loads((ARTIFACT_DIR / MODEL_FILE).read_text(encoding="utf-8"))
    cv = load_oof()
    summary = summarize(cv, config)
    calib = calibration_comparison(cv)
    conf, hit, per_q = dev_arrays(cv)
    weights, shift_info = _shift(res, model, per_q)
    tau = model["threshold"]
    decisions = publish(config=config)
    fc = forecast(decisions, res, conf, hit, tau, weights)
    report = {"cv": summary, "calibration": calib, "shift": shift_info, "forecast": fc,
              "threshold": tau, "model_threshold_detail": model["threshold_detail"]}
    if open_lockbox:
        dev, lockbox = splits.load_or_create(res, config["evaluation"]["lockbox_size"], config["evaluation"]["seed"])
        report["lockbox"] = evaluate_lockbox(res, dev, lockbox, tau)
    if ablations:
        table = run_ablations(config, model["dev_ids"])
        table.to_csv(OUTPUT_DIR / "ablations.csv", index=False)
        report["ablations"] = table.to_dict("records")
    elif (OUTPUT_DIR / "ablations.csv").exists():
        report["ablations"] = pd.read_csv(OUTPUT_DIR / "ablations.csv").to_dict("records")
    report["repeat_spread"] = _repeat_spread(cv)
    _figures(cv, calib, OUTPUT_DIR)
    (OUTPUT_DIR / "previsao.txt").write_text(render(fc, _explanation(report, model)), encoding="utf-8")
    save_json(OUTPUT_DIR / "evaluation.json", report)
    return report


def _shift(res: Resources, model: dict, per_q: pd.DataFrame):
    new_ids = [q for q in res.question_ids if res.parsed[q].is_new_import]
    old_ids = [q for q in res.question_ids if not res.parsed[q].is_new_import]
    engine = Engine(res, model["state"]["scorer"]["channels"], model["state"]["node_mode"])
    engine.fit(model["dev_ids"], seed=res.config["evaluation"]["seed"])
    new_feats = pd.DataFrame([p.features | {"conf": p.confidence} for p in engine.predict(new_ids)])
    old_unlabeled = [q for q in old_ids if q not in res.gold]
    old_feats = pd.DataFrame([p.features | {"conf": p.confidence} for p in engine.predict(old_unlabeled)])
    dom = shift.domain_weights(per_q, new_feats)
    info = {
        "domain_auc": dom["auc"],
        "effective_n_after_weighting": dom["effective_n"],
        "mean_confidence": {"dev_oof": float(per_q.conf.mean()), "bank_unlabeled": float(old_feats.conf.mean()),
                            "new_import": float(new_feats.conf.mean())},
        "feature_means": {"dev_oof": per_q[["agreement", "top_cosine", "segment_dispersion"]].mean().to_dict(),
                          "new_import": new_feats[["agreement", "top_cosine", "segment_dispersion"]].mean().to_dict()},
        "description": shift.describe(res, old_ids, new_ids),
    }
    save_json(OUTPUT_DIR / "shift_report.json", info)
    return dom["weights"], info


def _repeat_spread(cv) -> dict:
    """Standard deviation of the hit rate between CV repeats: the noise floor of the ablation table."""
    per_rep = cv.rows.groupby("rep").hit.mean()
    return {"repeats": int(len(per_rep)), "mean": float(per_rep.mean()), "std": float(per_rep.std(ddof=1))}


def _figures(cv, calib: dict, out_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="calibração perfeita")
    for name, color in (("raw", "#1f5fa8"), ("platt", "#c8691c")):
        table = calib[name]["reliability"]
        ax.plot([r["mean_conf"] for r in table], [r["hit_rate"] for r in table], marker="o", color=color,
                label=f"{name} (ECE {calib[name]['ece']:.3f})")
    ax.set_xlabel("confiança média do grupo")
    ax.set_ylabel("taxa de acerto observada")
    ax.set_title("Confiabilidade (OOF, 5 grupos de mesma frequência)")
    ax.legend(fontsize=8)
    q = cv.per_question()
    order = np.argsort(-q.conf.to_numpy(), kind="stable")
    hits = q.hit.to_numpy().astype(float)[order]
    k = np.arange(1, len(hits) + 1)
    ax = axes[1]
    ax.plot(k / len(hits), np.cumsum(hits) / k, color="#1f5fa8")
    ax.axhline(0.95, color="#b22222", lw=1, ls="--", label="régua 95%")
    ax.set_xlabel("cobertura (fração publicada)")
    ax.set_ylabel("precisão nas publicadas")
    ax.set_title("Precisão × cobertura (OOF, dev)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "confiabilidade.png", dpi=150)
    plt.close(fig)


def _explanation(report: dict, model: dict) -> list[str]:
    cv, fc, th = report["cv"], report["forecast"], report["model_threshold_detail"]
    lines = [
        "Como chegamos nestes números",
        "",
        f"1. Cobertura: lida direto do publicacao.csv ({fc['published']} publicadas de {fc['evaluated']} "
        "questões fora da folha-verdade). Não é estimativa, por isso o intervalo é degenerado.",
        f"2. Limiar: escolhido na validação cruzada repetida ({model['cv_summary']['questions']} questões de "
        f"desenvolvimento, 10 folds x {load_config()['evaluation']['repeats']} repetições) como o menor valor "
        "com P(precisão >= 0,95) >= 0,80 sob posterior Beta e precisão pontual >= 0,97. "
        f"Limiar = {model['threshold']:.3f}; nesse ponto o dev publicou {th['published']} com {th['hits']:.0f} acertos.",
        f"3. Precisão: posterior Beta por fonte (banco sem peso; importação nova com pesos de importância de um "
        f"classificador de domínio, AUC {report['shift']['domain_auc']:.2f}), mais o ruído binomial do conjunto "
        "publicado real. Intervalo de 90%.",
        f"4. Desempenho geral fora da amostra: acerto do nó {cv['hit_rate']:.2f}, acerto do ramo "
        f"{cv['branch_hit_rate']:.2f}, AUC da confiança {cv['auc']:.2f}, precisão nos 20% mais confiantes "
        f"{cv['precision_at_coverage']['0.20']:.2f}.",
        f"5. Validação do limiar (escolhido em 4/5, medido em 1/5): cobertura {cv['threshold_cv']['coverage']:.2f}, "
        f"precisão {cv['threshold_cv']['precision'] if cv['threshold_cv']['precision'] is not None else 'indefinida'}.",
    ]
    if fc["precision"] is None:
        lines.append("6. Nenhuma questão passou no limiar: a precisão prevista é indefinida (nan). Preferimos não "
                     "publicar a publicar abaixo da régua; tudo vai para revisão com sugestão.")
    if "lockbox" in report:
        lb = report["lockbox"]
        lines.append(f"7. Cofre (20 rótulos nunca vistos): acerto {lb['hit_rate']:.2f}; publicadas "
                     f"{lb['published']}, acertos {lb['published_hits']}.")
    pac = cv["precision_at_coverage"]
    lines += [
        f"8. Por que zero: no dev, os 10% mais confiantes acertam {pac['0.10']:.2f} e os 20% acertam "
        f"{pac['0.20']:.2f}; os dois erros do topo são questões que a equipe marcou como fora mas que parecem "
        "classificáveis (Q0043, Q0291). Com 80 rótulos, certificar 95% com 80% de confiança exige 32 acertos "
        "seguidos sem erro; com 90%, exige 45. Nenhum limiar chega lá.",
        "9. O que destrava a publicação: o professor confirmar as questões do topo da fila, ordenada por "
        "confiança. Cerca de 30 confirmações seguidas sem erro no topo bastam para certificar o limiar a 80%.",
        f"10. Mudança de distribuição: o classificador de domínio separa banco e importação nova com AUC "
        f"{report['shift']['domain_auc']:.2f}, quase o acaso. A mudança está nos metadados: a disciplina de origem "
        "só entra nas regras de fora e o assunto de origem entra apenas como canal fraco, com peso aprendido.",
    ]
    return lines
