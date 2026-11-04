"""`vde evaluate`: metrics, figures, ablations, shift report and previsao.txt."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from vde import shift
from vde.calibrate import min_run_to_certify
from vde.config import load_config
from vde.evaluate import calibration_comparison, run_ablations, save_json, summarize, with_slice
from vde.forecast import forecast, render
from vde.paths import ARTIFACT_DIR, OUTPUT_DIR
from vde.pipeline import MODEL_FILE, dev_arrays, load_engine, load_oof, publish
from vde.resources import Resources


def evaluate(ablations: bool = False) -> dict:
    config = load_config()
    res = Resources.load(config)
    model = json.loads((ARTIFACT_DIR / MODEL_FILE).read_text(encoding="utf-8"))
    cv = load_oof()
    summary = summarize(cv, config)
    calib = calibration_comparison(cv)
    _, _, per_q = dev_arrays(cv)
    weights, shift_info = _shift(res, model, per_q, config)
    tau = model["threshold"]
    decisions = publish(config=config)
    rows = with_slice(cv.rows, config["decision"].get("slice", "none"), config["decision"]["min_branch_mass"])
    fc = forecast(decisions, res, rows, tau, weights, config["forecast"]["new_import_n_scale"],
                  ece=summary["ece"], target=config["decision"]["target_precision"])
    report = {"cv": summary, "calibration": calib, "shift": shift_info, "forecast": fc,
              "threshold": tau, "model_threshold_detail": model["threshold_detail"]}
    if ablations:
        table = run_ablations(config, model["dev_ids"])
        table.to_csv(OUTPUT_DIR / "ablations.csv", index=False)
        report["ablations"] = table.to_dict("records")
    elif (OUTPUT_DIR / "ablations.csv").exists():
        report["ablations"] = pd.read_csv(OUTPUT_DIR / "ablations.csv").to_dict("records")
    report["repeat_spread"] = _repeat_spread(cv)
    report["top_errors"] = _top_errors(per_q, 20)
    _figures(cv, calib, OUTPUT_DIR)
    (OUTPUT_DIR / "previsao.txt").write_text(render(fc, _explanation(report, model)), encoding="utf-8")
    save_json(OUTPUT_DIR / "evaluation.json", report)
    return report


def _shift(res: Resources, model: dict, per_q: pd.DataFrame, config: dict):
    new_ids = [q for q in res.question_ids if res.parsed[q].is_new_import]
    old_ids = [q for q in res.question_ids if not res.parsed[q].is_new_import]
    # The final model, loaded exactly as publish loads it, so the numbers match publicacao.csv.
    engine = load_engine(res, model, config)
    new_feats = pd.DataFrame([p.features | {"conf": p.confidence} for p in engine.predict(new_ids)])
    old_unlabeled = [q for q in old_ids if q not in res.gold]
    old_feats = pd.DataFrame([p.features | {"conf": p.confidence} for p in engine.predict(old_unlabeled)])
    dom = shift.domain_weights(per_q, new_feats)
    info = {
        "domain_auc": dom["auc"],
        "effective_n_after_weighting": dom["effective_n"],
        "mean_confidence": {"labeled_oof": float(per_q.conf.mean()),
                            "bank_unlabeled": float(old_feats.conf.mean()),
                            "new_import": float(new_feats.conf.mean())},
        "share_conf_at_least_095": {"bank_unlabeled": float((old_feats.conf >= 0.95).mean()),
                                    "new_import": float((new_feats.conf >= 0.95).mean())},
        "labeled_from_new_import": int(sum(res.parsed[q].is_new_import for q in res.gold)),
        "feature_means": {"labeled_oof": per_q[["agreement", "top_cosine", "segment_dispersion"]].mean().to_dict(),
                          "new_import": new_feats[["agreement", "top_cosine", "segment_dispersion"]].mean().to_dict()},
        "description": shift.describe(res, old_ids, new_ids),
    }
    save_json(OUTPUT_DIR / "shift_report.json", info)
    weights = dict(zip(per_q.qid, dom["weights"]))
    return weights, info


def _repeat_spread(cv) -> dict:
    """Standard deviation of the hit rate between CV repeats: the noise floor of the ablation table."""
    per_rep = cv.rows.groupby("rep").hit.mean()
    return {"repeats": int(len(per_rep)), "mean": float(per_rep.mean()), "std": float(per_rep.std(ddof=1))}


def _top_errors(per_q: pd.DataFrame, k: int) -> list[dict]:
    """Misses among the k most confident labeled questions, computed instead of written by hand."""
    top = per_q.sort_values("conf", ascending=False, kind="stable").head(k)
    misses = top[~top.hit.astype(bool)]
    return [{"qid": r.qid, "conf": float(r.conf), "is_out": bool(r.is_out)} for r in misses.itertuples(index=False)]


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
    ax = axes[1]
    for i, (_, rep) in enumerate(cv.rows.groupby("rep")):
        order = np.argsort(-rep.conf.to_numpy(), kind="stable")
        hits = rep.hit.to_numpy().astype(float)[order]
        k = np.arange(1, len(hits) + 1)
        ax.plot(k / len(hits), np.cumsum(hits) / k, color="#1f5fa8", lw=0.8, alpha=0.35,
                label="uma repetição da CV" if i == 0 else None)
    ax.axhline(0.95, color="#b22222", lw=1, ls="--", label="régua 95%")
    ax.set_xlabel("cobertura (fração publicada)")
    ax.set_ylabel("precisão nas publicadas")
    ax.set_title("Precisão × cobertura (OOF, por repetição)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "confiabilidade.png", dpi=150)
    plt.close(fig)


def _fmt(value) -> str:
    return "indefinida" if value is None else f"{value:.2f}"


def _explanation(report: dict, model: dict) -> list[str]:
    cv, fc, th, sh = report["cv"], report["forecast"], report["model_threshold_detail"], report["shift"]
    cfg = load_config()
    dec = cfg["decision"]
    nested = cv["threshold_cv"]
    search = cv["threshold_cv_by_rule"]["search"]
    bank, new = fc["by_source"]["bank"], fc["by_source"]["new_import"]
    ev = fc["labeled_evidence"]
    sig = fc["calibration_sigma_sources"]
    pac = cv["precision_at_coverage"]
    run80 = min_run_to_certify(dec["target_precision"], 0.80)
    top_errors = report.get("top_errors", [])
    errors_text = ", ".join(f"{e['qid']} ({'fora na folha' if e['is_out'] else 'tema errado'})" for e in top_errors)

    def iv(x):
        return f"{x[0]:.2f}-{x[1]:.2f}"

    return [
        "Como chegamos nestes números",
        "",
        f"1. Cobertura: lida direto do publicacao.csv ({fc['published']} publicadas de {fc['evaluated']} questões "
        f"fora da folha-verdade: {bank['published']} do banco e {new['published']} da importação nova). Não é "
        "estimativa, por isso o intervalo é degenerado.",
        f"2. Regra de publicação, fixada antes de olhar a curva: publicar quando a confiança calibrada for >= "
        f"{dec['target_precision']:.2f}, no nível 1 da árvore (tema, não subtema), e só com evidência de texto e "
        f"massa do ramo >= {dec['min_branch_mass']:.2f}. A confiança é uma logística treinada fora da amostra; "
        f"ECE {cv['ece']:.3f} nas linhas da CV. Uma recalibração de Platt sobre essas linhas foi testada e "
        "descartada: não melhora o ECE e corta a cobertura pela metade.",
        f"3. Evidência: {cv['questions']} questões rotuladas, CV de 10 folds x {cfg['evaluation']['repeats']} "
        f"repetições. Em cada repetição (um modelo implantável), a regra publica em média "
        f"{ev['published_per_repeat']:.1f} rotuladas, com {ev['errors_per_repeat']:.2f} erros. São "
        f"{ev['questions']} questões distintas; cada uma conta uma vez, ponderada pela fração de repetições em "
        "que publica (tamanho efetivo de Kish), nunca como 20 linhas independentes.",
        f"4. Validação aninhada (regra aplicada em 1/5 das questões, 10 vezes): precisão "
        f"{_fmt(nested['precision'])}, limite inferior de 90% (Clopper-Pearson) {nested['cp_lower_90']:.2f}. Para "
        f"comparação, escolher o limiar olhando a curva daria {_fmt(search['precision'])}: é o otimismo da seleção, "
        "e por isso a regra é fixa.",
        f"5. Precisão prevista: a confiança calibrada de cada publicada é a sua chance de estar certa, então a "
        f"precisão esperada é a confiança média das publicadas ({_fmt(fc['precision_from_calibrated_confidence'])}). "
        f"A incerteza tem duas partes declaradas: um desvio de calibração comum a cada fonte, normal, e o ruído "
        f"binomial de cada questão. O desvio é o maior entre o ECE da CV ({sig['global_ece']:.3f}, medido em toda a "
        f"faixa de confiança) e o desvio-padrão da taxa de acerto das rotuladas que a regra publica "
        f"({sig['published_range_sd']:.3f}): são os únicos rótulos dentro da faixa publicada. Como a chance de "
        f"acerto não passa de 1, o desvio para cima é cortado e a média simulada fica abaixo da confiança média. "
        f"Banco: {bank['published']} publicadas, confiança "
        f"média {_fmt(bank['mean_confidence_of_published'])}, desvio {bank['calibration_sigma']:.3f}: "
        f"{bank['precision']:.2f} ({iv(bank['precision_interval'])}).",
        f"6. Importação nova: nenhum dos 100 rótulos vem dela ({sh['labeled_from_new_import']} de "
        f"{cv['questions']}), e o classificador de domínio quase não a separa do banco (AUC {sh['domain_auc']:.2f}). "
        f"O desvio de calibração é multiplicado por raiz de 2 ({new['calibration_sigma']:.3f}). {new['published']} "
        f"publicadas, confiança média {_fmt(new['mean_confidence_of_published'])}: {new['precision']:.2f} "
        f"({iv(new['precision_interval'])}). Ela publica mais ({new['coverage']:.0%} contra {bank['coverage']:.0%} "
        "no banco) sem rótulo para conferir: é a maior fonte de risco.",
        f"7. Combinado: {fc['precision']:.2f} ({iv(fc['precision_interval'])}, 90%); chance de ficar abaixo de "
        f"0,95: {fc['probability_precision_below_target']:.0%}. Checagem com os rótulos: as rotuladas que a regra "
        f"publicaria acertam {_fmt(bank['labeled_check']['rate'])}; com tamanho efetivo {bank['labeled_check']['effective_n']:.1f} "
        f"e prior de Jeffreys, isso dá {bank['labeled_check']['mean']:.2f} ({iv(bank['labeled_check']['interval'])}). "
        "É coerente com a previsão, mas largo demais para ser ela: com tão poucas questões, o prior puxa a média "
        "para baixo mesmo sem nenhum erro.",
        f"8. Topo da fila, por repetição: os 10% mais confiantes acertam {pac['0.10']:.2f} e os 20% acertam "
        f"{pac['0.20']:.2f}. Erros entre os 20 mais confiantes (confiança média entre repetições): "
        f"{errors_text or 'nenhum'}.",
        f"9. Por que o limiar fica em {dec['target_precision']:.2f}: baixar o limiar publica mais, mas a "
        "confiança média das publicadas encosta em 0,95 por volta de 0,90, e escolher o limiar pelos rótulos deu "
        f"precisão aninhada {_fmt(search['precision'])}. Na operação, cerca de {run80} confirmações seguidas do "
        "professor no topo da fila certificam 95% com 80% de confiança.",
    ]
