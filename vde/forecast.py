"""previsao.txt: predicted precision and coverage on the 559 evaluated questions, with intervals.

Coverage is not an estimate: it is read from publicacao.csv. The uncertainty is in precision.

Evidence: the labeled out of fold rows that would be published (conf >= tau, eligible). The same
question appears in many CV repeats, so each labeled question counts once, weighted by the share of
repeats in which it was published (Kish effective size), never as 20 independent rows.

  * bank: Beta posterior with a Jeffreys prior over those effective counts.
  * new import: the same rows reweighted by the domain classifier (importance weights), and the
    effective size multiplied by `new_import_n_scale` (0.5), because none of the 100 labels comes
    from the new import.

The realized published set adds binomial noise. The interval is a 90% simulation interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIMULATIONS = 20000
INTERVAL = 0.90
JEFFREYS = 0.5


def forecast(decisions, res, rows: pd.DataFrame, tau: float, weights: dict[str, float] | None,
             n_scale: float = 0.5, seed: int = 0) -> dict:
    evaluated = [d for d in decisions if d.id not in res.gold]
    groups = {
        "bank": [d for d in evaluated if not res.parsed[d.id].is_new_import],
        "new_import": [d for d in evaluated if res.parsed[d.id].is_new_import],
    }
    published = {g: [d for d in ds if d.decision == "publicar"] for g, ds in groups.items()}
    total_pub = sum(len(v) for v in published.values())
    coverage = total_pub / len(evaluated) if evaluated else 0.0
    evidence = labeled_evidence(rows, tau)
    stats = {
        "bank": effective_beta(evidence, None, 1.0),
        "new_import": effective_beta(evidence, weights, n_scale),
    }
    rng = np.random.default_rng(seed)
    lo, hi = (1 - INTERVAL) / 2, 1 - (1 - INTERVAL) / 2
    simulated = np.zeros(SIMULATIONS)
    by_source = {}
    for group, st in stats.items():
        k = len(published[group])
        p = rng.beta(st["a"], st["b"], SIMULATIONS)
        hits = rng.binomial(k, p) if k else np.zeros(SIMULATIONS)
        simulated += hits
        prec = hits / k if k else p
        by_source[group] = {
            "published": k,
            "coverage": k / len(groups[group]) if groups[group] else 0.0,
            "precision": float(np.mean(prec)),
            "precision_interval": (float(np.quantile(prec, lo)), float(np.quantile(prec, hi))),
            "mean_confidence_of_published": float(np.mean([d.confidence for d in published[group]])) if k else None,
            **st,
        }
    if total_pub:
        precision = simulated / total_pub
    else:
        precision = rng.beta(stats["bank"]["a"], stats["bank"]["b"], SIMULATIONS)
    calibrated = float(np.mean([d.confidence for ds in published.values() for d in ds])) if total_pub else None
    return {
        "evaluated": len(evaluated),
        "published": total_pub,
        "published_by_source": {g: len(v) for g, v in published.items()},
        "coverage": coverage,
        "coverage_interval": (coverage, coverage),
        "precision": float(np.mean(precision)),
        "precision_interval": (float(np.quantile(precision, lo)), float(np.quantile(precision, hi))),
        "probability_precision_below_target": float(np.mean(precision < 0.95)),
        "precision_from_calibrated_confidence": calibrated,
        "by_source": by_source,
        "labeled_evidence": {"questions": int(len(evidence)), "published_per_repeat": float(evidence.freq.sum()),
                             "errors_per_repeat": float((evidence.freq * (1 - evidence.hit)).sum())},
    }


def labeled_evidence(rows: pd.DataFrame, tau: float) -> pd.DataFrame:
    """One row per labeled question that is ever published: share of repeats published and hit rate."""
    n_reps = max(rows.rep.nunique(), 1)
    mask = rows.conf >= tau
    if "eligible" in rows:
        mask &= rows.eligible.astype(bool)
    pub = rows[mask]
    if pub.empty:
        return pd.DataFrame({"qid": [], "freq": [], "hit": []})
    grouped = pub.groupby("qid")
    return pd.DataFrame({"freq": grouped.size() / n_reps, "hit": grouped.hit.mean().astype(float)}).reset_index()


def effective_beta(evidence: pd.DataFrame, weights: dict[str, float] | None, n_scale: float) -> dict:
    """Jeffreys Beta posterior over Kish effective counts of the labeled evidence."""
    if evidence.empty:
        return {"a": JEFFREYS, "b": JEFFREYS, "effective_n": 0.0, "rate": None}
    v = evidence.freq.to_numpy(dtype=float)
    if weights is not None:
        v = v * np.array([weights.get(q, 1.0) for q in evidence.qid])
    hit = evidence.hit.to_numpy(dtype=float)
    n_eff = float(v.sum() ** 2 / (v ** 2).sum()) * n_scale
    rate = float((v * hit).sum() / v.sum())
    return {"a": JEFFREYS + rate * n_eff, "b": JEFFREYS + (1 - rate) * n_eff, "effective_n": n_eff, "rate": rate}


def render(fc: dict, extra_lines: list[str]) -> str:
    """The four header lines always carry numbers in the 0.00 format of LEIA-ME, never nan."""

    def fmt(x):
        return "0.00" if x is None or x != x else f"{x:.2f}"

    lines = [
        f"precisao_prevista: {fmt(fc['precision'])}",
        f"precisao_intervalo: {fmt(fc['precision_interval'][0])}-{fmt(fc['precision_interval'][1])}",
        f"cobertura_prevista: {fmt(fc['coverage'])}",
        f"cobertura_intervalo: {fmt(fc['coverage_interval'][0])}-{fmt(fc['coverage_interval'][1])}",
        "",
    ]
    return "\n".join(lines + extra_lines) + "\n"
