"""previsao.txt: predicted precision and coverage on the 559 evaluated questions, with intervals.

Coverage is not an estimate: it is read from publicacao.csv. The uncertainty is in precision.

Point estimate: the calibrated confidence of each published question is its chance of being right, so the
expected precision is the mean confidence of the published set. That is what the model was fitted to
mean, and its calibration was measured out of fold (ECE).

Uncertainty, per source, from two declared parts:
  * calibration shift: every published confidence of a source moves by the same delta ~ Normal(0, sigma).
    For the bank, sigma is the larger of two numbers: the CV expected calibration error (measured over the
    whole confidence range) and the posterior standard deviation of the hit rate of the labeled questions
    the rule publishes (the only labels inside the published range, about ten effective questions). The
    global ECE alone would claim a calibration that was never checked where the rule publishes. For the
    new import, sigma is divided by sqrt(new_import_n_scale), because none of the 100 labels comes from it.
  * binomial noise: each published question is then right with probability conf + delta.
The interval is a 90% simulation interval.

Cross check: the labeled out of fold rows that would be published (conf >= tau), each labeled question
counted once (Kish effective size, Jeffreys prior). With about ten effective questions this check is
wide, so it is reported, not used as the forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIMULATIONS = 20000
INTERVAL = 0.90
JEFFREYS = 0.5


def forecast(decisions, res, rows: pd.DataFrame, tau: float, weights: dict[str, float] | None,
             n_scale: float = 0.5, ece: float = 0.05, target: float = 0.95, seed: int = 0) -> dict:
    evaluated = [d for d in decisions if d.id not in res.gold]
    groups = {
        "bank": [d for d in evaluated if not res.parsed[d.id].is_new_import],
        "new_import": [d for d in evaluated if res.parsed[d.id].is_new_import],
    }
    published = {g: [d for d in ds if d.decision == "publicar"] for g, ds in groups.items()}
    total_pub = sum(len(v) for v in published.values())
    coverage = total_pub / len(evaluated) if evaluated else 0.0
    evidence = labeled_evidence(rows, tau)
    labeled = {"bank": effective_beta(evidence, None, 1.0), "new_import": effective_beta(evidence, weights, n_scale)}
    region_sd = beta_sd(labeled["bank"])
    bank_sigma = max(ece, region_sd)
    sigma = {"bank": bank_sigma, "new_import": bank_sigma / np.sqrt(n_scale)}
    rng = np.random.default_rng(seed)
    lo, hi = (1 - INTERVAL) / 2, 1 - (1 - INTERVAL) / 2
    simulated = np.zeros(SIMULATIONS)
    by_source = {}
    for group in groups:
        conf = np.array([d.confidence for d in published[group]], dtype=float)
        hits, prec = simulate_hits(conf, sigma[group], rng)
        simulated += hits
        by_source[group] = {
            "published": len(conf),
            "coverage": len(conf) / len(groups[group]) if groups[group] else 0.0,
            "precision": float(np.mean(prec)),
            "precision_interval": (float(np.quantile(prec, lo)), float(np.quantile(prec, hi))),
            "mean_confidence_of_published": float(conf.mean()) if len(conf) else None,
            "calibration_sigma": float(sigma[group]),
            "labeled_check": labeled[group] | _beta_interval(labeled[group], rng, lo, hi),
        }
    if total_pub:
        precision = simulated / total_pub
    else:
        precision = rng.beta(labeled["bank"]["a"], labeled["bank"]["b"], SIMULATIONS)
    calibrated = float(np.mean([d.confidence for ds in published.values() for d in ds])) if total_pub else None
    return {
        "method": "calibrated_confidence",
        "evaluated": len(evaluated),
        "published": total_pub,
        "published_by_source": {g: len(v) for g, v in published.items()},
        "coverage": coverage,
        "coverage_interval": (coverage, coverage),
        "precision": float(np.mean(precision)),
        "precision_interval": (float(np.quantile(precision, lo)), float(np.quantile(precision, hi))),
        "probability_precision_below_target": float(np.mean(precision < target)),
        "precision_from_calibrated_confidence": calibrated,
        "calibration_sigma_sources": {"global_ece": float(ece), "published_range_sd": float(region_sd)},
        "by_source": by_source,
        "labeled_evidence": {"questions": int(len(evidence)), "published_per_repeat": float(evidence.freq.sum()),
                             "errors_per_repeat": float((evidence.freq * (1 - evidence.hit)).sum())},
    }


def simulate_hits(conf: np.ndarray, sigma: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Hits of the published set under a shared calibration shift, and the matching precision."""
    if not len(conf):
        return np.zeros(SIMULATIONS), np.full(SIMULATIONS, np.nan)
    delta = rng.normal(0.0, sigma, SIMULATIONS)
    p = np.clip(conf[None, :] + delta[:, None], 0.0, 1.0)
    hits = (rng.random(p.shape) < p).sum(axis=1).astype(float)
    return hits, hits / len(conf)


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


def beta_sd(st: dict) -> float:
    """Standard deviation of a Beta(a, b) posterior."""
    a, b = st["a"], st["b"]
    return float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))


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


def _beta_interval(st: dict, rng, lo: float, hi: float) -> dict:
    draws = rng.beta(st["a"], st["b"], SIMULATIONS)
    return {"mean": float(draws.mean()), "interval": (float(np.quantile(draws, lo)), float(np.quantile(draws, hi)))}
