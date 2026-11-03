"""previsao.txt: predicted precision and coverage on the 559 evaluated questions, with intervals.

Coverage is not an estimate: it is read from publicacao.csv. The uncertainty is in precision, which
combines a Beta posterior per source (bank rows unweighted, new import rows importance weighted) with
the binomial noise of the realized published set.
"""

from __future__ import annotations

import numpy as np

from vde.calibrate import effective_counts

SIMULATIONS = 20000
INTERVAL = 0.90


def forecast(decisions, res, dev_conf, dev_hit, tau, weights, seed: int = 0) -> dict:
    evaluated = [d for d in decisions if d.id not in res.gold]
    groups = {
        "bank": [d for d in evaluated if not res.parsed[d.id].is_new_import],
        "new_import": [d for d in evaluated if res.parsed[d.id].is_new_import],
    }
    published = {g: [d for d in ds if d.decision == "publicar"] for g, ds in groups.items()}
    total_pub = sum(len(v) for v in published.values())
    coverage = total_pub / len(evaluated) if evaluated else 0.0
    mask = dev_conf >= tau
    stats = {
        "bank": effective_counts(dev_hit[mask], None),
        "new_import": effective_counts(dev_hit[mask], weights[mask] if weights is not None else None),
    }
    rng = np.random.default_rng(seed)
    simulated = np.zeros(SIMULATIONS)
    for group, (hits, n) in stats.items():
        k = len(published[group])
        if k == 0:
            continue
        p = rng.beta(1 + hits, 1 + max(n - hits, 0), SIMULATIONS)
        simulated += rng.binomial(k, p)
    lo, hi = (1 - INTERVAL) / 2, 1 - (1 - INTERVAL) / 2
    calibrated = float(np.mean([d.confidence for ds in published.values() for d in ds])) if total_pub else None
    if total_pub:
        precision = simulated / total_pub
        point, low, high = float(np.mean(precision)), float(np.quantile(precision, lo)), float(np.quantile(precision, hi))
    else:
        point = low = high = None
    return {
        "evaluated": len(evaluated),
        "published": total_pub,
        "published_by_source": {g: len(v) for g, v in published.items()},
        "coverage": coverage,
        "coverage_interval": (coverage, coverage),
        "precision": point,
        "precision_interval": (low, high),
        "precision_from_calibrated_confidence": calibrated,
        "dev_published_at_tau": int(mask.sum()),
        "dev_hits_at_tau": float(dev_hit[mask].sum()),
        "effective_counts": {g: {"hits": h, "n": n} for g, (h, n) in stats.items()},
    }


def render(fc: dict, extra_lines: list[str]) -> str:
    def fmt(x):
        return "nan" if x is None else f"{x:.2f}"

    lines = [
        f"precisao_prevista: {fmt(fc['precision'])}",
        f"precisao_intervalo: {fmt(fc['precision_interval'][0])}-{fmt(fc['precision_interval'][1])}",
        f"cobertura_prevista: {fmt(fc['coverage'])}",
        f"cobertura_intervalo: {fmt(fc['coverage_interval'][0])}-{fmt(fc['coverage_interval'][1])}",
        "",
    ]
    return "\n".join(lines + extra_lines) + "\n"
