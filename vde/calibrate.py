"""Calibration and the publish threshold chosen by a lower bound, not by the mean (ADR-6, ADR-10)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import beta, binom


@dataclass(frozen=True)
class ThresholdChoice:
    tau: float
    published: int
    hits: float
    point_precision: float
    posterior: float
    coverage: float
    effective_n: float

    def to_dict(self) -> dict:
        return asdict(self)


def clopper_pearson_lower(hits: int, n: int, confidence: float = 0.95) -> float:
    """One sided lower bound of a binomial proportion."""
    if n == 0 or hits == 0:
        return 0.0
    return float(beta.ppf(1 - confidence, hits, n - hits + 1))


def posterior_at_least(hits: float, n: float, target: float) -> float:
    """P(precision >= target) under a uniform prior, Beta(1 + hits, 1 + n - hits) posterior."""
    return float(beta.sf(target, 1 + hits, 1 + n - hits))


def min_run_to_certify(target: float = 0.95, level: float = 0.80) -> int:
    """Smallest run of hits with no error that gives P(precision >= target) >= level under a uniform prior.

    The posterior after n hits and 0 errors is Beta(1 + n, 1), so P(p >= target) = 1 - target ** (n + 1).
    """
    n = 0
    while 1 - target ** (n + 1) < level:
        n += 1
    return n


def choose_threshold_per_repeat(conf: np.ndarray, correct: np.ndarray, reps: np.ndarray,
                                eligible: np.ndarray | None = None, target: float = 0.95,
                                min_point: float = 0.97, min_published: float = 5.0) -> ThresholdChoice:
    """Search rule: walk down from the most confident row and stop at the first violation.

    Every CV repeat is one deployable model, so rows are never averaged across repeats: precision is
    pooled over repeats and the published count is the mean per repeat. Once at least `min_published`
    rows per repeat are above the threshold, the walk stops at the first point whose precision falls
    below `min_point` (fixed sequence, as in Learn then Test), so a lucky point after a dip is never taken.
    """
    conf, correct, ok = _prepare(conf, correct, eligible)
    n_reps = max(len(np.unique(reps)), 1)
    order = np.argsort(-np.where(ok, conf, -np.inf), kind="stable")
    sorted_conf = np.where(ok, conf, -np.inf)[order]
    pub = np.cumsum(ok[order])
    hits = np.cumsum((correct * ok)[order])
    best = _empty_choice()
    for i in range(len(conf)):
        tau = sorted_conf[i]
        if not np.isfinite(tau) or (i + 1 < len(conf) and sorted_conf[i + 1] == tau):
            continue
        if pub[i] / n_reps < min_published:
            continue
        if hits[i] / pub[i] < min_point:
            break
        best = threshold_stats(conf, correct, reps, ok, float(tau), target)
    return best


def threshold_stats(conf: np.ndarray, correct: np.ndarray, reps: np.ndarray, eligible: np.ndarray | None,
                    tau: float, target: float = 0.95) -> ThresholdChoice:
    """Published count per repeat, pooled precision and the Beta posterior at a given threshold."""
    conf, correct, ok = _prepare(conf, correct, eligible)
    n_reps = max(len(np.unique(reps)), 1)
    mask = ok & (conf >= tau)
    per_rep = mask.sum() / n_reps
    if per_rep == 0:
        return ThresholdChoice(float(tau), 0, 0.0, 1.0, 0.0, 0.0, 0.0)
    precision = float(correct[mask].sum() / mask.sum())
    return ThresholdChoice(float(tau), int(round(per_rep)), precision * per_rep, precision,
                           posterior_at_least(precision * per_rep, per_rep, target),
                           float(per_rep / (len(conf) / n_reps)), float(per_rep))


def ltt_threshold(conf: np.ndarray, correct: np.ndarray, alpha: float = 0.05, delta: float = 0.10) -> float:
    """Learn then Test with fixed sequence testing: thresholds certified to have risk <= alpha w.p. 1 - delta."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    certified = 1.01
    for tau in np.unique(conf)[::-1]:
        mask = conf >= tau
        n = int(mask.sum())
        errors = int(n - correct[mask].sum())
        p_value = float(binom.cdf(errors, n, alpha))
        if p_value > delta:
            break
        certified = float(tau)
    return certified


class Platt:
    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = a, b

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "Platt":
        from sklearn.linear_model import LogisticRegression

        logits = _logit(np.asarray(scores)).reshape(-1, 1)
        model = LogisticRegression(C=1e6, max_iter=1000).fit(logits, np.asarray(y, dtype=int))
        self.a, self.b = float(model.coef_[0, 0]), float(model.intercept_[0])
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * _logit(np.asarray(scores)) + self.b)))

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b}


def reliability_table(conf: np.ndarray, y: np.ndarray, bins: int = 5) -> list[dict]:
    """Equal frequency bins: mean confidence against observed hit rate."""
    order = np.argsort(conf)
    out = []
    for chunk in np.array_split(order, bins):
        if len(chunk):
            out.append({"n": int(len(chunk)), "mean_conf": float(np.mean(conf[chunk])),
                        "hit_rate": float(np.mean(y[chunk]))})
    return out


def expected_calibration_error(conf: np.ndarray, y: np.ndarray, bins: int = 5) -> float:
    table = reliability_table(np.asarray(conf), np.asarray(y, dtype=float), bins)
    total = sum(r["n"] for r in table)
    return float(sum(r["n"] * abs(r["mean_conf"] - r["hit_rate"]) for r in table) / max(total, 1))


def brier(conf: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(conf) - np.asarray(y, dtype=float)) ** 2))


def _prepare(conf, correct, eligible):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    ok = np.ones(len(conf), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    return conf, correct, ok


def _empty_choice() -> ThresholdChoice:
    return ThresholdChoice(1.01, 0, 0.0, 1.0, 0.0, 0.0, 0.0)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
