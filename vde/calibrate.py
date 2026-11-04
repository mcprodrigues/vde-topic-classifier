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


def effective_counts(correct: np.ndarray, weights: np.ndarray | None) -> tuple[float, float]:
    """Importance weighted (hits, n), rescaled to the effective sample size (Kish)."""
    if weights is None:
        return float(correct.sum()), float(len(correct))
    if weights.sum() <= 0:
        return 0.0, 0.0
    n_eff = weights.sum() ** 2 / (weights ** 2).sum()
    rate = float((weights * correct).sum() / weights.sum())
    return rate * n_eff, n_eff


def choose_threshold(conf: np.ndarray, correct: np.ndarray, weights: np.ndarray | None = None,
                     target: float = 0.95, level: float = 0.80, min_point: float = 0.97) -> ThresholdChoice:
    """Lowest threshold (largest coverage) whose posterior P(precision >= target) reaches `level`."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    best = ThresholdChoice(1.01, 0, 0.0, 1.0, 0.0, 0.0, 0.0)
    for tau in np.unique(conf)[::-1]:
        mask = conf >= tau
        w = None if weights is None else np.asarray(weights)[mask]
        hits, n = effective_counts(correct[mask], w)
        point = hits / n if n else 1.0
        post = posterior_at_least(hits, n, target)
        if post >= level and point >= min_point:
            best = ThresholdChoice(float(tau), int(mask.sum()), float(correct[mask].sum()), point, post,
                                   float(mask.mean()), n)
    return best


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


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
