"""Meta model over the evidence (direction 9). Two small, class agnostic models (ADR-9).

1. CandidateScorer: logit(v) = sum_c w_c * f_c(v), softmax over candidates. The weights are learned by
   maximizing the probability mass that falls on nodes that count as a hit under the case rule, so
   the 100 labels teach how much to trust each channel, never which topics exist.
2. CorrectnessModel: logistic regression from question level features to P(hit | chosen node).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

FLOOR = 1e-3
LOG_FLOOR = float(np.log(FLOOR))

FEATURES = [
    "branch_logit", "margin", "agreement", "special_law_hit", "code_article_hit", "anchor_conflict",
    "knn_out_share", "segment_support", "segment_dispersion", "true_false", "assertions", "negated",
    "log_length", "top_cosine", "descended", "bin_branch", "duplicate_hit",
]


@dataclass
class Design:
    nodes: list[int]
    features: np.ndarray
    target: np.ndarray | None = None


class CandidateScorer:
    def __init__(self, channel_names: list[str], l2: float = 0.01, max_weight: float = 20.0,
                 transform: str = "log"):
        self.channel_names = list(channel_names)
        self.transform = transform
        self.l2 = l2
        self.max_weight = max_weight
        self.weights = np.ones(len(self.channel_names))

    def design(self, table: dict[str, dict[int, float]], tree, proposing: set[str],
               correct: set[int] | None = None) -> Design:
        nodes = self._candidates(table, tree, proposing)
        feats = np.zeros((len(nodes), len(self.channel_names)))
        for j, name in enumerate(self.channel_names):
            scores = table.get(name, {})
            for i, node in enumerate(nodes):
                feats[i, j] = self._transform(scores.get(node, 0.0))
        target = None if correct is None else np.array([n in correct for n in nodes], dtype=bool)
        return Design(nodes, feats, target)

    def fit(self, designs: list[Design]) -> dict:
        usable = [d for d in designs if d.target is not None and d.target.any() and len(d.nodes)]
        if not usable:
            return {"usable": 0}
        result = minimize(self._loss, self.weights, args=(usable,), jac=True, method="L-BFGS-B",
                          bounds=[(0.0, self.max_weight)] * len(self.weights))
        self.weights = result.x
        return {"usable": len(usable), "unreachable": len(designs) - len(usable), "loss": float(result.fun)}

    def distribution(self, design: Design) -> dict[int, float]:
        if not design.nodes:
            return {}
        logits = design.features @ self.weights
        probs = np.exp(logits - logsumexp(logits))
        return {n: float(p) for n, p in zip(design.nodes, probs)}

    def to_dict(self) -> dict:
        return {"channels": self.channel_names, "weights": [float(w) for w in self.weights],
                "transform": self.transform}

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateScorer":
        scorer = cls(data["channels"], transform=data.get("transform", "log"))
        scorer.weights = np.array(data["weights"], dtype=float)
        return scorer

    def _transform(self, value: float) -> float:
        if value <= 0:
            return 0.0
        if self.transform == "linear":
            return value
        return float(np.log(max(value, FLOOR)) - LOG_FLOOR)

    def _candidates(self, table, tree, proposing: set[str]) -> list[int]:
        nodes: set[int] = set()
        for name in proposing:
            nodes.update(table.get(name, {}))
        nodes |= {tree.level1_of(n) for n in nodes}
        return sorted(nodes)

    def _loss(self, w: np.ndarray, designs: list[Design]) -> tuple[float, np.ndarray]:
        loss = self.l2 * float(w @ w)
        grad = 2 * self.l2 * w
        for d in designs:
            logits = d.features @ w
            log_z = logsumexp(logits)
            log_hit = logsumexp(logits[d.target])
            p = np.exp(logits - log_z)
            p_hit = np.exp(logits[d.target] - log_hit)
            loss -= log_hit - log_z
            grad -= p_hit @ d.features[d.target] - p @ d.features
        return loss, grad


class CorrectnessModel:
    """Standardized logistic regression, stored as plain numbers so publish needs no pickle."""

    def __init__(self, c: float = 0.3):
        self.c = c
        self.mean = np.zeros(len(FEATURES))
        self.scale = np.ones(len(FEATURES))
        self.coef = np.zeros(len(FEATURES))
        self.intercept = 0.0

    def fit(self, rows: list[dict], y: list[bool]) -> None:
        from sklearn.linear_model import LogisticRegression

        x = self._matrix(rows)
        self.mean = x.mean(axis=0)
        self.scale = np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0)
        labels = np.asarray(y, dtype=int)
        if labels.min() == labels.max():
            self.coef = np.zeros(len(FEATURES))
            self.intercept = float(np.log((labels.mean() + 1e-3) / (1 - labels.mean() + 1e-3)))
            return
        model = LogisticRegression(C=self.c, max_iter=2000)
        model.fit((x - self.mean) / self.scale, labels)
        self.coef = model.coef_[0]
        self.intercept = float(model.intercept_[0])

    def predict(self, rows: list[dict]) -> np.ndarray:
        z = ((self._matrix(rows) - self.mean) / self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> dict:
        return {"features": FEATURES, "c": self.c, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept}

    @classmethod
    def from_dict(cls, data: dict) -> "CorrectnessModel":
        if data["features"] != FEATURES:
            raise ValueError("model.json was trained with a different feature list")
        model = cls(data["c"])
        model.mean = np.array(data["mean"])
        model.scale = np.array(data["scale"])
        model.coef = np.array(data["coef"])
        model.intercept = float(data["intercept"])
        return model

    def _matrix(self, rows: list[dict]) -> np.ndarray:
        return np.array([[float(r[f]) for f in FEATURES] for r in rows], dtype=float).reshape(-1, len(FEATURES))
