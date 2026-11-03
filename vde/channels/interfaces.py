"""Common interface of the evidence channels.

A channel maps a question to non negative scores over canonical topic ids. Channels never decide:
they only feed the aggregation, which lets evaluate.py switch any of them off and measure its value.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vde.data.loader import GoldLabel
from vde.resources import Resources

Scores = dict[int, float]


class Channel(ABC):
    name: str = "base"
    # Channels whose scores depend on the gold sheet are refitted inside every cross validation fold.
    label_dependent: bool = False
    # Prior channels score every topic, so they must not add candidates to the softmax.
    proposes_candidates: bool = True

    def __init__(self, res: Resources):
        self.res = res
        self.config = res.config.get(self.name, {})

    def fit(self, labeled: dict[str, GoldLabel]) -> None:
        """Learn from the labeled questions of the current fold. Default: nothing to learn."""
        return None

    @abstractmethod
    def score(self, qid: str) -> Scores:
        """Scores over canonical topic ids. Missing ids mean no evidence."""


def top_normalized(scores: Scores, top_k: int | None = None) -> Scores:
    """Keep the top_k scores and divide by the maximum, so the best topic scores 1."""
    positive = {t: s for t, s in scores.items() if s > 0}
    if not positive:
        return {}
    ranked = sorted(positive.items(), key=lambda kv: -kv[1])
    if top_k:
        ranked = ranked[:top_k]
    best = ranked[0][1]
    return {t: s / best for t, s in ranked}


def pool_canonical(res: Resources, scores: Scores) -> Scores:
    """Move the mass of shadow nodes to their canonical twin and drop unknown or root ids."""
    pooled: Scores = {}
    for tid, value in scores.items():
        if tid not in res.tree:
            continue
        canon = res.tree.canonical(tid)
        if res.tree.level(canon) == 0:
            continue
        pooled[canon] = pooled.get(canon, 0.0) + value
    return pooled
