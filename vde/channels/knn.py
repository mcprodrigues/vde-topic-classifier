"""Nearest labeled neighbors in embedding space. Refitted per fold; a question never sees itself.

Similarities are computed against the labeled questions only (N x L, not N x N), so memory grows with
the number of labels, not with the size of the weekly import.
"""

from __future__ import annotations

import numpy as np

from vde.channels.interfaces import Channel, Scores, top_normalized
from vde.data.loader import GoldLabel
from vde.knowledge.embeddings import question_key


class KnnChannel(Channel):
    name = "knn"
    label_dependent = True
    embedding_based = True

    def __init__(self, res):
        super().__init__(res)
        self.k = int(self.config.get("k", 7))
        self.temperature = float(self.config.get("temperature", 0.05))
        self.extra_weight = float(self.config.get("extra_weight", 0.5))
        self.labeled: dict[str, GoldLabel] = {}
        self.labeled_ids: list[str] = []
        self.labeled_index: dict[str, int] = {}
        self.labeled_matrix = np.zeros((0, 0), dtype=np.float32)

    def fit(self, labeled: dict[str, GoldLabel]) -> None:
        self.labeled = dict(labeled)
        self.labeled_ids = list(labeled)
        self.labeled_index = {q: i for i, q in enumerate(self.labeled_ids)}
        self.labeled_matrix = self.res.embeddings.matrix([question_key(q) for q in self.labeled_ids])

    def score(self, qid: str) -> Scores:
        votes, _ = self._votes(qid)
        return top_normalized(votes)

    def out_share(self, qid: str) -> float:
        votes, out_mass = self._votes(qid)
        total = sum(votes.values()) + out_mass
        return out_mass / total if total > 0 else 0.0

    def _votes(self, qid: str) -> tuple[Scores, float]:
        if not self.labeled_ids:
            return {}, 0.0
        sims = self.labeled_matrix @ self.res.embeddings.get(question_key(qid))
        own = self.labeled_index.get(qid)
        if own is not None:
            sims[own] = -np.inf
        order = [i for i in np.argsort(-sims, kind="stable")[: self.k] if np.isfinite(sims[i])]
        if not order:
            return {}, 0.0
        best = sims[order[0]]
        votes: Scores = {}
        out_mass = 0.0
        tree = self.res.tree
        for i in order:
            weight = float(np.exp((sims[i] - best) / self.temperature))
            gold = self.labeled[self.labeled_ids[i]]
            if gold.is_out or gold.central is None:
                out_mass += weight
                continue
            for tid, w in [(gold.central, 1.0)] + [(e, self.extra_weight) for e in gold.extras]:
                if tid in tree and tree.level(tree.canonical(tid)) > 0:
                    canon = tree.canonical(tid)
                    votes[canon] = votes.get(canon, 0.0) + weight * w
        return votes, out_mass
