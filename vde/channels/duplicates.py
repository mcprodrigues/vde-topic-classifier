"""Near duplicate propagation (direction 8), on the full text only.

The stem alone is misleading: Q0090 and Q0339 share the stem "Assinale a alternativa correta" and
carry different gold labels. Similarity is computed on stem, assertions, command and alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vde.channels.interfaces import Channel, Scores
from vde.data.cleaning import fold
from vde.data.loader import GoldLabel


@dataclass(frozen=True)
class DuplicateMatch:
    labeled_id: str
    similarity: float
    is_out: bool


class DuplicateChannel(Channel):
    name = "duplicates"
    label_dependent = True

    def __init__(self, res):
        super().__init__(res)
        self.min_similarity = float(self.config.get("min_similarity", 0.92))
        self.ids = res.question_ids
        self.row = {q: i for i, q in enumerate(self.ids)}
        texts = [fold(res.parsed[q].full_text) for q in self.ids]
        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(texts)
        self.similarity = cosine_similarity(matrix)
        np.fill_diagonal(self.similarity, 0.0)
        self.labeled: dict[str, GoldLabel] = {}

    def fit(self, labeled: dict[str, GoldLabel]) -> None:
        self.labeled = dict(labeled)
        self.labeled_ids = list(labeled)
        self.labeled_rows = np.array([self.row[q] for q in self.labeled_ids], dtype=int)

    def match(self, qid: str) -> DuplicateMatch | None:
        if not self.labeled:
            return None
        sims = self.similarity[self.row[qid], self.labeled_rows]
        best = int(np.argmax(sims))
        if sims[best] < self.min_similarity or self.labeled_ids[best] == qid:
            return None
        gold = self.labeled[self.labeled_ids[best]]
        return DuplicateMatch(self.labeled_ids[best], float(sims[best]), gold.is_out)

    def score(self, qid: str) -> Scores:
        found = self.match(qid)
        if found is None or found.is_out:
            return {}
        gold = self.labeled[found.labeled_id]
        tree = self.res.tree
        scores: Scores = {}
        for tid, w in [(gold.central, 1.0)] + [(e, 0.5) for e in gold.extras]:
            if tid in tree:
                canon = tree.canonical(tid)
                scores[canon] = max(scores.get(canon, 0.0), w)
        return scores
