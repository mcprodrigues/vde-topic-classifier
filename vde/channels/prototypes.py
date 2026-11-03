"""Embedding similarity between the question and each topic card (the zero shot path, direction 3)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from vde.channels.interfaces import Channel, Scores
from vde.knowledge.embeddings import card_key, question_key, segment_key


class PrototypeChannel(Channel):
    name = "prototypes"

    def __init__(self, res):
        super().__init__(res)
        self.temperature = float(self.config.get("temperature", 0.05))
        self.top_k = int(self.config.get("top_k", 40))
        self.topic_ids = sorted(res.cards)
        self.card_matrix = res.embeddings.matrix([card_key(t) for t in self.topic_ids])

    def score(self, qid: str) -> Scores:
        return self._score_cached(qid)

    def score_vector(self, vector: np.ndarray) -> Scores:
        cos = self.card_matrix @ vector
        order = np.argsort(-cos)[: self.top_k]
        best = cos[order[0]]
        return {self.topic_ids[i]: float(np.exp((cos[i] - best) / self.temperature)) for i in order}

    def score_segment(self, qid: str, seg_key: str) -> Scores:
        key = segment_key(qid, seg_key)
        if not self.res.embeddings.has(key):
            return {}
        return self.score_vector(self.res.embeddings.get(key))

    def top_cosine(self, qid: str) -> float:
        return float(np.max(self.card_matrix @ self.res.embeddings.get(question_key(qid))))

    @lru_cache(maxsize=None)
    def _score_cached(self, qid: str) -> Scores:
        return self.score_vector(self.res.embeddings.get(question_key(qid)))
