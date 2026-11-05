"""BM25 between the question and the topic cards (the lexical half of the baseline)."""

from __future__ import annotations

import math
from collections import Counter

from vde.channels.interfaces import Channel, Scores, cached_by_qid, top_normalized
from vde.data.cleaning import tokenize


class LexicalChannel(Channel):
    name = "lexical"
    field = "text"

    def __init__(self, res):
        super().__init__(res)
        self.k1 = float(self.config.get("k1", 1.2))
        self.b = float(self.config.get("b", 0.75))
        self.top_k = int(self.config.get("top_k", 40))
        self.topic_ids = sorted(res.cards)
        docs = [terms(getattr(res.cards[t], self.field)) for t in self.topic_ids]
        self.doc_tf = [Counter(d) for d in docs]
        self.doc_len = [len(d) for d in docs]
        self.avg_len = sum(self.doc_len) / max(len(docs), 1)
        df = Counter(term for d in self.doc_tf for term in d)
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def score(self, qid: str) -> Scores:
        return self._score_cached(qid)

    def score_text(self, text: str) -> Scores:
        return top_normalized(self._bm25(terms(text)), self.top_k)

    @cached_by_qid
    def _score_cached(self, qid: str) -> Scores:
        return self.score_text(self.res.parsed[qid].full_text)

    def _bm25(self, query: list[str]) -> Scores:
        q_terms = Counter(t for t in query if t in self.idf)
        scores: Scores = {}
        for idx, tf in enumerate(self.doc_tf):
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avg_len)
            total = 0.0
            for term in q_terms:
                f = tf.get(term)
                if f:
                    total += self.idf[term] * f * (self.k1 + 1) / (f + norm)
            if total > 0:
                scores[self.topic_ids[idx]] = total
        return scores


class TitleLexicalChannel(LexicalChannel):
    """Same BM25 over path and title only: the zero knowledge baseline E0."""

    name = "lexical_titles"
    field = "title_text"


def terms(text: str) -> list[str]:
    """Unigrams plus adjacent bigrams, so that 'prisao preventiva' outweighs two loose words."""
    unigrams = tokenize(text)
    return unigrams + [f"{a}_{b}" for a, b in zip(unigrams, unigrams[1:])]
