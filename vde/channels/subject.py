"""Weak evidence from the origin subject (direction 5). Never the answer, never decides level 2.

With learn_confusion enabled, the channel also learns P(gold branch | subject key) from the fold
(experiment E8). That table is learned on the bank format only and may not transfer to the new import.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from vde.channels.interfaces import Channel, Scores, pool_canonical, top_normalized
from vde.data.loader import GoldLabel
from vde.knowledge.subject_map import subject_key


class SubjectChannel(Channel):
    name = "subject"

    def __init__(self, res):
        super().__init__(res)
        self.learn_confusion = bool(self.config.get("learn_confusion", False))
        # The subject map itself never sees a label; only the E8 confusion table does.
        self.label_dependent = self.learn_confusion
        self.confusion: dict[str, Counter] = {}
        self.labeled: dict[str, GoldLabel] = {}

    def fit(self, labeled: dict[str, GoldLabel]) -> None:
        self.labeled = dict(labeled)
        if not self.learn_confusion:
            return
        table: dict[str, Counter] = defaultdict(Counter)
        tree = self.res.tree
        for qid, gold in labeled.items():
            key = subject_key(self.res.parsed[qid].origin_subject)
            if key and gold.central is not None and gold.central in tree:
                table[key][tree.level1_of(gold.central)] += 1
        self.confusion = dict(table)

    def score(self, qid: str) -> Scores:
        key = subject_key(self.res.parsed[qid].origin_subject)
        raw: Scores = {}
        for tid, value in self.res.subjects.get(key, []):
            raw[tid] = max(raw.get(tid, 0.0), value)
        if self.learn_confusion and key in self.confusion:
            raw = self._add_confusion(qid, key, raw)
        return top_normalized(pool_canonical(self.res, raw))

    def _add_confusion(self, qid: str, key: str, raw: Scores) -> Scores:
        """Leave one out: a training question never counts its own label."""
        counts = Counter(self.confusion[key])
        own = self.labeled.get(qid)
        if own is not None and own.central is not None and own.central in self.res.tree:
            counts[self.res.tree.level1_of(own.central)] -= 1
        total = sum(c for c in counts.values() if c > 0)
        for branch, count in counts.items():
            if count > 0:
                raw[branch] = raw.get(branch, 0.0) + count / total
        return raw
