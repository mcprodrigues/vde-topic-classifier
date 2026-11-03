"""Bin detector as a negative prior (direction 5): topics that absorbed many questions start lower.

The aggregation learns a non negative weight for it; a weight near zero means the prior adds nothing.
"""

from __future__ import annotations

import math
from functools import cached_property

from vde.channels.interfaces import Channel, Scores


class BinPriorChannel(Channel):
    name = "bin_prior"
    proposes_candidates = False

    def score(self, qid: str) -> Scores:
        return self._prior

    @cached_property
    def _prior(self) -> Scores:
        tree = self.res.tree
        return {t: 1.0 / (1.0 + math.log1p(tree.source_count(t) / 50.0)) for t in tree.candidates()}
