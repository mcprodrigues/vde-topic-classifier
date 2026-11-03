"""Read the question in parts (direction 4): which level 1 branch does each segment point to?

The output is not a channel score but features: how much the segments agree with the chosen branch,
how dispersed they are (a patchwork question touches many topics), and candidate extras.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from vde.channels.lexical import LexicalChannel
from vde.channels.prototypes import PrototypeChannel
from vde.resources import Resources

LEXICAL_SHARE = 0.5


@dataclass(frozen=True)
class SegmentVote:
    key: str
    kind: str
    branch: int
    strength: float


class SegmentAnalyzer:
    def __init__(self, res: Resources, lexical: LexicalChannel, prototypes: PrototypeChannel):
        self.res = res
        self.lexical = lexical
        self.prototypes = prototypes

    def votes(self, qid: str) -> tuple[SegmentVote, ...]:
        return self._votes_cached(qid)

    def support(self, qid: str, branch: int) -> float:
        """Share of informative segments that point to `branch` (1.0 when there is a single segment)."""
        votes = self.votes(qid)
        if len(votes) <= 1:
            return 1.0
        return sum(v.branch == branch for v in votes) / len(votes)

    def dispersion(self, qid: str) -> float:
        """Distinct branches among segments divided by the number of segments (0 for one segment)."""
        votes = self.votes(qid)
        if len(votes) <= 1:
            return 0.0
        return (len({v.branch for v in votes}) - 1) / (len(votes) - 1)

    def branches(self, qid: str) -> dict[int, int]:
        counts: dict[int, int] = {}
        for vote in self.votes(qid):
            counts[vote.branch] = counts.get(vote.branch, 0) + 1
        return counts

    @lru_cache(maxsize=None)
    def _votes_cached(self, qid: str) -> tuple[SegmentVote, ...]:
        parsed = self.res.parsed[qid]
        out = []
        for seg in parsed.segments():
            scores = self._segment_scores(qid, seg.key, seg.text)
            if not scores:
                continue
            best = max(scores, key=scores.get)
            out.append(SegmentVote(seg.key, seg.kind, self.res.tree.level1_of(best), scores[best]))
        return tuple(out)

    def _segment_scores(self, qid: str, seg_key: str, text: str) -> dict[int, float]:
        proto = self.prototypes.score_segment(qid, seg_key)
        lex = self.lexical.score_text(text)
        merged = {t: (1 - LEXICAL_SHARE) * proto.get(t, 0.0) + LEXICAL_SHARE * lex.get(t, 0.0)
                  for t in set(proto) | set(lex)}
        return {t: s for t, s in merged.items() if s > 0}
