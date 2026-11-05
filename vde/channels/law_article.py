"""High precision channel: deterministic extraction of cited laws and articles (direction 2).

"Lei nº 11.340/2006" goes to the topic of the law. "art. 157 do CP" goes through the code atlas.
An article without an explicit code is attributed to the nearest code or law mention, and left
unresolved when that is ambiguous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vde.channels.interfaces import Channel, Scores, cached_by_qid, pool_canonical, top_normalized
from vde.data.cleaning import fold

CODE_KEYS = frozenset({"CP", "CPP"})
RIGHT_WINDOW = 80
LEFT_WINDOW = 40

_LAW_NUMBER = re.compile(
    r"\b(decreto[\s-]*lei|lei(?:\s+federal)?(?:\s+complementar)?)\s*(?:n\s*[o.]?\s*|numero\s*)?(\d{1,2}\.?\d{3})\b")
_ARTICLES = re.compile(r"\bart(?:igo)?s?\s*\.?\s*((?:\d{1,3}(?:\s*-\s*[a-z]\b)?(?:\s*(?:,|e)\s*)?)+)")
_ONE_ARTICLE = re.compile(r"(\d{1,3})(?:\s*-\s*([a-z])\b)?")


@dataclass(frozen=True)
class Anchor:
    law_key: str
    article: str | None
    tema_id: int | None
    scope: str

    @property
    def in_scope(self) -> bool:
        return self.scope == "in" and self.tema_id is not None


class LawArticleChannel(Channel):
    name = "law_article"
    # Which anchors feed the score: special laws, code articles (CP and CPP), or both.
    anchor_kinds = ("law", "code")

    def score(self, qid: str) -> Scores:
        raw: Scores = {}
        for anchor in self.anchors(qid):
            if anchor.in_scope and self._kind(anchor) in self.anchor_kinds:
                raw[anchor.tema_id] = raw.get(anchor.tema_id, 0.0) + 1.0
        return top_normalized(pool_canonical(self.res, raw))

    def anchors(self, qid: str) -> tuple[Anchor, ...]:
        return self._anchors_cached(qid)

    def in_scope_branches(self, qid: str) -> set[int]:
        tree = self.res.tree
        return {tree.branch_of(a.tema_id) for a in self.anchors(qid)
                if a.in_scope and a.tema_id in tree and self._kind(a) in self.anchor_kinds}

    def out_reasons(self, qid: str) -> list[str]:
        return sorted({a.law_key + (f" art. {a.article}" if a.article else "")
                       for a in self.anchors(qid) if a.scope == "out"})

    def _kind(self, anchor: Anchor) -> str:
        return "code" if anchor.law_key in CODE_KEYS else "law"

    @cached_by_qid
    def _anchors_cached(self, qid: str) -> tuple[Anchor, ...]:
        text = fold(self.res.parsed[qid].full_text)
        mentions = self._mentions(text)
        anchors = [self._law_anchor(entry) for _, _, entry in mentions if entry.kind != "code"]
        for start, article in self._articles(text):
            entry = self._attribute(start, mentions)
            if entry is not None:
                anchors.append(self._article_anchor(entry, article))
        return tuple(a for a in anchors if a is not None)

    def _mentions(self, text: str) -> list[tuple[int, int, object]]:
        """(start, end, LawEntry) for every law number or popular name in the text."""
        found = []
        for match in _LAW_NUMBER.finditer(text):
            entry = self.res.laws.by_law_number(match.group(2))
            if entry is not None:
                found.append((match.start(), match.end(), entry))
        for pos, entry in self.res.laws.find_aliases(text):
            if not any(a <= pos < b for a, b, _ in found):
                found.append((pos, pos + 1, entry))
        return sorted(found, key=lambda m: m[0])

    def _articles(self, text: str) -> list[tuple[int, str]]:
        out = []
        for match in _ARTICLES.finditer(text):
            for one in _ONE_ARTICLE.finditer(match.group(1)):
                suffix = one.group(2)
                out.append((match.start(), one.group(1) + (f"-{suffix.upper()}" if suffix else "")))
        return out

    def _attribute(self, start: int, mentions: list):
        right = [m for m in mentions if 0 <= m[0] - start <= RIGHT_WINDOW]
        if right:
            return right[0][2]
        left = [m for m in mentions if 0 <= start - m[1] <= LEFT_WINDOW]
        if left:
            return left[-1][2]
        distinct = {m[2].law_key for m in mentions}
        if len(distinct) == 1:
            return mentions[0][2]
        return None

    def _law_anchor(self, entry) -> Anchor:
        return Anchor(entry.law_key, None, entry.tema_id, entry.scope)

    def _article_anchor(self, entry, article: str) -> Anchor | None:
        hit = self.res.atlas.lookup(entry.law_key, article)
        if hit is not None:
            return Anchor(entry.law_key, article, hit.tema_id, hit.scope)
        if entry.kind == "code":
            return None
        return Anchor(entry.law_key, article, entry.tema_id, entry.scope)


class SpecialLawChannel(LawArticleChannel):
    """Only special law anchors: a cited special law usually has a single destination."""

    name = "special_law"
    anchor_kinds = ("law",)


class CodeArticleChannel(LawArticleChannel):
    """Only CP and CPP article anchors, which often describe the facts of a case, not its topic."""

    name = "code_article"
    anchor_kinds = ("code",)
