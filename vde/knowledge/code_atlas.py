"""Atlas of article ranges: the penal and procedure codes are already a tree aligned with the topics.

Ranges generalize to articles never seen in the gold sheet (art. 155 to 156 is theft, and so on).
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass

import pandas as pd

from vde.knowledge import canonical_tree
from vde.knowledge.base import artifact_path, propose_with_llm, proposal_path, write_provenance

FILE = "code_atlas.csv"
COLUMNS = ["code", "art_from", "art_to", "tema_id", "scope", "note"]
_ARTICLE = re.compile(r"^(\d+)(?:-([A-Z]))?$")


@dataclass(frozen=True)
class AtlasRange:
    code: str
    start: float
    end: float
    tema_id: int | None
    scope: str
    note: str

    @property
    def width(self) -> float:
        return self.end - self.start


class CodeAtlas:
    def __init__(self, ranges: list[AtlasRange]):
        self.ranges: dict[str, list[AtlasRange]] = {}
        for r in ranges:
            self.ranges.setdefault(r.code, []).append(r)

    def lookup(self, code: str, article: str) -> AtlasRange | None:
        """Narrowest range of `code` containing `article` (such as '121-A')."""
        key = article_key(article)
        if key is None:
            return None
        hits = [r for r in self.ranges.get(code, []) if r.start <= key <= r.end]
        return min(hits, key=lambda r: r.width) if hits else None

    def has_code(self, code: str) -> bool:
        return code in self.ranges


def article_key(article: str) -> float | None:
    """'121' -> 121.0, '121-A' -> 121.01, '359-L' -> 359.12."""
    match = _ARTICLE.match(article.strip().upper().replace(" ", ""))
    if not match:
        return None
    number = int(match.group(1))
    suffix = match.group(2)
    return number + ((ord(suffix) - 64) / 100 if suffix else 0.0)


def load() -> CodeAtlas:
    df = pd.read_csv(artifact_path(FILE), dtype=str).fillna("")
    return CodeAtlas([_range(row) for row in df.itertuples(index=False)])


def build(source: str = "session") -> str:
    if source == "llm":
        return str(_propose())
    target = artifact_path(FILE)
    _validate(pd.read_csv(target, dtype=str).fillna(""))
    write_provenance(target, produced_by="proposed by an LLM from the codes structure and the topic tree",
                     llm_used=True, proposer="claude-opus-5-5 (authoring session)",
                     llm_saw_questions="not verifiable: the authoring session had read access to the repository, questions included", human_review="done 2026-11-04: reviewed by the author, no row changed")
    return str(target)


def _range(row) -> AtlasRange:
    tema = int(row.tema_id) if row.tema_id.strip() else None
    return AtlasRange(row.code, article_key(row.art_from), article_key(row.art_to), tema, row.scope, row.note)


def _validate(df: pd.DataFrame) -> None:
    tree = canonical_tree.load()
    candidates = set(tree.candidates())
    for row in df.itertuples(index=False):
        if article_key(row.art_from) is None or article_key(row.art_to) is None:
            raise ValueError(f"code_atlas: bad article in {row}")
        if row.scope == "in" and int(row.tema_id) not in candidates:
            raise ValueError(f"code_atlas: invalid topic {row.tema_id} in {row}")


def _propose():
    tree = canonical_tree.load()
    topics = "\n".join(f"{t};{tree.path(t)}" for t in tree.candidates())
    system = "You map article ranges of Brazilian codes to topics. You never see exam questions. CSV only."
    prompt = (f"Columns: {','.join(COLUMNS)}. code is CP or CPP. Cover the whole codes with contiguous "
              f"ranges. Use only these topic ids, or scope=out when no topic fits:\n{topics}")
    rows = list(csv.reader(io.StringIO(propose_with_llm(system, prompt).strip().strip("`"))))
    target = proposal_path(FILE.replace(".csv", ".proposed.csv"))
    with open(target, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return target
