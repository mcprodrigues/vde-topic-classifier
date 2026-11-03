"""Index from a cited law (number or popular name) to a topic, plus the codes whose articles go to code_atlas.

The reviewed CSV is the artifact. build(source="llm") only writes a proposal for human review.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import pandas as pd

from vde.data.cleaning import fold
from vde.knowledge import canonical_tree
from vde.knowledge.base import artifact_path, propose_with_llm, proposal_path, write_provenance

FILE = "law_index.csv"
COLUMNS = ["law_key", "kind", "number", "year", "tema_id", "scope", "aliases", "note"]


@dataclass(frozen=True)
class LawEntry:
    law_key: str
    kind: str
    number: str
    year: str
    tema_id: int | None
    scope: str
    aliases: tuple[str, ...]
    note: str


class LawIndex:
    def __init__(self, entries: list[LawEntry]):
        self.entries = entries
        self.by_number = {e.number: e for e in entries if e.number}
        self.by_key = {e.law_key: e for e in entries}
        self.aliases = sorted(((a, e) for e in entries for a in e.aliases), key=lambda x: -len(x[0]))

    def by_law_number(self, number: str) -> LawEntry | None:
        return self.by_number.get(number.replace(".", "").lstrip("0"))

    def find_aliases(self, folded_text: str) -> list[tuple[int, LawEntry]]:
        """Positions of popular law names in an accent folded text, longest names first."""
        found, taken = [], []
        for alias, entry in self.aliases:
            start = 0
            while True:
                pos = folded_text.find(alias, start)
                if pos < 0:
                    break
                end = pos + len(alias)
                bounded = (pos == 0 or not folded_text[pos - 1].isalnum()) and (
                    end == len(folded_text) or not folded_text[end].isalnum())
                if bounded and not any(a <= pos < b for a, b in taken):
                    found.append((pos, entry))
                    taken.append((pos, end))
                start = end
        return sorted(found, key=lambda x: x[0])


def load() -> LawIndex:
    df = pd.read_csv(artifact_path(FILE), dtype=str).fillna("")
    return LawIndex([_entry(row) for row in df.itertuples(index=False)])


def build(source: str = "session") -> str:
    """Validate the reviewed CSV, or ask an LLM for a proposal when source == 'llm'."""
    if source == "llm":
        return str(_propose())
    target = artifact_path(FILE)
    _validate(pd.read_csv(target, dtype=str).fillna(""))
    write_provenance(target, produced_by="proposed by an LLM from the topic tree, reviewed as data",
                     llm_used=True, proposer="claude-opus-5-5 (authoring session)",
                     llm_saw_questions=False, human_review="pending: see manual_decisions.csv")
    return str(target)


def _entry(row) -> LawEntry:
    aliases = tuple(fold(a).strip() for a in row.aliases.split(";") if a.strip())
    tema = int(row.tema_id) if row.tema_id.strip() else None
    return LawEntry(row.law_key, row.kind, row.number.replace(".", ""), row.year, tema, row.scope, aliases, row.note)


def _validate(df: pd.DataFrame) -> None:
    tree = canonical_tree.load()
    for row in df.itertuples(index=False):
        if row.scope == "in" and (not row.tema_id or int(row.tema_id) not in tree.candidates()):
            raise ValueError(f"law_index: {row.law_key} points to an invalid topic {row.tema_id!r}")


def _propose():
    tree = canonical_tree.load()
    topics = "\n".join(f"{t};{tree.path(t)}" for t in tree.candidates())
    system = ("You build a lookup table for a Brazilian criminal law topic tree. You never see exam "
              "questions. Answer only with CSV, no prose.")
    prompt = (f"Columns: {','.join(COLUMNS)}. One row per Brazilian law commonly cited in criminal law "
              "exams. scope is in (has a topic), out (no topic in the tree) or code (CP, CPP). "
              f"Use only these topic ids:\n{topics}")
    text = propose_with_llm(system, prompt)
    rows = list(csv.reader(io.StringIO(text.strip().strip("`"))))
    target = proposal_path(FILE.replace(".csv", ".proposed.csv"))
    with open(target, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return target
