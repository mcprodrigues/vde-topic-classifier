"""Topic cards: a real description of each of the 351 publishable topics, used as zero shot prototypes.

The source tree has almost empty descriptions, and 290 topics have no gold example. A card joins the
path, the title and reviewed key terms. The LLM proposer only ever receives topic paths.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import pandas as pd

from vde.knowledge import canonical_tree
from vde.knowledge.base import artifact_path, propose_with_llm, proposal_path, write_provenance

FILE = "topic_cards.csv"
TITLE_WEIGHT = 2


@dataclass(frozen=True)
class TopicCard:
    tema_id: int
    path: str
    title: str
    keywords: tuple[str, ...]

    @property
    def text(self) -> str:
        """Document used by the lexical and embedding channels."""
        titles = " ".join([self.title] * TITLE_WEIGHT)
        return f"{self.path}. {titles}. {'; '.join(self.keywords)}"

    @property
    def title_text(self) -> str:
        """Title and path only: the zero knowledge baseline (experiment E0)."""
        return f"{self.path}. {self.title}"


def load() -> dict[int, TopicCard]:
    tree = canonical_tree.load()
    df = pd.read_csv(artifact_path(FILE), dtype={"tema_id": int}).fillna("")
    cards = {}
    for row in df.itertuples(index=False):
        keywords = tuple(k.strip() for k in row.keywords.split(";") if k.strip())
        cards[row.tema_id] = TopicCard(row.tema_id, tree.path(row.tema_id), tree.title(row.tema_id), keywords)
    return cards


def build(source: str = "session") -> str:
    if source == "llm":
        return str(_propose())
    target = artifact_path(FILE)
    _validate(pd.read_csv(target, dtype={"tema_id": int}).fillna(""))
    write_provenance(target, produced_by="key terms proposed by an LLM per topic path, reviewed as data",
                     llm_used=True, proposer="claude-opus-5-5 (authoring session)", llm_saw_questions="not verifiable: the authoring session had read access to the repository, questions included",
                     pseudo_questions=False, human_review="done 2026-11-04: reviewed by the author, no row changed")
    return str(target)


def _validate(df: pd.DataFrame) -> None:
    tree = canonical_tree.load()
    missing = set(tree.candidates()) - set(df.tema_id)
    unknown = set(df.tema_id) - set(tree.candidates())
    if missing or unknown:
        raise ValueError(f"topic_cards: missing {sorted(missing)} unknown {sorted(unknown)}")


def _propose():
    tree = canonical_tree.load()
    system = ("You write key terms for a Brazilian criminal law topic, as they appear in public exam "
              "questions. You never see the questions themselves. CSV only.")
    rows = [["tema_id", "keywords"]]
    for tid in tree.candidates():
        prompt = (f"Topic path: {tree.path(tid)}. Give 6 to 15 Portuguese key terms, articles and "
                  "precedents that identify this topic and not its siblings, separated by ';'. One line.")
        rows.append([str(tid), propose_with_llm(system, prompt, max_tokens=400).strip().splitlines()[0]])
    target = proposal_path(FILE.replace(".csv", ".proposed.csv"))
    with open(target, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return target

