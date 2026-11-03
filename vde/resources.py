"""Everything the channels read, loaded once from the annexes and the versioned artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vde.data.loader import GoldLabel, load_gold, load_questions
from vde.data.segments import ParsedQuestion, parse
from vde.knowledge import canonical_tree, code_atlas, embeddings, law_index, subject_map, topic_cards
from vde.knowledge.canonical_tree import Tree
from vde.knowledge.code_atlas import CodeAtlas
from vde.knowledge.embeddings import Embeddings
from vde.knowledge.law_index import LawIndex
from vde.knowledge.topic_cards import TopicCard
from vde.paths import ANNEX_DIR


@dataclass
class Resources:
    config: dict
    tree: Tree
    parsed: dict[str, ParsedQuestion]
    gold: dict[str, GoldLabel]
    cards: dict[int, TopicCard]
    embeddings: Embeddings
    laws: LawIndex
    atlas: CodeAtlas
    subjects: dict[str, list[tuple[int, float]]]

    @property
    def question_ids(self) -> list[str]:
        return list(self.parsed)

    @classmethod
    def load(cls, config: dict, annex_dir: Path = ANNEX_DIR) -> "Resources":
        """Read only: every loader here calls load(), never build()."""
        return cls(
            config=config,
            tree=canonical_tree.load(),
            parsed={q.id: parse(q) for q in load_questions(annex_dir)},
            gold=load_gold(annex_dir),
            cards=topic_cards.load(),
            embeddings=embeddings.load(),
            laws=law_index.load(),
            atlas=code_atlas.load(),
            subjects=subject_map.load(),
        )
