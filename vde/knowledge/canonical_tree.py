"""Canonical topic tree: active topics, shadow nodes mapped to their canonical twin, bin statistics.

The source tree has 20 level 1 "shadow" nodes (no description, no children) that duplicate a
canonical topic, such as "Teoria da Pena (Parte 1)" or "Penal > Lei Maria da Penha". The gold
sheet uses the canonical twin, so every score is pooled on the canonical node (ADR-5).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from vde.data.cleaning import fold
from vde.data.loader import load_topics
from vde.knowledge.base import artifact_path, write_provenance
from vde.paths import ANNEX_DIR, ARTIFACT_DIR

FILE = "canonical_tree.csv"
MANUAL_DECISIONS = "manual_decisions.csv"
BIN_MIN_SOURCE_COUNT = 250
SHADOW_MATCH_RATIO = 0.86


@dataclass
class Tree:
    nodes: pd.DataFrame
    _children: dict[int, list[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.nodes = self.nodes.set_index("tema_id", drop=False)
        for tid, parent in zip(self.nodes.tema_id, self.nodes.parent_id):
            if pd.notna(parent):
                self._children.setdefault(int(parent), []).append(int(tid))

    def __contains__(self, tid: int) -> bool:
        return tid in self.nodes.index

    def parent(self, tid: int) -> int | None:
        value = self.nodes.at[tid, "parent_id"]
        return None if pd.isna(value) else int(value)

    def children(self, tid: int) -> list[int]:
        return list(self._children.get(tid, []))

    def level(self, tid: int) -> int:
        return int(self.nodes.at[tid, "level"])

    def title(self, tid: int) -> str:
        return str(self.nodes.at[tid, "title"])

    def path(self, tid: int) -> str:
        return str(self.nodes.at[tid, "path"])

    def discipline(self, tid: int) -> str:
        return str(self.nodes.at[tid, "discipline"])

    def canonical(self, tid: int) -> int:
        return int(self.nodes.at[tid, "canonical_id"])

    def is_shadow(self, tid: int) -> bool:
        return bool(self.nodes.at[tid, "is_shadow"])

    def is_bin(self, tid: int) -> bool:
        return bool(self.nodes.at[tid, "is_bin"])

    def source_count(self, tid: int) -> int:
        return int(self.nodes.at[tid, "source_count"])

    def level1_of(self, tid: int) -> int:
        """Canonical level 1 ancestor of a node (the node itself when it is level 1)."""
        node = self.canonical(tid)
        while self.level(node) > 1:
            node = self.canonical(self.parent(node))
        return node

    def candidates(self) -> list[int]:
        """Nodes we may publish in: canonical, active, level 1 or 2. Never level 0, never a shadow."""
        mask = (self.nodes.level >= 1) & (~self.nodes.is_shadow)
        return [int(t) for t in self.nodes.index[mask]]

    def level1_nodes(self) -> list[int]:
        return [t for t in self.candidates() if self.level(t) == 1]

    def correct_set(self, central: int | None, extras: tuple[int, ...] = ()) -> set[int]:
        """Nodes that count as a hit under the case rule: a gold topic, its parent or its children."""
        hits: set[int] = set()
        for gold in ([central] if central is not None else []) + list(extras):
            if gold not in self:
                continue
            hits.add(gold)
            parent = self.parent(gold)
            if parent is not None:
                hits.add(parent)
            hits.update(self.children(gold))
        return hits


def build(annex_dir: Path = ANNEX_DIR) -> Path:
    """Derive the canonical tree from temas.csv plus the manual overrides file."""
    topics = load_topics(annex_dir)
    active = topics[topics.status == "active"].copy()
    nodes = pd.DataFrame({
        "tema_id": active.tema_id.astype(int),
        "parent_id": active.parent_id,
        "discipline": active.disciplina,
        "level": active.nivel.astype(int),
        "title": active.titulo,
        "path": active.caminho,
        "source_count": active.descricao.fillna("").str.extract(r"(\d+) quest", expand=False).fillna(0).astype(int),
        "has_description": active.descricao.notna(),
    })
    has_children = set(nodes.parent_id.dropna().astype(int))
    shadow_map = _detect_shadows(nodes, has_children)
    shadow_map.update(_manual_overrides())
    nodes["canonical_id"] = [shadow_map.get(t, t) for t in nodes.tema_id]
    nodes["is_shadow"] = nodes.canonical_id != nodes.tema_id
    nodes["is_bin"] = (nodes.level == 1) & (nodes.source_count >= BIN_MIN_SOURCE_COUNT)
    target = artifact_path(FILE)
    nodes.to_csv(target, index=False)
    write_provenance(target, produced_by="deterministic rules over temas.csv plus manual_decisions.csv",
                     llm_used=False, shadows=int(nodes.is_shadow.sum()),
                     inactive_dropped=int((topics.status != "active").sum()))
    return target


def load() -> Tree:
    nodes = pd.read_csv(artifact_path(FILE))
    nodes["parent_id"] = nodes["parent_id"].astype("Int64")
    nodes["is_shadow"] = nodes["is_shadow"].astype(bool)
    nodes["is_bin"] = nodes["is_bin"].astype(bool)
    nodes["has_description"] = nodes["has_description"].astype(bool)
    return Tree(nodes)


def _detect_shadows(nodes: pd.DataFrame, has_children: set[int]) -> dict[int, int]:
    """Map childless, undescribed level 1 nodes to the most similar described level 1 topic."""
    level1 = nodes[nodes.level == 1]
    described = level1[level1.has_description]
    suspects = level1[~level1.has_description & ~level1.tema_id.isin(has_children)]
    mapping = {}
    for row in suspects.itertuples(index=False):
        key = _title_key(row.title)
        ratios = [(difflib.SequenceMatcher(None, key, _title_key(c.title)).ratio(), int(c.tema_id))
                  for c in described.itertuples(index=False)]
        best_ratio, best = max(ratios)
        if best_ratio >= SHADOW_MATCH_RATIO:
            mapping[int(row.tema_id)] = best
    return mapping


def _title_key(title: str) -> str:
    text = fold(title)
    text = re.sub(r"\(parte \d+\)", " ", text)
    text = re.sub(r"[^a-z ]", " ", text)
    words = [w.rstrip("s") for w in text.split() if w not in {"parte", "de", "do", "da", "em", "dos", "das"}]
    return " ".join(words)


def _manual_overrides() -> dict[int, int]:
    """Rows of manual_decisions.csv with artifact=canonical_tree and decision 'canonical=<id>'."""
    path = ARTIFACT_DIR / MANUAL_DECISIONS
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    out = {}
    for row in df[df.artifact == "canonical_tree"].itertuples(index=False):
        if row.decision.startswith("canonical="):
            out[int(row.subject)] = int(row.decision.split("=", 1)[1])
    return out
