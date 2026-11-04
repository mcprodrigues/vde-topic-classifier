"""Canonical topic tree: active topics, shadow nodes mapped to their canonical twin, bin statistics.

The source tree has 20 level 1 "shadow" nodes (no description, no children) that duplicate a
canonical topic, such as "Teoria da Pena (Parte 1)" or "Penal > Lei Maria da Penha". The gold
sheet uses the canonical twin, so every score is pooled on the canonical node (ADR-5).
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from vde.data.cleaning import fold
from vde.data.loader import load_gold, load_topics
from vde.knowledge.base import artifact_path, write_provenance
from vde.paths import ANNEX_DIR, ARTIFACT_DIR

FILE = "canonical_tree.csv"
TWINS_FILE = "branch_twins.csv"
MANUAL_DECISIONS = "manual_decisions.csv"
BIN_MIN_SOURCE_COUNT = 250
SHADOW_MATCH_RATIO = 0.86
TWIN_MIN_EXTRA = 2


@dataclass
class Tree:
    nodes: pd.DataFrame
    twins: dict[int, int] = field(default_factory=dict)
    _children: dict[int, list[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.nodes = self.nodes.set_index("tema_id", drop=False)
        records = self.nodes.to_dict("index")
        # Plain dicts: these lookups run millions of times during cross validation.
        self._parent = {t: (None if pd.isna(r["parent_id"]) else int(r["parent_id"])) for t, r in records.items()}
        self._level = {t: int(r["level"]) for t, r in records.items()}
        self._title = {t: str(r["title"]) for t, r in records.items()}
        self._path = {t: str(r["path"]) for t, r in records.items()}
        self._discipline = {t: str(r["discipline"]) for t, r in records.items()}
        self._canonical = {t: int(r["canonical_id"]) for t, r in records.items()}
        self._shadow = {t: bool(r["is_shadow"]) for t, r in records.items()}
        self._bin = {t: bool(r["is_bin"]) for t, r in records.items()}
        self._count = {t: int(r["source_count"]) for t, r in records.items()}
        for tid, parent in self._parent.items():
            if parent is not None:
                self._children.setdefault(parent, []).append(tid)
        self._level1 = {t: self._compute_level1(t) for t in records if self._level[t] >= 1}

    def __contains__(self, tid: int) -> bool:
        return tid in self._level

    def parent(self, tid: int) -> int | None:
        return self._parent[tid]

    def children(self, tid: int) -> list[int]:
        return list(self._children.get(tid, []))

    def level(self, tid: int) -> int:
        return self._level[tid]

    def title(self, tid: int) -> str:
        return self._title[tid]

    def path(self, tid: int) -> str:
        return self._path[tid]

    def discipline(self, tid: int) -> str:
        return self._discipline[tid]

    def canonical(self, tid: int) -> int:
        return self._canonical[tid]

    def is_shadow(self, tid: int) -> bool:
        return self._shadow[tid]

    def is_bin(self, tid: int) -> bool:
        return self._bin[tid]

    def source_count(self, tid: int) -> int:
        return self._count[tid]

    def level1_of(self, tid: int) -> int:
        """Canonical level 1 ancestor of a node (the node itself when it is level 1)."""
        return self._level1[tid]

    def candidates(self) -> list[int]:
        """Nodes we may publish in: canonical, active, level 1 or 2. Never level 0, never a shadow."""
        return [t for t, lvl in self._level.items() if lvl >= 1 and not self._shadow[t]]

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

    def branch_of(self, tid: int) -> int:
        """Publishable level 1 branch of a node: level1_of followed by the twin rule."""
        return self.twin_branch(self._level1[tid])

    def twin_branch(self, branch: int) -> int:
        """Level 1 branch the team actually uses when two disciplines hold the same subject (ADR-5)."""
        return self.twins.get(branch, branch)

    def _compute_level1(self, tid: int) -> int:
        node = self._canonical[tid]
        while self._level[node] > 1:
            node = self._canonical[self._parent[node]]
        return node


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


def learn_twins(tree: Tree, labeled: dict, min_extra: int = TWIN_MIN_EXTRA) -> dict[int, int]:
    """Twin rule (ADR-5) learned from labels: branch A goes to branch B of another discipline when A shows
    up as an extra next to a B central at least `min_extra` times and is never central itself."""
    central: Counter = Counter()
    pairs: Counter = Counter()
    for gold in labeled.values():
        if gold.is_out or gold.central is None or gold.central not in tree:
            continue
        b = tree.level1_of(gold.central)
        central[b] += 1
        for extra in gold.extras:
            if extra in tree:
                a = tree.level1_of(extra)
                if a != b and tree.discipline(a) != tree.discipline(b):
                    pairs[(a, b)] += 1
    twins: dict[int, int] = {}
    for (a, b), n in pairs.most_common():
        if n >= min_extra and central[a] == 0 and a not in twins:
            twins[a] = b
    return twins


def build_twins(annex_dir: Path = ANNEX_DIR) -> Path:
    """Write branch_twins.csv from the gold sheet, as a readable record of what learn_twins finds."""
    tree = load()
    tree.twins = {}
    labeled = load_gold(annex_dir)
    twins = learn_twins(tree, labeled)
    rows = []
    for a, b in sorted(twins.items()):
        rows.append({"from_branch": a, "to_branch": b, "from_path": tree.path(a), "to_path": tree.path(b),
                     "reason": f"{tree.path(a)} appears only as an extra next to a {tree.path(b)} central"})
    target = artifact_path(TWINS_FILE)
    pd.DataFrame(rows, columns=["from_branch", "to_branch", "from_path", "to_path", "reason"]).to_csv(target, index=False)
    write_provenance(target, produced_by="learn_twins over the gold sheet; refitted inside every CV fold",
                     llm_used=False, used_labels=True, min_extra=TWIN_MIN_EXTRA, twins=len(rows))
    return target


def load() -> Tree:
    nodes = pd.read_csv(artifact_path(FILE))
    nodes["parent_id"] = nodes["parent_id"].astype("Int64")
    nodes["is_shadow"] = nodes["is_shadow"].astype(bool)
    nodes["is_bin"] = nodes["is_bin"].astype(bool)
    nodes["has_description"] = nodes["has_description"].astype(bool)
    return Tree(nodes, _load_twins())


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


def _load_twins() -> dict[int, int]:
    path = ARTIFACT_DIR / TWINS_FILE
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(a): int(b) for a, b in zip(df.from_branch, df.to_branch)}
