"""Names accepted by `vde build <artifact>`, in dependency order."""

from __future__ import annotations

from typing import Callable

from vde.knowledge import canonical_tree, code_atlas, embeddings, law_index, subject_map, topic_cards

BUILD_ORDER = ["canonical_tree", "branch_twins", "law_index", "code_atlas", "topic_cards", "subject_map", "embeddings"]


def builders(config: dict, source: str = "session") -> dict[str, Callable[[], str]]:
    emb = config.get("embeddings", {})
    return {
        "canonical_tree": lambda: str(canonical_tree.build()),
        "branch_twins": lambda: str(canonical_tree.build_twins()),
        "law_index": lambda: law_index.build(source),
        "code_atlas": lambda: code_atlas.build(source),
        "topic_cards": lambda: topic_cards.build(source),
        "subject_map": lambda: subject_map.build(),
        "embeddings": lambda: embeddings.build(emb.get("backend", "lsa"), emb.get("dimensions", 256),
                                               emb.get("seed", 0), emb.get("model", embeddings.DEFAULT_ST_MODEL)),
    }
