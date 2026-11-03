"""Lockbox split of the gold sheet (ADR-1): 20 questions held out until the final run."""

from __future__ import annotations

import json

import numpy as np

from vde.knowledge.base import sha256_file, write_provenance
from vde.paths import ARTIFACT_DIR

FILE = "splits.json"


def load_or_create(res, size: int, seed: int) -> tuple[list[str], list[str]]:
    """Return (dev, lockbox). The split is versioned so it never changes between runs."""
    path = ARTIFACT_DIR / FILE
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["dev"], data["lockbox"]
    dev, lockbox = _stratified(res, size, seed)
    path.write_text(json.dumps({"seed": seed, "dev": dev, "lockbox": lockbox}, indent=2) + "\n", encoding="utf-8")
    write_provenance(path, produced_by="stratified random split by gold discipline and out flag", llm_used=False,
                     lockbox_size=len(lockbox))
    return dev, lockbox


def fingerprint() -> str:
    path = ARTIFACT_DIR / FILE
    return sha256_file(path) if path.exists() else ""


def _stratified(res, size: int, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    strata: dict[str, list[str]] = {}
    for qid, gold in sorted(res.gold.items()):
        key = "out" if gold.is_out else res.tree.discipline(gold.central)
        strata.setdefault(key, []).append(qid)
    total = len(res.gold)
    lockbox = []
    for key, members in sorted(strata.items()):
        quota = max(1, round(size * len(members) / total))
        lockbox += list(rng.choice(members, size=min(quota, len(members)), replace=False))
    lockbox = sorted(lockbox)[:size] if len(lockbox) > size else sorted(lockbox)
    dev = sorted(q for q in res.gold if q not in set(lockbox))
    return dev, lockbox
