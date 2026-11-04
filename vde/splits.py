"""Split of the gold sheet (ADR-1).

v0.1 held out a 20 question lockbox. Two of its questions (Q0252, Q0303) were read during error
analysis, so it was no longer blind; since v0.2 `lockbox_size = 0` and every label goes to CV.
"""

from __future__ import annotations

import json

import numpy as np

from vde.knowledge.base import sha256_file, write_provenance
from vde.paths import ARTIFACT_DIR

FILE = "splits.json"
MERGED_NOTE = ("The v0.1 lockbox (20 questions) was inspected during error analysis (Q0252, Q0303), "
               "so it was merged back and all labels are used in cross validation.")


def load_or_create(res, size: int, seed: int) -> tuple[list[str], list[str]]:
    """Return (dev, lockbox). The split is versioned so it never changes between runs."""
    path = ARTIFACT_DIR / FILE
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if len(data["lockbox"]) == size:
            return data["dev"], data["lockbox"]
    dev, lockbox = (sorted(res.gold), []) if size == 0 else _stratified(res, size, seed)
    record = {"seed": seed, "dev": dev, "lockbox": lockbox}
    if size == 0:
        record["note"] = MERGED_NOTE
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
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
