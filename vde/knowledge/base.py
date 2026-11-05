"""Shared helpers for knowledge artifacts: provenance sidecars, the manifest and the offline LLM client."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from vde.paths import ANNEX_DIR, ARTIFACT_DIR, ROOT

ANNEX_PREFIX = "anexos/"

MANIFEST_NAME = "MANIFEST.json"
DEFAULT_LLM_MODEL = "claude-opus-5-5"


def artifact_path(name: str) -> Path:
    return ARTIFACT_DIR / name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(artifact_file: Path, **fields) -> Path:
    """Write <artifact>.prov.json next to the artifact with its hash and how it was produced."""
    record = {
        "artifact": artifact_file.name,
        "sha256": sha256_file(artifact_file),
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "code_commit": _git_commit(),
        **fields,
    }
    target = artifact_file.with_name(artifact_file.name + ".prov.json")
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def refresh_manifest() -> dict:
    """Record the sha256 of every versioned artifact so publish can detect drift."""
    entries = {}
    for path in sorted(ARTIFACT_DIR.glob("*")):
        if path.is_file() and path.name != MANIFEST_NAME and not path.name.endswith(".prov.json"):
            entries[path.name] = sha256_file(path)
    for path in sorted(ANNEX_DIR.glob("*")):
        if path.is_file() and path.suffix in {".csv", ".jsonl"}:
            entries[ANNEX_PREFIX + path.name] = sha256_file(path)
    (ARTIFACT_DIR / MANIFEST_NAME).write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return entries


def verify_manifest() -> list[str]:
    """Artifacts or annexes whose hash differs from the manifest, or that are missing.

    Names of annexes start with ANNEX_PREFIX. publish always stops on an artifact problem; an annex that
    changed (a new weekly import) is accepted only with `vde publish --allow-annex-change`, which classifies
    the new questions through the stored LSA transform and keeps every artifact check.
    """
    manifest_file = ARTIFACT_DIR / MANIFEST_NAME
    if not manifest_file.exists():
        return [MANIFEST_NAME]
    expected = json.loads(manifest_file.read_text(encoding="utf-8"))
    problems = []
    for name, digest in expected.items():
        path = ANNEX_DIR / name[len(ANNEX_PREFIX):] if name.startswith(ANNEX_PREFIX) else ARTIFACT_DIR / name
        if not path.exists() or sha256_file(path) != digest:
            problems.append(name)
    return problems


def split_problems(problems: list[str]) -> tuple[list[str], list[str]]:
    """(artifact problems, annex problems) of a verify_manifest() result."""
    annex = [p for p in problems if p.startswith(ANNEX_PREFIX)]
    return [p for p in problems if p not in annex], annex


def propose_with_llm(system: str, prompt: str, model: str = DEFAULT_LLM_MODEL, max_tokens: int = 8000) -> str:
    """Offline proposal step. Imported lazily so the publish path never loads an LLM client."""
    import anthropic  # noqa: PLC0415  (lazy on purpose, see ADR-8)

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if getattr(block, "type", "") == "text")


def proposal_path(name: str) -> Path:
    path = ARTIFACT_DIR / "proposals" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=5, check=False)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"
