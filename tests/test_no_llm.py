"""ADR-8 as code: publish runs with no API key, no network and no LLM client, and reproduces the CSV."""

import ast
import builtins
import hashlib
import os
import socket
from pathlib import Path

from vde.paths import OUTPUT_DIR, PACKAGE_DIR

LLM_MODULES = {"anthropic", "openai", "sentence_transformers", "torch", "transformers"}


def test_publish_without_llm_network_or_keys(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.endswith("API_KEY") or key.startswith(("ANTHROPIC", "OPENAI")):
            monkeypatch.delenv(key, raising=False)

    def blocked(*args, **kwargs):
        raise OSError("network disabled during publish test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in LLM_MODULES:
            raise ImportError(f"{name} is not allowed on the publish path")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    from vde.pipeline import publish

    out = tmp_path / "publicacao.csv"
    publish(out)
    assert _sha256(out) == _sha256(OUTPUT_DIR / "publicacao.csv")


def test_no_module_level_llm_imports():
    offenders = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders += [f"{path.name}:{n}" for n in names if n.split(".")[0] in LLM_MODULES]
    assert offenders == []


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
