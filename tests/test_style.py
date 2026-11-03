"""House rules: private functions and methods come last, and the code has no em or en dash characters."""

import ast

from vde.paths import PACKAGE_DIR

DASHES = ("—", "–")


def test_private_definitions_come_last():
    problems = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        problems += _order_problems(path.name, "<module>", tree.body)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                problems += _order_problems(path.name, node.name, node.body)
    assert problems == []


def test_no_dash_characters_in_code():
    offenders = [f"{p.name}:{i}" for p in sorted(PACKAGE_DIR.rglob("*.py")) + sorted(PACKAGE_DIR.rglob("*.toml"))
                 for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                 if any(d in line for d in DASHES)]
    assert offenders == []


def _order_problems(filename: str, scope: str, body: list) -> list[str]:
    seen_private = None
    problems = []
    for node in body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        is_private = node.name.startswith("_") and not node.name.startswith("__")
        if is_private and seen_private is None:
            seen_private = node.name
        elif not is_private and seen_private is not None:
            problems.append(f"{filename}:{scope}: public {node.name} after private {seen_private}")
    return problems
