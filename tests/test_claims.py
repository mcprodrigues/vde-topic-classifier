"""The numbers the README states are checked against evaluation.json, so a rerun cannot leave them stale."""

import json
import re

import pytest

from vde.paths import OUTPUT_DIR, ROOT

CLAIM = re.compile(r"<!-- claim:(\w+) -->(.*?)<!-- /claim -->")


@pytest.fixture(scope="module")
def claims():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    return {name: value for name, value in CLAIM.findall(text)}


@pytest.fixture(scope="module")
def report():
    return json.loads((OUTPUT_DIR / "evaluation.json").read_text(encoding="utf-8"))


def test_every_marked_claim_is_checked(claims):
    assert set(claims) == {"published", "coverage", "precision", "below", "nested", "p10"}


def test_readme_claims_match_the_evaluation(claims, report):
    fc, cv = report["forecast"], report["cv"]
    expected = {
        "published": str(fc["published"]),
        "coverage": _br(fc["coverage"]),
        "precision": _br(fc["precision"]),
        "below": f"{fc['probability_precision_below_target']:.0%}",
        "nested": _br(cv["threshold_cv"]["precision"]),
        "p10": _br(cv["precision_at_coverage"]["0.10"]),
    }
    assert claims == expected


def _br(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")
