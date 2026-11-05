import re

import pandas as pd
import pytest

from vde.data.loader import load_questions
from vde.knowledge import canonical_tree
from vde.paths import OUTPUT_DIR


@pytest.fixture(scope="module")
def publication():
    return pd.read_csv(OUTPUT_DIR / "publicacao.csv", dtype=str).fillna("")


def test_one_row_per_question(publication):
    assert list(publication.id) == [q.id for q in load_questions()]
    assert list(publication.columns) == ["id", "decisao", "tema_central_id", "extras_ids", "confianca", "motivo"]


def test_decision_rules(publication):
    assert set(publication.decisao) <= {"publicar", "revisar", "fora"}
    needs_topic = publication[publication.decisao.isin(["publicar", "revisar"])]
    assert (needs_topic.tema_central_id != "").all()
    needs_reason = publication[publication.decisao.isin(["revisar", "fora"])]
    assert (needs_reason.motivo != "").all()
    assert (publication[publication.decisao == "fora"].tema_central_id == "").all()
    conf = publication.confianca.astype(float)
    assert conf.between(0, 1).all()


def test_output_topics_are_valid_never_a_root_and_never_a_shadow(publication):
    tree = canonical_tree.load()
    rows = publication[publication.decisao != "fora"]
    topics = [int(v) for v in rows.tema_central_id] + [int(e) for v in rows.extras_ids for e in v.split(";") if e]
    for tid in topics:
        assert tid in tree and tree.level(tid) >= 1 and not tree.is_shadow(tid), tid


HEADER = [
    ("precisao_prevista", r"\d\.\d{2}"),
    ("precisao_intervalo", r"\d\.\d{2}-\d\.\d{2}"),
    ("cobertura_prevista", r"\d\.\d{2}"),
    ("cobertura_intervalo", r"\d\.\d{2}-\d\.\d{2}"),
]


def test_forecast_header_is_numeric():
    lines = (OUTPUT_DIR / "previsao.txt").read_text(encoding="utf-8").splitlines()
    for line, (key, pattern) in zip(lines, HEADER):
        assert re.fullmatch(rf"{key}: {pattern}", line.strip()), line


def test_forecast_coverage_matches_publication(publication):
    lines = (OUTPUT_DIR / "previsao.txt").read_text(encoding="utf-8").splitlines()
    coverage = float(lines[2].split(":")[1])
    evaluated = publication[~publication.motivo.str.startswith("folha-verdade")]
    assert abs(coverage - (evaluated.decisao == "publicar").mean()) < 0.005


def test_published_rows_meet_the_model_threshold(publication):
    import json

    from vde.paths import ARTIFACT_DIR

    tau = json.loads((ARTIFACT_DIR / "model.json").read_text(encoding="utf-8"))["threshold"]
    evaluated = publication[~publication.motivo.str.startswith("folha-verdade")]
    published = evaluated[evaluated.decisao == "publicar"]
    assert (published.confianca.astype(float) >= round(tau, 4) - 1e-4).all()


def test_forecast_reports_each_source_and_stays_inside_its_interval():
    import json

    report = json.loads((OUTPUT_DIR / "evaluation.json").read_text(encoding="utf-8"))
    fc = report["forecast"]
    assert set(fc["by_source"]) == {"bank", "new_import"}
    for block in [fc] + list(fc["by_source"].values()):
        low, high = block["precision_interval"]
        assert low <= block["precision"] <= high
    assert report["shift"]["labeled_from_new_import"] == 0


def test_forecast_is_centered_on_the_calibrated_confidence_of_the_published():
    import json

    fc = json.loads((OUTPUT_DIR / "evaluation.json").read_text(encoding="utf-8"))["forecast"]
    assert fc["method"] == "calibrated_confidence"
    # Clipping at 1 can only pull the simulated mean below the mean confidence, and only a little.
    assert 0 <= fc["precision_from_calibrated_confidence"] - fc["precision"] < 0.02
    assert fc["by_source"]["new_import"]["calibration_sigma"] > fc["by_source"]["bank"]["calibration_sigma"]
