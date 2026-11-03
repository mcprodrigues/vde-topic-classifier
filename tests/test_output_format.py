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


def test_published_topics_are_valid_and_never_a_root(publication):
    tree = canonical_tree.load()
    for value in publication[publication.decisao == "publicar"].tema_central_id:
        tid = int(value)
        assert tid in tree and tree.level(tid) >= 1
