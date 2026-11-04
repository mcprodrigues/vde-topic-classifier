"""Engine and decision safety: no evidence never publishes, folds never leak state, config is pinned."""

import json

import pytest

from vde.config import load_config
from vde.data.loader import Question
from vde.data.segments import parse
from vde.decide import Decision, decide, evidence_gate
from vde.engine import Engine
from vde.knowledge import embeddings
from vde.paths import ARTIFACT_DIR
from vde.pipeline import MODEL_FILE, load_engine, write_publication
from vde.resources import Resources

SYNTHETIC = [
    Question("Q9001", "", "", {}, "", ""),
    Question("Q9002", "<p>Julgue o item.</p>", "", {"C": "Certo", "E": "Errado"}, "", ""),
    Question("Q9003", "xyzzy plugh qwerty", "", {"A": "foo", "B": "bar"}, "Direito Civil", "Contratos"),
]


@pytest.fixture(scope="module")
def deployed():
    """The published engine, with three synthetic questions that carry no legal content at all."""
    config = load_config()
    res = Resources.load(config)
    for question in SYNTHETIC:
        parsed = parse(question)
        res.parsed[parsed.id] = parsed
        res.embeddings.add(*embeddings.question_texts(parsed))
    model = json.loads((ARTIFACT_DIR / MODEL_FILE).read_text(encoding="utf-8"))
    return res, model, load_engine(res, model, config)


def test_questions_without_evidence_never_publish(deployed):
    res, model, engine = deployed
    for pred in engine.predict([q.id for q in SYNTHETIC]):
        assert not pred.text_evidence
        decision = decide(pred, res, threshold=0.0)  # even a threshold of zero must not publish them
        assert decision.decision == "revisar"
        assert decision.central is not None
        assert decision.reason.startswith("sem evidência suficiente")


def test_confidence_does_not_extrapolate_on_an_empty_question(deployed):
    _, _, engine = deployed
    empty = engine.predict(["Q9001"])[0]
    assert empty.confidence < 0.9


def test_evidence_gate_conditions():
    assert evidence_gate(True, 0.9, 0.3) == ""
    assert "massa do ramo" in evidence_gate(True, 0.2, 0.3)
    assert "nenhum termo" in evidence_gate(False, 0.9, 0.3)


def test_write_publication_refuses_rows_that_break_the_format(tmp_path):
    with pytest.raises(ValueError, match="needs tema_central_id"):
        write_publication([Decision("Q0001", "revisar", None, (), 0.0, "x")], tmp_path / "p.csv")
    with pytest.raises(ValueError, match="needs a motivo"):
        write_publication([Decision("Q0001", "fora", None, (), 0.0, "")], tmp_path / "p.csv")


def test_load_engine_refuses_a_config_the_model_was_not_trained_with(deployed):
    res, model, _ = deployed
    changed = load_config(overrides={"knn": {"k": 3}})
    with pytest.raises(RuntimeError, match="config.toml differs"):
        load_engine(res, model, changed)


def test_a_refit_engine_predicts_like_a_fresh_one():
    """CV reuses one engine across folds. Its caches must not keep anything learned in an earlier fold,
    such as the twin rule, which changes when the labels that support it leave the training part."""
    res = Resources.load(load_config())
    labeled = sorted(res.gold)
    twin_support = [q for q in labeled if any(e in res.tree and res.tree.level1_of(e) == 5539
                                              for e in res.gold[q].extras)]
    assert twin_support, "the gold sheet no longer supports the Sumarissimo twin; pick another case"
    train_b = [q for q in labeled if q not in twin_support]
    probe = [q for q in res.question_ids if q not in res.gold]

    reused = Engine(res)
    reused.fit(labeled, seed=1)
    reused.predict(probe)  # fills every cache under the twins of the full gold sheet
    reused.fit(train_b, seed=1)
    after = reused.predict(probe)

    fresh = Engine(Resources.load(load_config()))
    fresh.fit(train_b, seed=1)
    expected = fresh.predict(probe)
    for a, b in zip(after, expected):
        assert a.features == b.features
        assert a.confidence == pytest.approx(b.confidence)
