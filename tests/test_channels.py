import pytest

from vde.channels import build_channels
from vde.config import load_config
from vde.resources import Resources


@pytest.fixture(scope="module")
def res():
    return Resources.load(load_config())


def test_special_law_anchor_resolves_maria_da_penha(res):
    channel = build_channels(res, ["law_article"])["law_article"]
    hits = [q for q in res.question_ids if any(a.tema_id == 4704 for a in channel.anchors(q))]
    assert len(hits) >= 5


def test_knn_never_votes_with_its_own_label(res):
    knn = build_channels(res, ["knn"])["knn"]
    one = next(iter(res.gold))
    knn.fit({one: res.gold[one]})
    assert knn.score(one) == {}


def test_scores_are_canonical_and_normalized(res):
    channels = build_channels(res, ["lexical", "prototypes", "subject"])
    for channel in channels.values():
        for qid in res.question_ids[:50]:
            scores = channel.score(qid)
            if scores:
                assert abs(max(scores.values()) - 1.0) < 1e-9
                assert all(not res.tree.is_shadow(t) and res.tree.level(t) > 0 for t in scores)


def test_min_run_to_certify():
    from vde.calibrate import min_run_to_certify

    assert min_run_to_certify(0.95, 0.80) == 31
    assert min_run_to_certify(0.95, 0.90) == 44
