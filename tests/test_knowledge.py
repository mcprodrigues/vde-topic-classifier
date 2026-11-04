from vde.knowledge import canonical_tree, code_atlas, law_index, topic_cards
from vde.knowledge.code_atlas import article_key


def test_tree_has_twenty_shadows_mapped_to_canonical_twins():
    tree = canonical_tree.load()
    shadows = tree.nodes[tree.nodes.is_shadow]
    assert len(shadows) == 20
    assert tree.canonical(6734) == 4703  # Processo Penal > LEP goes to Legislacao Especial > LEP
    assert tree.canonical(6726) == 5199  # Teoria da Pena (Parte 1) goes to Teoria da Pena
    assert all(not tree.is_shadow(t) for t in tree.candidates())


def test_correct_set_follows_the_case_rule():
    tree = canonical_tree.load()
    hits = tree.correct_set(5562)  # Recursos em Especies > Apelacao
    assert 5562 in hits and 5560 in hits  # the node and its parent
    assert 5561 not in hits  # a sibling subtopic is an error
    parent_hits = tree.correct_set(5560)
    assert 5561 in parent_hits  # children of a level 1 gold node are hits


def test_level1_of_never_returns_a_root_or_shadow():
    tree = canonical_tree.load()
    for tid in tree.candidates():
        top = tree.level1_of(tid)
        assert tree.level(top) == 1 and not tree.is_shadow(top)


def test_article_keys_and_atlas_lookup():
    assert article_key("121") == 121.0
    assert abs(article_key("121-A") - 121.01) < 1e-9
    atlas = code_atlas.load()
    assert atlas.lookup("CP", "121-A").tema_id == 5151
    assert atlas.lookup("CP", "157").tema_id == 5160
    assert atlas.lookup("CPP", "28-A").tema_id == 5502
    assert atlas.lookup("CPP", "515").scope == "out"


def test_law_index_and_cards_cover_known_cases():
    laws = law_index.load()
    assert laws.by_law_number("11.340").tema_id == 4704
    assert laws.find_aliases("nos termos da lei maria da penha")[0][1].law_key == "LEI_11340"
    cards = topic_cards.load()
    assert len(cards) == len(canonical_tree.load().candidates())


def test_lsa_transform_reproduces_cached_vectors():
    import numpy as np

    from vde.data.loader import load_questions
    from vde.data.segments import parse
    from vde.knowledge import embeddings

    emb = embeddings.load()
    assert emb.lsa is not None
    parsed = [parse(q) for q in load_questions()[:5]]
    keys = [embeddings.question_key(p.id) for p in parsed]
    again = embeddings.transform([p.full_text for p in parsed], emb.lsa)
    assert np.abs(again - emb.matrix(keys)).max() < 1e-4


def test_twins_learned_from_labels_match_the_versioned_record():
    import pandas as pd

    from vde.data.loader import load_gold
    from vde.knowledge import canonical_tree
    from vde.paths import ARTIFACT_DIR

    tree = canonical_tree.load()
    learned = canonical_tree.learn_twins(tree, load_gold())
    record = pd.read_csv(ARTIFACT_DIR / "branch_twins.csv")
    assert learned == dict(zip(record.from_branch.astype(int), record.to_branch.astype(int)))


def test_manifest_covers_the_annexes():
    import json

    from vde.paths import ARTIFACT_DIR

    manifest = json.loads((ARTIFACT_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    assert {"anexos/questoes.jsonl", "anexos/folha_verdade.csv", "anexos/temas.csv"} <= set(manifest)
