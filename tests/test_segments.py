from vde.data.cleaning import fold, html_to_text, tokenize
from vde.data.loader import Question
from vde.data.segments import parse


def test_html_to_text_keeps_breaks_and_unescapes():
    assert html_to_text("a<br>b &amp; c<span>d</span>") == "a\nb & c d"


def test_fold_and_tokenize_remove_accents_and_stopwords():
    assert fold("Ação Penal") == "acao penal"
    assert tokenize("A prisão em flagrante do réu") == ["prisao", "flagrante", "reu"]


def test_parse_splits_line_assertions():
    q = Question("Q9999", "Analise.<br>I - primeira assertiva longa sobre prisao preventiva.<br>"
                 "II - segunda assertiva longa sobre fianca e liberdade.", "Estao corretas",
                 {"A": "I", "B": "II", "C": "I e II"}, "Processo Penal", "")
    parsed = parse(q)
    assert parsed.kind == "assertions"
    assert len(parsed.assertions) == 2
    assert [s.key for s in parsed.segments()] == ["stem", "as:1", "as:2"]


def test_parse_detects_true_false_and_negation():
    q = Question("Q0400", "Assinale a alternativa INCORRETA sobre o tema.", "", {"A": "Certo", "B": "Errado"}, "", "")
    parsed = parse(q)
    assert parsed.kind == "true_false"
    assert parsed.negated
    assert parsed.is_new_import
