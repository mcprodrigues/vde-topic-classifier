"""Map the noisy origin subject (assunto_origem) to topics by string evidence only.

The subject is treated as weak evidence, never as the answer (ADR-2). The two sources use different
formats: the bank writes "Ação Penal (Parte 01)", the new import writes edital style strings with law
numbers, so the key strips the part suffix and law numbers are resolved through law_index.
"""

from __future__ import annotations

import re

import pandas as pd

from vde.data.cleaning import fold, tokenize
from vde.data.loader import load_questions
from vde.knowledge import canonical_tree, law_index
from vde.knowledge.base import artifact_path, write_provenance
from vde.paths import ANNEX_DIR

FILE = "subject_map.csv"
MIN_TITLE_COVERAGE = 0.75
MAX_MATCHES = 3
_LAW_NUMBER = re.compile(r"\blei\s*(?:complementar\s*)?(?:n[o.]*\s*)?(\d{1,2}\.?\d{3})")
_PART = re.compile(r"\(\s*parte\s*\d+\s*\)")


def subject_key(raw: str) -> str:
    text = _PART.sub(" ", fold(raw or ""))
    return " ".join(text.split())


def load() -> dict[str, list[tuple[int, float]]]:
    df = pd.read_csv(artifact_path(FILE), dtype={"subject_key": str}).fillna("")
    out: dict[str, list[tuple[int, float]]] = {}
    for row in df.itertuples(index=False):
        out.setdefault(row.subject_key, []).append((int(row.tema_id), float(row.score)))
    return out


def build(annex_dir=ANNEX_DIR) -> str:
    tree = canonical_tree.load()
    laws = law_index.load()
    titles = {t: set(tokenize(tree.title(t))) for t in tree.candidates()}
    subjects = sorted({subject_key(q.origin_subject) for q in load_questions(annex_dir)} - {""})
    rows = []
    for key in subjects:
        rows += _law_matches(key, laws, tree)
        rows += _title_matches(key, titles)
    df = pd.DataFrame(rows, columns=["subject_key", "tema_id", "score", "method"])
    df = df.sort_values(["subject_key", "score"], ascending=[True, False]).drop_duplicates(["subject_key", "tema_id"])
    target = artifact_path(FILE)
    df.to_csv(target, index=False)
    write_provenance(target, produced_by="token overlap between subject and topic titles plus law_index",
                     llm_used=False, used_labels=False, subjects=len(subjects), mapped=int(df.subject_key.nunique()))
    return str(target)


def _law_matches(key: str, laws, tree) -> list[tuple]:
    rows = []
    for number in _LAW_NUMBER.findall(key):
        entry = laws.by_law_number(number)
        if entry and entry.scope == "in" and entry.tema_id in tree:
            rows.append((key, entry.tema_id, 1.0, "law"))
    for _, entry in laws.find_aliases(key):
        if entry.scope == "in" and entry.tema_id is not None:
            rows.append((key, entry.tema_id, 1.0, "law_alias"))
    return rows


def _title_matches(key: str, titles: dict[int, set[str]]) -> list[tuple]:
    tokens = set(tokenize(key))
    if not tokens:
        return []
    scored = []
    for tid, title_tokens in titles.items():
        if not title_tokens:
            continue
        coverage = len(tokens & title_tokens) / len(title_tokens)
        precision = len(tokens & title_tokens) / len(tokens)
        if coverage >= MIN_TITLE_COVERAGE:
            scored.append((key, tid, round(0.5 * coverage + 0.5 * precision, 4), "title"))
    return sorted(scored, key=lambda r: -r[2])[:MAX_MATCHES]
