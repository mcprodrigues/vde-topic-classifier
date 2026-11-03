"""Load the annex files into typed records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from vde.paths import ANNEX_DIR

NEW_IMPORT_FIRST_ID = "Q0360"


@dataclass(frozen=True)
class Question:
    id: str
    stem_html: str
    command: str
    alternatives: dict
    origin_discipline: str
    origin_subject: str

    @property
    def is_new_import(self) -> bool:
        return self.id >= NEW_IMPORT_FIRST_ID


@dataclass(frozen=True)
class GoldLabel:
    id: str
    central: int | None
    extras: tuple[int, ...]
    is_out: bool


def load_questions(annex_dir: Path = ANNEX_DIR) -> list[Question]:
    """Read questoes.jsonl, keeping the file order."""
    questions = []
    with open(annex_dir / "questoes.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                questions.append(_question_from_json(json.loads(line)))
    return questions


def load_topics(annex_dir: Path = ANNEX_DIR) -> pd.DataFrame:
    """Read temas.csv with integer ids and a nullable parent id."""
    df = pd.read_csv(annex_dir / "temas.csv", dtype={"tema_id": int})
    df["parent_id"] = df["parent_id"].astype("Int64")
    df["nivel"] = df["nivel"].astype(int)
    return df


def load_gold(annex_dir: Path = ANNEX_DIR) -> dict[str, GoldLabel]:
    """Read folha_verdade.csv into GoldLabel records keyed by question id."""
    df = pd.read_csv(annex_dir / "folha_verdade.csv", dtype=str).fillna("")
    gold = {}
    for row in df.itertuples(index=False):
        extras = tuple(int(x) for x in row.extras_ids.split(";") if x.strip())
        central = int(row.tema_central_id) if row.tema_central_id.strip() else None
        gold[row.id] = GoldLabel(row.id, central, extras, row.fora_da_taxonomia.strip() == "1")
    return gold


def _question_from_json(raw: dict) -> Question:
    return Question(
        id=raw["id"],
        stem_html=raw.get("enunciado_html") or "",
        command=raw.get("comando") or "",
        alternatives=dict(raw.get("alternativas") or {}),
        origin_discipline=(raw.get("disciplina_origem") or "").strip(),
        origin_subject=(raw.get("assunto_origem") or "").strip(),
    )
