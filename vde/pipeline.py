"""Train the final model (offline) and publish (no LLM, no network, artifacts only)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from vde import splits
from vde.config import load_config
from vde.decide import Decision, decide
from vde.engine import Engine
from vde.evaluate import CvResult, cross_validate, save_json, summarize
from vde.knowledge.base import refresh_manifest, verify_manifest, write_provenance
from vde.paths import ARTIFACT_DIR, OUTPUT_DIR
from vde.resources import Resources

MODEL_FILE = "model.json"
OOF_FILE = "cv_oof.csv"
PUBLICATION_FILE = "publicacao.csv"
COLUMNS = ["id", "decisao", "tema_central_id", "extras_ids", "confianca", "motivo"]


def train_model(config: dict | None = None) -> dict:
    """Repeated CV on the dev split picks the threshold, then the engine is refitted on all labels."""
    config = config or load_config()
    res = Resources.load(config)
    ev = config["evaluation"]
    dev, lockbox = splits.load_or_create(res, ev["lockbox_size"], ev["seed"])
    cv = cross_validate(res, dev, ev["folds"], ev["repeats"], ev["seed"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv.rows.to_csv(OUTPUT_DIR / OOF_FILE, index=False)
    summary = summarize(cv, config)
    engine = Engine(res)
    fit_info = engine.fit(sorted(res.gold), seed=ev["seed"])
    model = {
        "state": engine.export_state(),
        "threshold": summary["threshold"]["tau"],
        "threshold_detail": summary["threshold"],
        "labeled_ids": sorted(res.gold),
        "dev_ids": dev,
        "lockbox_ids": lockbox,
        "splits_sha256": splits.fingerprint(),
        "cv_summary": {k: v for k, v in summary.items() if k != "threshold"},
        "fit_info": fit_info,
    }
    target = ARTIFACT_DIR / MODEL_FILE
    save_json(target, model)
    write_provenance(target, produced_by="vde build model: repeated CV on dev plus a final fit on 100 labels",
                     llm_used=False, repeats=ev["repeats"], folds=ev["folds"])
    refresh_manifest()
    return model


def publish(out_path: Path | None = None, config: dict | None = None, strict_manifest: bool = True) -> list[Decision]:
    """Regenerate publicacao.csv from the annexes and the versioned artifacts only."""
    problems = verify_manifest()
    if problems and strict_manifest:
        raise RuntimeError(f"artifacts differ from MANIFEST.json: {problems}. Rebuild or refresh the manifest.")
    config = config or load_config()
    res = Resources.load(config)
    model = json.loads((ARTIFACT_DIR / MODEL_FILE).read_text(encoding="utf-8"))
    engine = Engine(res, model["state"]["scorer"]["channels"], model["state"]["node_mode"])
    engine.load_state(model["state"], model["labeled_ids"])
    decisions = [decide(p, res, model["threshold"]) for p in engine.predict(res.question_ids)]
    write_publication(decisions, out_path or OUTPUT_DIR / PUBLICATION_FILE)
    return decisions


def write_publication(decisions: list[Decision], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(COLUMNS)
        for d in decisions:
            writer.writerow([d.id, d.decision, "" if d.central is None else d.central,
                             ";".join(str(e) for e in d.extras), f"{d.confidence:.4f}", d.reason])


def load_oof() -> CvResult:
    import pandas as pd

    return CvResult(pd.read_csv(OUTPUT_DIR / OOF_FILE), 0.0)


def dev_arrays(cv: CvResult) -> tuple[np.ndarray, np.ndarray, "object"]:
    q = cv.per_question()
    return q.conf.to_numpy(), q.hit.to_numpy().astype(float), q
