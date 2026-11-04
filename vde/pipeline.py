"""Train the final model (offline) and publish (no LLM, no network, artifacts only)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from vde import splits
from vde.calibrate import Platt
from vde.config import fingerprint, load_config
from vde.decide import Decision, decide
from vde.engine import Engine
from vde.evaluate import CvResult, cross_validate, crossfit_recalibration, save_json, summarize
from vde.knowledge.base import refresh_manifest, split_problems, verify_manifest, write_provenance
from vde.paths import ARTIFACT_DIR, OUTPUT_DIR
from vde.resources import Resources

MODEL_FILE = "model.json"
OOF_FILE = "cv_oof.csv"
PUBLICATION_FILE = "publicacao.csv"
COLUMNS = ["id", "decisao", "tema_central_id", "extras_ids", "confianca", "motivo"]
NEEDS_TOPIC = ("publicar", "revisar")
NEEDS_REASON = ("revisar", "fora")


def train_model(config: dict | None = None) -> dict:
    """Repeated CV on the dev split picks the threshold, then the engine is refitted on all labels.

    With `aggregate.recalibration = "platt"`, the deployed confidence is followed by a Platt recalibration
    fitted on the outer out of fold rows: those rows come from engines served by a scorer fitted on more
    labels than their correctness model saw, which is exactly the situation of the final engine.
    """
    config = config or load_config()
    res = Resources.load(config)
    ev = config["evaluation"]
    dev, lockbox = splits.load_or_create(res, ev["lockbox_size"], ev["seed"])
    cv = cross_validate(res, dev, ev["folds"], ev["repeats"], ev["seed"])
    recalibrate = config["aggregate"].get("recalibration", "none") == "platt"
    rows = crossfit_recalibration(cv.rows, ev["seed"]) if recalibrate else cv.rows.assign(conf_raw=cv.rows.conf)
    cv = CvResult(rows, cv.seconds)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv.rows.to_csv(OUTPUT_DIR / OOF_FILE, index=False)
    summary = summarize(cv, config)
    engine = Engine(res)
    fit_info = engine.fit(sorted(res.gold), seed=ev["seed"])
    if recalibrate:
        engine.recalibration = Platt().fit(cv.rows.conf_raw.to_numpy(), cv.rows.hit.to_numpy())
    model = {
        "state": engine.export_state(),
        "threshold": summary["threshold"]["tau"],
        "threshold_detail": summary["threshold"],
        "config_sha256": fingerprint(config),
        "versions": _versions(),
        "labeled_ids": sorted(res.gold),
        "dev_ids": dev,
        "lockbox_ids": lockbox,
        "lockbox_note": "merged back in v0.2, see splits.json" if not lockbox else "",
        "splits_sha256": splits.fingerprint(),
        "cv_summary": {k: v for k, v in summary.items() if k != "threshold"},
        "fit_info": fit_info,
    }
    target = ARTIFACT_DIR / MODEL_FILE
    save_json(target, model)
    write_provenance(target, produced_by="vde build model: repeated CV on dev plus a final fit on 100 labels",
                     llm_used=False, repeats=ev["repeats"], folds=ev["folds"], versions=model["versions"])
    refresh_manifest()
    return model


def publish(out_path: Path | None = None, config: dict | None = None,
            allow_annex_change: bool = False) -> list[Decision]:
    """Regenerate publicacao.csv from the annexes and the versioned artifacts only.

    Artifacts must always match the manifest. Annexes may differ only with `allow_annex_change` (a new
    import). The configuration must be the one the model was trained with.
    """
    artifact_problems, annex_problems = split_problems(verify_manifest())
    if artifact_problems:
        raise RuntimeError(f"artifacts differ from MANIFEST.json: {artifact_problems}. Rebuild them.")
    if annex_problems and not allow_annex_change:
        raise RuntimeError(f"annexes differ from MANIFEST.json: {annex_problems}. Rebuild, or pass "
                           "--allow-annex-change to classify a new import with the stored artifacts.")
    config = config or load_config()
    res = Resources.load(config)
    model = json.loads((ARTIFACT_DIR / MODEL_FILE).read_text(encoding="utf-8"))
    engine = load_engine(res, model, config)
    decisions = [decide(p, res, model["threshold"]) for p in engine.predict(res.question_ids)]
    write_publication(decisions, out_path or OUTPUT_DIR / PUBLICATION_FILE)
    return decisions


def load_engine(res: Resources, model: dict, config: dict) -> Engine:
    """The deployed engine, exactly as publish uses it. Refuses a configuration the model was not trained with."""
    expected = model.get("config_sha256")
    if expected != fingerprint(config):
        raise RuntimeError("vde/config.toml differs from the configuration model.json was trained with "
                           f"(expected {expected}, got {fingerprint(config)}). Run `vde build model`.")
    engine = Engine(res, model["state"]["scorer"]["channels"], model["state"]["node_mode"])
    engine.load_state(model["state"], model["labeled_ids"])
    return engine


def write_publication(decisions: list[Decision], path: Path) -> None:
    """Write the CSV in the LEIA-ME format, and refuse to write a row that breaks it."""
    for d in decisions:
        if d.decision in NEEDS_TOPIC and d.central is None:
            raise ValueError(f"{d.id}: '{d.decision}' needs tema_central_id ({d.reason})")
        if d.decision in NEEDS_REASON and not d.reason:
            raise ValueError(f"{d.id}: '{d.decision}' needs a motivo")
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


def _versions() -> dict:
    """Library versions the model was trained with (pinned in requirements.lock)."""
    import sys

    import pandas
    import scipy
    import sklearn

    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pandas.__version__, "scikit-learn": sklearn.__version__}
