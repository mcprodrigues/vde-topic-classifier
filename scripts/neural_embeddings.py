"""Compare the LSA embeddings with multilingual E5 small, base and large (outputs/embeddings_neural.csv).

Offline experiment, never part of `vde publish`. It needs the optional dependency:
    pip install -e ".[neural]"

Each backend runs in its own process with temporary artifact and output directories (a copy of
artifacts/), so the versioned artifacts are never touched. For every backend: build the embeddings,
run the same repeated CV and final fit as `vde build model`, publish into the temporary directory and
record the CV metrics and the number of published questions among the 559 evaluated ones.

    python scripts/neural_embeddings.py                      # all backends, writes outputs/embeddings_neural.csv
    python scripts/neural_embeddings.py --only lsa           # a single backend
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKENDS = {
    "lsa": {"backend": "lsa"},
    "intfloat/multilingual-e5-small": {"backend": "st", "model": "intfloat/multilingual-e5-small"},
    "intfloat/multilingual-e5-base": {"backend": "st", "model": "intfloat/multilingual-e5-base"},
    "intfloat/multilingual-e5-large": {"backend": "st", "model": "intfloat/multilingual-e5-large"},
}
COLUMNS = ["embeddings", "hit", "auc", "ece", "p10", "p20", "aurc", "nested_precision", "nested_published",
           "published", "embed_seconds"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="comma separated backend names (default: all)")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "embeddings_neural.csv"))
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run_one(args.worker)))
        return 0
    names = [n.strip() for n in args.only.split(",")] if args.only else list(BACKENDS)
    rows = []
    for name in names:
        rows.append(_in_subprocess(name))
        print(rows[-1], flush=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return 0


def run_one(name: str) -> dict:
    """Runs inside the worker process, whose VDE_ARTIFACT_DIR and VDE_OUTPUT_DIR point to a temporary copy."""
    from vde.config import load_config
    from vde.knowledge import embeddings
    from vde.knowledge.base import refresh_manifest
    from vde.pipeline import publish, train_model

    spec = BACKENDS[name]
    config = load_config(overrides={"embeddings": spec})
    emb = config["embeddings"]
    start = time.time()
    embeddings.build(emb["backend"], emb.get("dimensions", 256), emb.get("seed", 0),
                     emb.get("model", embeddings.DEFAULT_ST_MODEL))
    embed_seconds = round(time.time() - start)
    refresh_manifest()
    model = train_model(config)
    decisions = publish(config=config)
    s = model["cv_summary"]
    evaluated = [d for d in decisions if d.id not in set(model["labeled_ids"])]
    return {
        "embeddings": name + (" (adotado)" if name == "lsa" else ""),
        "hit": round(s["hit_rate"], 4), "auc": round(s["auc"], 4), "ece": round(s["ece"], 4),
        "p10": round(s["precision_at_coverage"]["0.10"], 4), "p20": round(s["precision_at_coverage"]["0.20"], 4),
        "aurc": round(s["aurc"], 4), "nested_precision": round(s["threshold_cv"]["precision"] or 0.0, 4),
        "nested_published": round(s["threshold_cv"]["published_per_repeat"], 2),
        "published": sum(d.decision == "publicar" for d in evaluated), "embed_seconds": embed_seconds,
    }


def _in_subprocess(name: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="vde-neural-") as tmp:
        artifacts, outputs = Path(tmp) / "artifacts", Path(tmp) / "outputs"
        shutil.copytree(ROOT / "artifacts", artifacts)
        outputs.mkdir()
        env = os.environ | {"VDE_ARTIFACT_DIR": str(artifacts), "VDE_OUTPUT_DIR": str(outputs)}
        done = subprocess.run([sys.executable, __file__, "--worker", name], env=env, check=True,
                              capture_output=True, text=True)
        return json.loads(done.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    raise SystemExit(main())
