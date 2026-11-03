"""Repeated cross validation, ablations, lockbox check, threshold selection and reliability.

What is train, calibration and test among the 100 labels (ADR-1):
  * lockbox (20): never used until `vde evaluate --open-lockbox`.
  * dev (80): repeated 10 fold CV. Inside every outer fold the engine fits channels and weights on
    the training part only and trains the correctness model on inner out of fold features.
  * The threshold is chosen on the dev out of fold confidences; its honest precision comes from a
    second level cross validation over questions (threshold CV), because picking the best of many
    thresholds on the same rows is optimistic.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from vde.aggregate import FEATURES
from vde.calibrate import (Platt, brier, choose_threshold, clopper_pearson_lower, expected_calibration_error,
                           ltt_threshold, reliability_table)
from vde.config import load_config
from vde.engine import Engine
from vde.resources import Resources

THRESHOLD_CV_FOLDS = 5


@dataclass
class CvResult:
    rows: pd.DataFrame
    seconds: float

    def per_question(self) -> pd.DataFrame:
        """One row per question: mean confidence over repeats and the majority hit."""
        grouped = self.rows.groupby("qid")
        return pd.DataFrame({
            "conf": grouped.conf.mean(),
            "hit": grouped.hit.mean() >= 0.5,
            "branch_hit": grouped.branch_hit.mean() >= 0.5,
            "is_out": grouped.is_out.first(),
            "is_new_import": grouped.is_new_import.first(),
            **{f: grouped[f].mean() for f in FEATURES},
        }).reset_index()


def cross_validate(res: Resources, ids: list[str], folds: int, repeats: int, seed: int,
                   channel_names: list[str] | None = None, node_mode: str = "branch") -> CvResult:
    engine = Engine(res, channel_names, node_mode)
    rng = np.random.default_rng(seed)
    rows = []
    start = time.time()
    for rep in range(repeats):
        order = _stratified_order(res, ids, rng)
        for k in range(folds):
            test = order[k::folds]
            held = set(test)
            train = [q for q in ids if q not in held]
            engine.fit(train, seed=seed + rep * folds + k)
            for pred in engine.predict(test):
                gold = res.gold[pred.qid]
                rows.append({"rep": rep, "fold": k, "qid": pred.qid, "conf": pred.confidence,
                             "hit": engine.is_hit(pred, gold), "branch_hit": engine.branch_hit(pred, gold),
                             "node": pred.choice.node if pred.choice else None,
                             "level": res.tree.level(pred.choice.node) if pred.choice else None,
                             "is_out": gold.is_out, "is_new_import": res.parsed[pred.qid].is_new_import,
                             **pred.features})
    return CvResult(pd.DataFrame(rows), time.time() - start)


def summarize(cv: CvResult, config: dict, weights: np.ndarray | None = None) -> dict:
    q = cv.per_question()
    dec = config["decision"]
    conf, hit = q.conf.to_numpy(), q.hit.to_numpy().astype(float)
    choice = choose_threshold(conf, hit, weights, dec["target_precision"], dec["posterior_level"],
                              dec["min_point_precision"])
    rows = cv.rows
    return {
        "questions": int(len(q)),
        "hit_rate": float(rows.hit.mean()),
        "branch_hit_rate": float(rows.branch_hit.mean()),
        "auc": float(roc_auc_score(rows.hit, rows.conf)) if rows.hit.nunique() > 1 else float("nan"),
        "aurc": aurc(rows.conf.to_numpy(), rows.hit.to_numpy()),
        "precision_at_coverage": {f"{c:.2f}": precision_at_coverage(conf, hit, c) for c in (0.1, 0.2, 0.3, 0.5)},
        "threshold": choice.to_dict(),
        "ltt_threshold": ltt_threshold(conf, hit, alpha=1 - dec["target_precision"], delta=0.10),
        "threshold_cv": threshold_cv(conf, hit, config, weights),
        "ece": expected_calibration_error(conf, hit),
        "brier": brier(conf, hit),
        "level2_share": float((rows.level == 2).mean()),
        "seconds": round(cv.seconds, 1),
    }


def aurc(conf: np.ndarray, hit: np.ndarray) -> float:
    """Area under the risk coverage curve (lower is better)."""
    order = np.argsort(-conf, kind="stable")
    errors = 1 - np.asarray(hit, dtype=float)[order]
    return float(np.mean(np.cumsum(errors) / np.arange(1, len(errors) + 1)))


def precision_at_coverage(conf: np.ndarray, hit: np.ndarray, coverage: float) -> float:
    order = np.argsort(-conf, kind="stable")
    k = max(1, int(round(coverage * len(conf))))
    return float(np.mean(np.asarray(hit, dtype=float)[order[:k]]))


def threshold_cv(conf: np.ndarray, hit: np.ndarray, config: dict, weights: np.ndarray | None = None,
                 seed: int = 0) -> dict:
    """Choose the threshold on 4/5 of the questions, measure it on the other 1/5, pool the results."""
    dec = config["decision"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(conf))
    published, hits = 0, 0.0
    for k in range(THRESHOLD_CV_FOLDS):
        test = order[k::THRESHOLD_CV_FOLDS]
        train = np.setdiff1d(order, test)
        w = None if weights is None else weights[train]
        tau = choose_threshold(conf[train], hit[train], w, dec["target_precision"], dec["posterior_level"],
                               dec["min_point_precision"]).tau
        mask = conf[test] >= tau
        published += int(mask.sum())
        hits += float(hit[test][mask].sum())
    return {"published": published, "hits": hits, "coverage": published / len(conf),
            "precision": hits / published if published else None,
            "cp_lower_90": clopper_pearson_lower(int(hits), published, 0.90) if published else 0.0}


def calibration_comparison(cv: CvResult, seed: int = 0) -> dict:
    """E13: raw logistic output against cross fitted Platt and isotonic on the out of fold scores."""
    from sklearn.isotonic import IsotonicRegression

    q = cv.per_question()
    conf, hit = q.conf.to_numpy(), q.hit.to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(conf))
    platt, iso = np.zeros_like(conf), np.zeros_like(conf)
    for k in range(5):
        test = order[k::5]
        train = np.setdiff1d(order, test)
        if hit[train].min() == hit[train].max():
            platt[test] = iso[test] = conf[test]
            continue
        platt[test] = Platt().fit(conf[train], hit[train]).predict(conf[test])
        iso[test] = IsotonicRegression(out_of_bounds="clip").fit(conf[train], hit[train]).predict(conf[test])
    return {name: {"ece": expected_calibration_error(c, hit), "brier": brier(c, hit),
                   "reliability": reliability_table(c, hit)}
            for name, c in (("raw", conf), ("platt", platt), ("isotonic", iso))}


ABLATIONS = {
    "E0_titles_only": {"channels": ["lexical_titles"]},
    "E1_bm25_knn": {"channels": ["lexical", "knn"]},
    "E1_bm25_knn_proto": {"channels": ["lexical", "knn", "prototypes"]},
    "full": {},
    "full_minus_special_law": {"drop": ["special_law"]},
    "full_minus_code_article": {"drop": ["code_article"]},
    "full_minus_knn": {"drop": ["knn"]},
    "full_minus_subject": {"drop": ["subject"]},
    "full_minus_prototypes": {"drop": ["prototypes"]},
    "full_minus_lexical": {"drop": ["lexical"]},
    "full_minus_duplicates": {"drop": ["duplicates"]},
    "full_plus_segments": {"add": ["segments"]},
    "full_plus_bin_prior": {"add": ["bin_prior"]},
    "full_plus_subject_confusion": {"config": {"subject": {"learn_confusion": True}}},
    "E3_leaf_top1": {"node_mode": "leaf"},
    "E3_always_level1": {"node_mode": "level1"},
    "linear_transform": {"config": {"aggregate": {"transform": "linear"}}},
}


def run_ablations(base_config: dict, ids: list[str], names: list[str] | None = None) -> pd.DataFrame:
    ev = base_config["evaluation"]
    out = []
    for name, spec in ABLATIONS.items():
        if names and name not in names:
            continue
        config = load_config(overrides=spec.get("config"))
        res = Resources.load(config)
        channels = spec.get("channels") or [c for c in config["channels"]["enabled"] if c not in spec.get("drop", [])]
        channels = channels + [c for c in spec.get("add", []) if c not in channels]
        cv = cross_validate(res, ids, ev["folds"], ev["ablation_repeats"], ev["seed"], channels,
                            spec.get("node_mode", "branch"))
        s = summarize(cv, config)
        out.append({"experiment": name, "channels": "+".join(channels), "hit_rate": s["hit_rate"],
                    "branch_hit_rate": s["branch_hit_rate"], "auc": s["auc"], "aurc": s["aurc"],
                    "prec_at_20": s["precision_at_coverage"]["0.20"], "coverage_at_tau": s["threshold"]["coverage"],
                    "level2_share": s["level2_share"], "seconds": s["seconds"]})
    return pd.DataFrame(out)


def evaluate_lockbox(res: Resources, dev: list[str], lockbox: list[str], tau: float) -> dict:
    engine = Engine(res)
    engine.fit(dev, seed=res.config["evaluation"]["seed"])
    rows = []
    for pred in engine.predict(lockbox):
        gold = res.gold[pred.qid]
        rows.append({"qid": pred.qid, "conf": pred.confidence, "hit": engine.is_hit(pred, gold),
                     "published": pred.confidence >= tau, "is_out": gold.is_out})
    df = pd.DataFrame(rows)
    pub = df[df.published]
    return {"n": int(len(df)), "hit_rate": float(df.hit.mean()), "published": int(len(pub)),
            "published_hits": int(pub.hit.sum()), "rows": df.to_dict("records")}


def save_json(path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _stratified_order(res: Resources, ids: list[str], rng) -> list[str]:
    """Shuffle within strata (discipline or out) and interleave, so every fold gets a fair mix."""
    strata: dict[str, list[str]] = {}
    for qid in ids:
        gold = res.gold[qid]
        key = "out" if gold.is_out else res.tree.discipline(gold.central)
        strata.setdefault(key, []).append(qid)
    order = []
    for key in sorted(strata):
        order += list(rng.permutation(strata[key]))
    return order


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not serializable: {type(value)}")
