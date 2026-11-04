"""Repeated cross validation, ablations, threshold selection and reliability.

What is train, calibration and test among the 100 labels (ADR-1, revised in v0.2):
  * The 20 question lockbox of v0.1 was inspected during error analysis, so it is no longer blind.
    It was merged back: all 100 labels go through repeated 10 fold CV.
  * Inside every outer fold the engine fits channels and weights on the training part only and
    trains the correctness model on inner out of fold features.
  * Metrics are computed one repeat at a time (each repeat is one deployable model), never on the
    confidence averaged over repeats, which no deployed model produces.
  * The threshold is chosen on the out of fold rows; its honest precision comes from a nested split
    over questions (choose on 4/5 of the questions, measure on the other 1/5, repeated).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from vde.aggregate import FEATURES
from vde.calibrate import (Platt, brier, choose_threshold_per_repeat, clopper_pearson_lower,
                           expected_calibration_error, ltt_threshold, reliability_table, threshold_stats)
from vde.config import load_config
from vde.engine import Engine
from vde.resources import Resources

THRESHOLD_CV_FOLDS = 5
THRESHOLD_CV_SPLITS = 10


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


def summarize(cv: CvResult, config: dict) -> dict:
    rows = cv.rows
    dec = config["decision"]
    q = cv.per_question()
    curve = per_repeat_curve(rows)
    choice = select_threshold(rows, dec)
    return {
        "questions": int(rows.qid.nunique()),
        "repeats": int(rows.rep.nunique()),
        "hit_rate": float(rows.hit.mean()),
        "branch_hit_rate": float(rows.branch_hit.mean()),
        "auc": float(roc_auc_score(rows.hit, rows.conf)) if rows.hit.nunique() > 1 else float("nan"),
        "aurc": curve["aurc_mean"],
        "precision_at_coverage": curve["mean"],
        "precision_at_coverage_q10": curve["q10"],
        "threshold": choice.to_dict(),
        "ltt_threshold": ltt_threshold(rows.conf.to_numpy(), rows.hit.to_numpy(), alpha=1 - dec["target_precision"],
                                       delta=0.10),
        "threshold_rule": dec["threshold_rule"],
        "threshold_cv": threshold_cv(rows, dec),
        "threshold_cv_by_rule": {r: threshold_cv(rows, dec, r) for r in ("calibrated", "search")},
        "ece": expected_calibration_error(rows.conf.to_numpy(), rows.hit.to_numpy()),
        "brier": brier(rows.conf.to_numpy(), rows.hit.to_numpy()),
        "ece_mean_over_repeats": expected_calibration_error(q.conf.to_numpy(), q.hit.to_numpy().astype(float)),
        "level2_share": float((rows.level == 2).mean()),
        "seconds": round(cv.seconds, 1),
    }


def per_repeat_curve(rows: pd.DataFrame, coverages=(0.1, 0.2, 0.3, 0.5)) -> dict:
    """Precision at coverage and AURC computed inside each repeat, then summarized across repeats."""
    table = {c: [] for c in coverages}
    aurcs = []
    for _, rep in rows.groupby("rep"):
        conf, hit = rep.conf.to_numpy(), rep.hit.to_numpy().astype(float)
        aurcs.append(aurc(conf, hit))
        for c in coverages:
            table[c].append(precision_at_coverage(conf, hit, c))
    return {
        "mean": {f"{c:.2f}": float(np.mean(v)) for c, v in table.items()},
        "q10": {f"{c:.2f}": float(np.quantile(v, 0.10)) for c, v in table.items()},
        "aurc_mean": float(np.mean(aurcs)),
    }


def select_threshold(rows: pd.DataFrame, dec: dict, rule: str | None = None):
    """`calibrated`: publish when the calibrated P(hit) reaches the target, a rule fixed before looking at
    the curve, so its nested estimate has no selection optimism. `search`: walk the OOF curve."""
    eligible = rows["eligible"].to_numpy() if "eligible" in rows else None
    args = (rows.conf.to_numpy(), rows.hit.to_numpy(), rows.rep.to_numpy(), eligible)
    if (rule or dec["threshold_rule"]) == "calibrated":
        return threshold_stats(*args, dec["target_precision"], dec["target_precision"])
    return choose_threshold_per_repeat(*args, dec["target_precision"], dec["min_point_precision"],
                                       dec["min_published"])


def aurc(conf: np.ndarray, hit: np.ndarray) -> float:
    """Area under the risk coverage curve (lower is better)."""
    order = np.argsort(-conf, kind="stable")
    errors = 1 - np.asarray(hit, dtype=float)[order]
    return float(np.mean(np.cumsum(errors) / np.arange(1, len(errors) + 1)))


def precision_at_coverage(conf: np.ndarray, hit: np.ndarray, coverage: float) -> float:
    order = np.argsort(-conf, kind="stable")
    k = max(1, int(round(coverage * len(conf))))
    return float(np.mean(np.asarray(hit, dtype=float)[order[:k]]))


def threshold_cv(rows: pd.DataFrame, dec: dict, rule: str | None = None, seed: int = 0) -> dict:
    """Nested check of the threshold: choose it on 4/5 of the questions, measure it on the other 1/5.

    The split is over questions (all repeats of a question stay together) and is itself repeated.
    Precision is reported pooled and per CV repeat, so the spread of one deployable model is visible.
    """
    rng = np.random.default_rng(seed)
    qids = np.array(sorted(rows.qid.unique()))
    measured = []
    taus = []
    for _ in range(THRESHOLD_CV_SPLITS):
        order = rng.permutation(qids)
        for k in range(THRESHOLD_CV_FOLDS):
            test = set(order[k::THRESHOLD_CV_FOLDS])
            in_test = rows.qid.isin(test)
            tau = select_threshold(rows[~in_test], dec, rule).tau
            taus.append(tau)
            part = rows[in_test]
            mask = part.conf >= tau
            if "eligible" in part:
                mask &= part.eligible.astype(bool)
            measured.append(part.assign(pub=mask))
    df = pd.concat(measured)
    per_split = len(df) / THRESHOLD_CV_SPLITS
    published = float(df.pub.sum()) / THRESHOLD_CV_SPLITS
    hits = float(df[df.pub].hit.sum()) / THRESHOLD_CV_SPLITS
    by_rep = df[df.pub].groupby("rep").hit.agg(["sum", "size"])
    rep_prec = (by_rep["sum"] / by_rep["size"]).to_numpy() if len(by_rep) else np.array([])
    n_reps = rows.rep.nunique()
    return {
        "published_per_repeat": published / n_reps,
        "hits_per_repeat": hits / n_reps,
        "coverage": published / per_split if per_split else 0.0,
        "precision": hits / published if published else None,
        "precision_q10_over_repeats": float(np.quantile(rep_prec, 0.10)) if len(rep_prec) else None,
        "precision_min_over_repeats": float(rep_prec.min()) if len(rep_prec) else None,
        "cp_lower_90": clopper_pearson_lower(int(round(hits / n_reps)), int(round(published / n_reps)), 0.90)
        if published else 0.0,
        "tau_spread": [float(np.min(taus)), float(np.median(taus)), float(np.max(taus))],
    }


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


def run_ablations(base_config: dict, ids: list[str], names: list[str] | None = None,
                  repeats: int | None = None) -> pd.DataFrame:
    ev = base_config["evaluation"]
    out = []
    for name, spec in ABLATIONS.items():
        if names and name not in names:
            continue
        config = load_config(overrides=spec.get("config"))
        res = Resources.load(config)
        channels = spec.get("channels") or [c for c in config["channels"]["enabled"] if c not in spec.get("drop", [])]
        channels = channels + [c for c in spec.get("add", []) if c not in channels]
        cv = cross_validate(res, ids, ev["folds"], repeats or ev["ablation_repeats"], ev["seed"], channels,
                            spec.get("node_mode", "branch"))
        s = summarize(cv, config)
        out.append({"experiment": name, "channels": "+".join(channels), "hit_rate": s["hit_rate"],
                    "branch_hit_rate": s["branch_hit_rate"], "auc": s["auc"], "aurc": s["aurc"],
                    "prec_at_10": s["precision_at_coverage"]["0.10"], "prec_at_20": s["precision_at_coverage"]["0.20"],
                    "nested_precision": s["threshold_cv"]["precision"],
                    "nested_published": s["threshold_cv"]["published_per_repeat"],
                    "level2_share": s["level2_share"], "seconds": s["seconds"]})
    return pd.DataFrame(out)


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
