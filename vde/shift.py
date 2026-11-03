"""Robustness to the new import (direction 10): is Q0360 to Q0659 different, and how much?

A domain classifier separates the labeled bank questions from the new import using the same features
the correctness model sees. Its odds become importance weights for the labeled rows, so the precision
estimate for the new import is reweighted toward the questions that look like it.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from vde.aggregate import FEATURES

WEIGHT_CLIP = (0.1, 10.0)


def domain_weights(labeled_features: pd.DataFrame, new_features: pd.DataFrame, seed: int = 0) -> dict:
    """Importance weights p(new | x) / p(bank | x) for labeled rows, and the domain AUC."""
    x = pd.concat([labeled_features[FEATURES], new_features[FEATURES]]).to_numpy(dtype=float)
    y = np.r_[np.zeros(len(labeled_features)), np.ones(len(new_features))]
    x = StandardScaler().fit_transform(x)
    model = LogisticRegression(C=0.3, max_iter=2000)
    proba = cross_val_predict(model, x, y, cv=5, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    p_lab = np.clip(proba[: len(labeled_features)], 1e-3, 1 - 1e-3)
    prior = len(labeled_features) / len(new_features)
    weights = np.clip(p_lab / (1 - p_lab) * prior, *WEIGHT_CLIP)
    weights = weights / weights.mean()
    n_eff = float(weights.sum() ** 2 / (weights ** 2).sum())
    return {"auc": auc, "weights": weights, "effective_n": n_eff}


def describe(res, old_ids: list[str], new_ids: list[str]) -> dict:
    """Plain statistics of both sources, for the report."""
    def stats(ids):
        parsed = [res.parsed[q] for q in ids]
        return {
            "n": len(ids),
            "median_length": float(np.median([p.length for p in parsed])),
            "share_true_false": float(np.mean([p.kind == "true_false" for p in parsed])),
            "share_assertions": float(np.mean([p.kind == "assertions" for p in parsed])),
            "share_empty_subject": float(np.mean([not p.origin_subject for p in parsed])),
            "origin_discipline": dict(Counter(p.origin_discipline or "(vazio)" for p in parsed).most_common()),
        }
    return {"bank": stats(old_ids), "new_import": stats(new_ids)}
