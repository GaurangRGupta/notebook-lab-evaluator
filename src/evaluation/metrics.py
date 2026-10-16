"""Metrics for rubric-score regression and for binary sub-checks."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_recall_fscore_support


def _corr(fn, y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return None  # undefined / meaningless
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(fn(y, p)[0])


def round_half_up(x):
    """Marks are rounded the way a grader would (2.5 -> 3). Week 5 fix: np.rint rounds half to even
    (2.5 -> 2, 3.5 -> 4), which made agreement metrics depend on parity."""
    return np.floor(np.asarray(x, float) + 0.5)


def regression_metrics(y_true, y_pred, max_score: float) -> dict:
    """y_pred should already be clipped to [0, max_score].
    within_1: share of notebooks with |round_half_up(pred) - human| <= 1.
    exact:    share with round_half_up(pred) == human."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    pr = round_half_up(p)
    err = p - y
    return {
        "n": int(len(y)),
        "MAE": float(np.mean(np.abs(err))) if len(y) else None,
        "RMSE": float(np.sqrt(np.mean(err ** 2))) if len(y) else None,
        "pearson_r": _corr(pearsonr, y, p),
        "spearman_rho": _corr(spearmanr, y, p),
        "within_1": float(np.mean(np.abs(pr - y) <= 1)) if len(y) else None,
        "exact": float(np.mean(pr == y)) if len(y) else None,
        "MAE_as_fraction_of_max": float(np.mean(np.abs(err)) / max_score) if len(y) else None,
    }


def bootstrap_ci(y_true, y_pred, metric="MAE", n_boot=2000, seed=0, min_n=10):
    """95% percentile CI. Returns None when n < min_n: a CI on a handful of points is not meaningful."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y) < min_n:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        e = p[i] - y[i]
        vals.append(np.mean(np.abs(e)) if metric == "MAE" else np.sqrt(np.mean(e ** 2)))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def binary_metrics(y_true, y_pred) -> dict:
    y, p = np.asarray(y_true, int), np.asarray(y_pred, int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return {"n": int(len(y)), "precision": float(pr), "recall": float(rc), "f1": float(f1),
            "support_positive": int(y.sum())}
