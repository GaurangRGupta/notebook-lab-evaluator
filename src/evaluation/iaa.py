"""Inter-annotator agreement and label aggregation.

Reported on PRE-discussion labels. The mean pairwise annotator MAE is an estimate of
human annotation variability on this rubric; it is a useful reference point for the
model's MAE, not a hard performance ceiling.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.metrics import cohen_kappa_score


def krippendorff_alpha(matrix: np.ndarray, level: str = "interval") -> float | None:
    """matrix: items x annotators, NaN = missing. level: 'interval' or 'ordinal'."""
    m = np.asarray(matrix, float)
    units = [row[~np.isnan(row)] for row in m]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None
    values = np.unique(np.concatenate(units))
    if len(values) < 2:
        return None
    idx = {v: i for i, v in enumerate(values)}
    # coincidence matrix
    o = np.zeros((len(values), len(values)))
    for u in units:
        mu = len(u)
        for a in range(mu):
            for b in range(mu):
                if a != b:
                    o[idx[u[a]], idx[u[b]]] += 1.0 / (mu - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if level == "interval":
        delta = (values[:, None] - values[None, :]) ** 2
    elif level == "ordinal":
        delta = np.zeros_like(o)
        for i in range(len(values)):
            for j in range(len(values)):
                lo, hi = min(i, j), max(i, j)
                s = n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2
                delta[i, j] = s ** 2
    else:
        raise ValueError(level)
    d_o = (o * delta).sum() / n
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return None if d_e == 0 else float(1 - d_o / d_e)


def icc_2_1(matrix: np.ndarray) -> float | None:
    """ICC(2,1): two-way random effects, absolute agreement, single rater (complete data)."""
    x = np.asarray(matrix, float)
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape
    if n < 2 or k < 2:
        return None
    gm = x.mean()
    msr = k * ((x.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return None if denom == 0 else float((msr - mse) / denom)


def agreement_report(df, dimension: str, max_score: int, item_col="notebook_id", ann_col="annotator") -> dict:
    wide = df.pivot(index=item_col, columns=ann_col, values=dimension)
    mat = wide.to_numpy(float)
    anns = list(wide.columns)
    pair = {}
    maes, within1, qwk = [], [], []
    for a, b in combinations(range(len(anns)), 2):
        ok = ~np.isnan(mat[:, a]) & ~np.isnan(mat[:, b])
        ya, yb = mat[ok, a].astype(int), mat[ok, b].astype(int)
        k = cohen_kappa_score(ya, yb, weights="quadratic", labels=list(range(max_score + 1)))
        mae = float(np.mean(np.abs(ya - yb)))
        w1 = float(np.mean(np.abs(ya - yb) <= 1))
        pair[f"{anns[a]}-{anns[b]}"] = {"qwk": float(k), "mae": mae, "within_1": w1}
        maes.append(mae); within1.append(w1); qwk.append(k)
    return {
        "dimension": dimension, "n_items": int(len(wide)), "annotators": anns,
        "krippendorff_alpha_ordinal": krippendorff_alpha(mat, "ordinal"),
        "krippendorff_alpha_interval": krippendorff_alpha(mat, "interval"),
        "icc_2_1": icc_2_1(mat),
        "mean_pairwise_qwk": float(np.mean(qwk)) if qwk else None,
        "mean_pairwise_mae": float(np.mean(maes)) if maes else None,
        "mean_pairwise_within_1": float(np.mean(within1)) if within1 else None,
        "pairwise": pair,
        "note": "mean_pairwise_mae estimates human annotation variability; it is not a hard ceiling for the model.",
    }


def aggregate_labels(df, dimensions, item_col="notebook_id", tolerance=1, adjudicated=None):
    """Median of annotators. Items whose annotator range exceeds `tolerance` are flagged
    for discussion; an adjudicated table (item_col + dimensions) overrides those items."""
    import pandas as pd
    rows = []
    for item, g in df.groupby(item_col):
        r = {item_col: item}
        for d in dimensions:
            v = g[d].to_numpy(float)
            r[d] = float(np.median(v))
            r[f"{d}_range"] = float(v.max() - v.min())
            r[f"{d}_needs_discussion"] = bool(v.max() - v.min() > tolerance)
            r[f"{d}_source"] = "median"
        rows.append(r)
    out = pd.DataFrame(rows)
    if adjudicated is not None and len(adjudicated):
        adj = adjudicated.set_index(item_col)
        for i, r in out.iterrows():
            if r[item_col] in adj.index:
                for d in dimensions:
                    if d in adj.columns and not pd.isna(adj.loc[r[item_col], d]):
                        out.at[i, d] = float(adj.loc[r[item_col], d])
                        out.at[i, f"{d}_source"] = "adjudicated"
    return out
