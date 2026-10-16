"""Week 4 error analysis: where does the improved model differ from the baseline?

Uses only predictions the pipeline already produced:
  * grouped-CV out-of-fold predictions on TRAIN+VAL (every dev notebook predicted by a model
    that never saw its student), and
  * hold-out predictions on validation and test (test computed once, after freezing).
A prediction "succeeds" when the rounded prediction is within +-1 mark of the human mark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import round_half_up

DIMS = ["code", "explanation", "documentation"]


def _category(ok_b: bool, ok_i: bool) -> str:
    if ok_b and ok_i:
        return "both succeed"
    if not ok_b and ok_i:
        return "baseline fails, improved succeeds"
    if ok_b and not ok_i:
        return "baseline succeeds, improved fails"
    return "both fail"


def compare_predictions(run, baseline: str, improved: str) -> pd.DataFrame:
    rows = []
    feat_cols = [c for c in ("tfidf_cos_mean", "keyword_coverage", "sem_cos_mean", "sem_concept_coverage",
                             "nli_support", "nli_coverage", "nli_contradiction", "output_match_rate",
                             "error_cell_rate") if c in run.df.columns]
    feats = run.df.set_index("notebook_id")
    key = run.manifest.set_index("notebook_id")
    label = key["raw_path"].str.extract(r"(S\d+_LAB_\w+?)\.ipynb")[0] if run.status.startswith("SYNTHETIC") else None
    # out-of-fold (train+val)
    ob, oi = run.oof.get(baseline), run.oof.get(improved)
    if ob and oi:
        for d in DIMS:
            y = run.dev[d].to_numpy(float)
            for n, t, pb, pi in zip(ob["notebook_id"], y, ob[d], oi[d]):
                rows.append((n, "cv_train_val", d, t, pb, pi))
    # hold-out
    pr = pd.read_csv(run.out / "predictions.csv")
    pr = pr[pr.variant.isin([baseline, improved])]
    for (n, s, d), g in pr.groupby(["notebook_id", "split", "dimension"]):
        g = g.set_index("variant")
        if baseline in g.index and improved in g.index:
            rows.append((n, s, d, float(g.loc[baseline, "human"]), float(g.loc[baseline, "predicted"]),
                         float(g.loc[improved, "predicted"])))
    out = []
    for n, s, d, t, pb, pi in rows:
        ok_b, ok_i = abs(round_half_up(pb) - t) <= 1, abs(round_half_up(pi) - t) <= 1
        r = {"notebook_id": n, "label": None if label is None else label.get(n), "eval": s, "dimension": d,
             "human": t, f"pred_{baseline}": round(pb, 2), f"pred_{improved}": round(pi, 2),
             "abs_err_baseline": round(abs(pb - t), 2), "abs_err_improved": round(abs(pi - t), 2),
             "category": _category(ok_b, ok_i)}
        r.update({c: round(float(feats.loc[n, c]), 3) for c in feat_cols})
        out.append(r)
    return pd.DataFrame(out)
