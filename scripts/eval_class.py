"""Compare human marks in data/class/annotations/marking_sheet.csv with the model predictions in
results/class_mu/predictions.csv.

Only rows with status == "marked" and all three marks filled are used. With no marked rows the
script writes a "no human marks yet" result and exits 0, and it never invents numbers.

Metrics per dimension (code 0-8, explanation 0-8, documentation 0-4, overall 0-20) and per variant
(tfidf, minilm): n, MAE, RMSE, Pearson r, Spearman rho, share within +/-1 mark, and quadratic
weighted kappa (QWK) on marks rounded to integers. Correlations are None when n < 3 or a side is
constant.

    python scripts/eval_class.py [--sheet ...] [--pred ...] [--out results/class_mu/eval.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
DIMS = {"code": ("code_0_8", 8), "explanation": ("explanation_0_8", 8),
        "documentation": ("documentation_0_4", 4), "overall": (None, 20)}
VARIANTS = ("tfidf", "minilm")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def metrics(human, pred, max_score):
    h = np.asarray(human, float)
    p = np.asarray(pred, float)
    n = len(h)
    if n == 0:
        return {"n": 0}
    err = p - h
    out = {"n": n, "mae": round(float(np.mean(np.abs(err))), 3),
           "rmse": round(float(math.sqrt(np.mean(err ** 2))), 3),
           "within_1": round(float(np.mean(np.abs(err) <= 1)), 3),
           "pearson": None, "spearman": None, "qwk": None}
    if n >= 3 and np.std(h) > 0 and np.std(p) > 0:
        out["pearson"] = round(float(pearsonr(h, p)[0]), 3)
        out["spearman"] = round(float(spearmanr(h, p)[0]), 3)
    hi = np.clip(np.rint(h), 0, max_score).astype(int)
    pi = np.clip(np.rint(p), 0, max_score).astype(int)
    if len(set(hi) | set(pi)) > 1:
        out["qwk"] = round(float(cohen_kappa_score(hi, pi, weights="quadratic",
                                                   labels=list(range(max_score + 1)))), 3)
    return out


def marked_rows(sheet):
    rows = []
    for r in csv.DictReader(open(sheet, encoding="utf-8")):
        if (r.get("status") or "").strip().lower() != "marked":
            continue
        vals = [_num(r.get(c)) for c in ("code_0_8", "explanation_0_8", "documentation_0_4")]
        if any(v is None for v in vals):
            continue
        rows.append(r | {"_h": dict(zip(("code", "explanation", "documentation"), vals))})
    return rows


def evaluate(sheet, pred):
    preds = {r["notebook_id"]: r for r in csv.DictReader(open(pred, encoding="utf-8"))}
    rows = [r for r in marked_rows(sheet) if r["notebook_id"] in preds]
    if not rows:
        return {"status": "no human marks yet", "n_marked": 0,
                "note": "Fill code_0_8, explanation_0_8, documentation_0_4 and set status=marked."}
    res = {"status": "ok", "n_marked": len(rows),
           "n_team_member_author": sum(r.get("team_member_author") == "yes" for r in rows),
           "annotators": sorted({r.get("annotator", "") for r in rows}), "variants": {}}
    for v in VARIANTS:
        res["variants"][v] = {}
        for dim, (_, mx) in DIMS.items():
            if dim == "overall":
                h = [sum(r["_h"].values()) for r in rows]
            else:
                h = [r["_h"][dim] for r in rows]
            p = [_num(preds[r["notebook_id"]].get(f"{v}_{dim}")) for r in rows]
            keep = [(a, b) for a, b in zip(h, p) if b is not None]
            res["variants"][v][dim] = metrics([a for a, _ in keep], [b for _, b in keep], mx)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=ROOT / "data/class/annotations/marking_sheet.csv")
    ap.add_argument("--pred", default=ROOT / "results/class_mu/predictions.csv")
    ap.add_argument("--out", default=ROOT / "results/class_mu/eval.json")
    a = ap.parse_args(argv)
    res = evaluate(a.sheet, a.pred)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
