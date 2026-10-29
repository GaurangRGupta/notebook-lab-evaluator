"""Score the kept MU class notebooks with the frozen Week 6 asset.

IMPORTANT: the models were trained on synthetic labs only and no human marks exist for the class
data. Every mark written here is an out-of-domain, UNVALIDATED prediction. Nothing is sent to any
external service (the optional LLM grader is disabled for class labs in src/asset.py).

    python scripts/run_class.py            # after scripts/build_class_manifest.py

Reads data/class/manifest.csv and data/class/raw/ (gitignored). Writes:
  data/class/interim/<id>.ipynb          anonymised copies (gitignored)
  results/class_mu/predictions.csv       one row per notebook, TF-IDF and MiniLM variants
  results/class_mu/alignment.csv         how cells were aligned to tasks; parse failures
  results/class_mu/summary.json          descriptive statistics per lab and variant
  results/class_mu/reports/<id>.json     full report per notebook (anonymised ids)
  data/class/annotations/marking_sheet.csv   blank sheet for the team / faculty
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asset import evaluate_notebook, load_asset  # noqa: E402
from src.preprocessing.anonymize import Scrubber, load_class_list  # noqa: E402

CLASS = ROOT / "data" / "class"
OUT = ROOT / "results" / "class_mu"
CLASS_ROLL = r"(?<![A-Za-z0-9])[Jj]\s?\d{3}(?!\d)"
VARIANTS = {"week2_tfidf": "tfidf", "week3_combined": "minilm"}


def describe(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "mean": round(statistics.mean(xs), 3),
            "sd": round(statistics.stdev(xs), 3) if len(xs) > 1 else None,
            "min": round(min(xs), 3), "median": round(statistics.median(xs), 3), "max": round(max(xs), 3)}


def main():
    rows = [r for r in csv.DictReader((CLASS / "manifest.csv").open(encoding="utf-8")) if r["decision"] == "keep"]
    names = load_class_list(CLASS / "class_list.local.txt")
    if not names:
        print("WARNING: data/class/class_list.local.txt is empty; names in cells will not be scrubbed")
    scrub = Scrubber(CLASS_ROLL, names)
    asset = load_asset()
    (CLASS / "interim").mkdir(parents=True, exist_ok=True)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    preds, align, failures = [], [], []
    for r in rows:
        nid, lab = r["notebook_id"], r["assignment_id"]
        src = CLASS / r["raw_path"]
        if not src.is_file():
            failures.append({"notebook_id": nid, "stage": "read", "error": "raw file missing"})
            continue
        try:
            nb = nbformat.read(str(src), as_version=4)
            anon = scrub.scrub_notebook(nb)
            dst = CLASS / "interim" / f"{nid}.ipynb"
            nbformat.write(anon, str(dst))
            rep = evaluate_notebook(dst.read_bytes(), lab, asset)
        except Exception as e:  # report, never hide
            failures.append({"notebook_id": nid, "stage": "evaluate", "error": f"{type(e).__name__}: {e}"})
            traceback.print_exc()
            continue
        (OUT / "reports" / f"{nid}.json").write_text(json.dumps(rep, indent=2))
        row = {"notebook_id": nid, "assignment_id": lab, "group_key": r["group_key"],
               "team_member_author": r["team_member_author"], "manifest_flags": r["flags"]}
        for v, short in VARIANTS.items():
            m = rep["variant_marks"][v]
            for d in ("code", "explanation", "documentation", "overall"):
                row[f"{short}_{d}"] = m[d]
        row.update({"output_match_rate": rep["output_match_rate"], "n_errors": len(rep["errors"]),
                    "flag_output_ok_explanation_weak": rep["flag_output_ok_explanation_weak"],
                    "confidence": rep["confidence"] if not isinstance(rep["confidence"], dict)
                    else rep["confidence"].get("level", json.dumps(rep["confidence"])),
                    "n_missing_concepts": sum(len(v) for v in rep["missing_concepts"].values())
                    if isinstance(rep["missing_concepts"], dict) else len(rep["missing_concepts"])})
        preds.append(row)
        a = rep["alignment"]
        align.append({"notebook_id": nid, "assignment_id": lab, **{k: a.get(k, 0) for k in
                      ("method", "marker", "q_header", "output", "similarity", "inherited", "nearest", "unaligned")}})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("predictions.csv", preds), ("alignment.csv", align)):
        if data:
            with (OUT / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    with (OUT / "failures.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["notebook_id", "stage", "error"]); w.writeheader(); w.writerows(failures)

    summary = {"status": "out-of-domain, UNVALIDATED predictions. No human marks exist for these notebooks.",
               "models": "frozen Week 6 asset trained on synthetic labs (results/week6/asset)",
               "n_kept": len(rows), "n_scored": len(preds), "n_failed": len(failures), "per_lab": {}}
    by_lab = defaultdict(list)
    for p in preds:
        by_lab[p["assignment_id"]].append(p)
    for lab, ps in sorted(by_lab.items()):
        e = {"n": len(ps)}
        for short in VARIANTS.values():
            e[short] = {d: describe([p[f"{short}_{d}"] for p in ps]) for d in ("code", "explanation", "documentation", "overall")}
        e["output_match_rate"] = describe([p["output_match_rate"] for p in ps])
        e["n_flag_output_ok_explanation_weak"] = sum(bool(p["flag_output_ok_explanation_weak"]) for p in ps)
        al = [x for x in align if x["assignment_id"] == lab]
        tot = sum(x["marker"] + x["q_header"] + x["output"] + x["similarity"] + x["inherited"] + x["nearest"] + x["unaligned"] for x in al)
        e["cells_nearest_task"] = sum(x["nearest"] for x in al)
        e["cells_unaligned"] = sum(x["unaligned"] for x in al)
        e["cells_total"] = tot
        summary["per_lab"][lab] = e
    allp = preds
    summary["all"] = {short: {d: describe([p[f"{short}_{d}"] for p in allp]) for d in ("code", "explanation", "documentation", "overall")}
                      for short in VARIANTS.values()}
    tf = [p["tfidf_overall"] for p in allp]; mi = [p["minilm_overall"] for p in allp]
    summary["tfidf_vs_minilm_overall_mean_abs_diff"] = round(statistics.mean(abs(a - b) for a, b in zip(tf, mi)), 3) if allp else None
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    sheet = CLASS / "annotations" / "marking_sheet.csv"
    with sheet.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["notebook_id", "assignment_id", "team_member_author", "annotator", "code_0_8",
                    "explanation_0_8", "documentation_0_4", "status", "justification"])
        for r in rows:
            w.writerow([r["notebook_id"], r["assignment_id"], r["team_member_author"], "", "", "", "", "unmarked", ""])
    print(f"scored {len(preds)}/{len(rows)}; failures {len(failures)} -> {OUT}")


if __name__ == "__main__":
    main()
