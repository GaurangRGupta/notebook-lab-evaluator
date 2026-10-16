"""Collect results/week5/* into one JSON bundle that the Word report generator reads.
No number is typed by hand anywhere in the report: everything comes from this bundle."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "week5"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def recs(df):
    return clean(df.replace({np.nan: None}).to_dict(orient="records"))


def main():
    m = pd.read_csv(R / "metrics.csv")
    idx = ["dataset", "split", "dimension", "model", "model_label", "n"]
    assert not m.duplicated(idx + ["metric"]).any()
    wide = m.set_index(idx + ["metric"])["value"].unstack("metric").reset_index()
    ci = m[m.metric == "MAE"][["dataset", "split", "dimension", "model", "mae_ci95_low", "mae_ci95_high"]]
    wide = wide.merge(ci, on=["dataset", "split", "dimension", "model"], how="left")
    ts = (R / "test_summary.txt").read_text()
    mt = re.search(r"(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) failed)?", ts)
    v2 = ROOT / "results/week5/v2_run"
    frozen = pd.read_csv(ROOT / "results/frozen_week2_baseline/predictions.csv")
    v1p = pd.read_csv(ROOT / "results/week5/v1_run/predictions.csv")
    v1w2 = v1p[v1p.variant == "week2_tfidf"].reset_index(drop=True)
    cal = json.loads((v2 / "concept_calibration.json").read_text())
    rob = pd.read_csv(R / "text_robustness_results.csv")
    bundle = {
        "metrics": recs(wide),
        "paired": recs(pd.read_csv(R / "paired_comparison.csv")),
        "error_cases": recs(pd.read_csv(R / "error_cases.csv")),
        "stress": recs(pd.read_csv(R / "robustness_results.csv")),
        "text_robustness_summary": recs(rob.groupby("variant")[["tfidf_cos", "semantic_cos"]].mean().reset_index()),
        "text_robustness": recs(rob),
        "probes": recs(pd.read_csv(R / "probe_results.csv")),
        "root_cause_kind": recs(pd.read_csv(R / "root_cause_by_explanation_kind.csv")),
        "overfitting": recs(pd.read_csv(R / "overfitting_check.csv")),
        "audit": json.loads((R / "dataset_audit.json").read_text()),
        "config": json.loads((R / "experiment_config.json").read_text()),
        "design_check": {"rows": int(len(pd.read_csv(R / "design_check.csv"))),
                         "inconsistent": int((~pd.read_csv(R / "design_check.csv").consistent).sum())},
        "tests": {"passed": int(mt.group(1)), "skipped": int(mt.group(2) or 0), "failed": int(mt.group(3) or 0),
                  "skipped_lines": [l for l in ts.splitlines() if l.startswith("SKIPPED")]},
        "week2_frozen_reproduced": bool(np.allclose(frozen.predicted, v1w2.predicted)),
        "concept_calibration": {"threshold": cal["threshold"], "eval": cal["eval"], "n_tune": cal["n_tune"], "n_eval": cal["n_eval"]},
        "iaa_v1": json.loads((ROOT / "results/week5/v1_run/iaa.json").read_text()),
        "skipped_invalid": recs(pd.read_csv(R / "_invalid_input_check/skipped_notebooks.csv")) if (R / "_invalid_input_check/skipped_notebooks.csv").exists() else [],
    }
    # did the rounding fix change any Week 2 (v1) agreement metric?
    old = json.loads((ROOT / "results/frozen_week2_baseline/metrics.json").read_text())["results"]
    new = json.loads((ROOT / "results/week5/v1_run/metrics.json").read_text())["variants"]["week2_tfidf"]["holdout"]
    bundle["rounding_fix_changes"] = [{"dimension": d, "split": sp, "metric": k, "before": old[d][sp]["ridge"][k], "after": new[d][sp]["model"][k]}
                                      for d in ("code", "explanation", "documentation") for sp in ("val", "test")
                                      for k in ("within_1", "exact") if old[d][sp]["ridge"][k] != new[d][sp]["model"][k]]
    (R / "report_bundle.json").write_text(json.dumps(clean(bundle), indent=1))
    print("bundle written:", R / "report_bundle.json")


if __name__ == "__main__":
    main()
