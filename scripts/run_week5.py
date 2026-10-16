"""Reproduce the complete Week 5 evaluation.

    python scripts/fetch_models.py      # once: MiniLM ONNX export + provenance check
    python scripts/run_week5.py         # writes results/week5/*

Nothing in the report is typed by hand: scripts/build_week5_report.js reads these files.
All data are SYNTHETIC (see docs/week5_protocol.md).
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "week5"
DIMS = ["code", "explanation", "documentation"]
MAX = {"code": 8, "explanation": 8, "documentation": 4, "overall": 20}


def sh(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def metrics_long(run, dataset, evidence):
    from src.models.features import VARIANT_LABELS
    met = json.loads((run.out / "metrics.json").read_text())
    rows = []
    for v, r in met["variants"].items():
        views = {s: {**{d: r["holdout"][d].get(s) for d in DIMS}, "overall": r["holdout_overall"].get(s)} for s in ("val", "test")}
        views["cv_train_val"] = {**{d: r["grouped_cv_train_val"][d] for d in DIMS}, "overall": r["grouped_cv_overall"]}
        for split, block in views.items():
            for d, b in block.items():
                if not b:
                    continue
                for key, model_name, label in (("model", v, VARIANT_LABELS.get(v, v)), ("mean_predictor", "mean_predictor", "Mean predictor")):
                    if key == "mean_predictor" and v != "week2_tfidf":
                        continue  # identical for every variant; report once
                    x = b[key]
                    ci = r["holdout"][d].get(split, {}).get("MAE_95CI") if (key == "model" and d in DIMS and split in ("val", "test")) else None
                    for m in ("MAE", "RMSE", "pearson_r", "spearman_rho", "within_1", "exact"):
                        rows.append({"dataset": dataset, "evidence": evidence, "model": model_name, "model_label": label,
                                     "split": split, "dimension": d, "max_score": MAX[d], "n": x["n"], "metric": m,
                                     "value": x[m], "mae_ci95_low": ci[0] if (ci and m == "MAE") else None,
                                     "mae_ci95_high": ci[1] if (ci and m == "MAE") else None})
    return rows


def predictions_long(run, dataset, evidence):
    pr = pd.read_csv(run.out / "predictions.csv")
    key = run.manifest.set_index("notebook_id")
    pr["dataset"], pr["evidence"] = dataset, evidence
    pr["anon_label"] = pr.notebook_id.map(key["raw_path"].str.extract(r"((?:V2_)?S\d+_LAB_SYN\d)")[0])
    pr["assignment_id"] = pr.notebook_id.map(key["assignment_id"])
    pr["group"] = pr.notebook_id.map(key["group_key"])
    pr["error"] = pr.predicted - pr.human
    pr["abs_error"] = pr.error.abs()
    tot = pr.groupby(["variant", "notebook_id", "split"]).agg(human=("human", "sum"), predicted=("predicted", "sum")).reset_index()
    tot["dimension"] = "overall"
    for c in ("dataset", "evidence"):
        tot[c] = pr[c].iloc[0]
    for c in ("anon_label", "assignment_id", "group"):
        tot[c] = tot.notebook_id.map(pr.drop_duplicates("notebook_id").set_index("notebook_id")[c])
    tot["error"] = tot.predicted - tot.human
    tot["abs_error"] = tot.error.abs()
    return pd.concat([pr, tot], ignore_index=True)


def diagnose(r) -> tuple[str, str, str]:
    """Rule-based error category, root cause and mitigation derived from the PLANTED properties and
    the measured features of this case (so every statement is checkable against the files)."""
    kinds = {r["planted_explanation"].split(", ")[0].split("=")[1], r["planted_explanation"].split(", ")[1].split("=")[1]}
    code = r["planted_code"]
    eb, ei = r["baseline_error"], r["improved_error"]
    if r["dimension"] == "explanation":
        if "wrong" in kinds and max(eb, ei) > 0:
            return ("incorrect but lexically/semantically similar explanation",
                    f"Planted wrong explanation(s) reuse the reference vocabulary; TF-IDF cosine {r['tfidf_cos_mean']:.2f} and "
                    f"semantic cosine {r['sem_cos_mean']:.2f} stay high, and neither representation encodes the inverted relation, so both models over-score.",
                    "Relation-aware check (NLI contradiction vs reference, Week 4 component) or rubric items for key relations; human review when similarity is high but concepts conflict.")
        if "paraphrase" in kinds and min(eb, ei) < 0:
            return ("paraphrase (correct explanation in different words)",
                    f"Correct explanation with low word overlap (TF-IDF {r['tfidf_cos_mean']:.2f}); lexical keyword coverage {r['keyword_coverage']:.2f} misses concepts, "
                    f"semantic cosine {r['sem_cos_mean']:.2f} recovers only part of the gap, so both models under-score.",
                    "Several reference explanations per prompt, semantic concept matching with more concept phrasings, calibrate on real paraphrases.")
        if kinds & {"vague", "partial", "empty"} and max(eb, ei) > 0:
            return ("incomplete or vague explanation over-scored",
                    f"Planted {sorted(kinds)}; generic on-topic text still gets non-trivial similarity (semantic {r['sem_cos_mean']:.2f}) and the length feature carries weight, "
                    "so the score is pulled towards the middle of the scale.",
                    "Concept-level rubric (credit per concept rather than similarity), penalty for missing required concepts.")
        if "correct" in kinds and min(eb, ei) < 0:
            return ("correct explanation under-scored (regression to the mean)",
                    "Linear model trained on 36 notebooks compresses extreme marks towards the training mean.",
                    "More labelled data; per-concept scoring; calibrate predictions on validation.")
        return ("other explanation error", "See features in this row.", "Inspect manually.")
    if "T2=wrong" in code and r["assignment_id"] == "LAB_SYN1":
        return ("required construct missing (wrong method, correct output)",
                f"T2 uses CountVectorizer, which prints the expected matrix shape, so output match stays high (output_match_rate {r['output_match_rate']:.2f}); "
                "only construct coverage signals the problem and its weight is small.",
                "Treat missing required constructs as a task failure (rule), or add per-task construct features.")
    if "unexecuted" in code:
        return ("missing output (unexecuted cell)",
                "The rubric gives half credit for correct but unexecuted code; the model only sees missing output and an output mismatch.",
                "Re-execute notebooks in a sandbox, or a separate feature for unexecuted-but-parseable cells.")
    if max(abs(eb), abs(ei)) <= 0.5:
        return ("control: both models correct", "Code track features (syntax, errors, output match) determine this mark unambiguously.", "None needed.")
    return ("other code error", "See features in this row.", "Inspect manually.")


def select_error_cases(run, design):
    """Rule from docs/week5_protocol.md section 7 (fixed before the run)."""
    from src.evaluation.metrics import round_half_up
    ea = pd.read_csv(run.out / "error_analysis.csv")
    ea = ea[ea.dimension.isin(["explanation", "code"])].copy()
    ea["e_b"], ea["e_i"] = ea.abs_err_baseline, ea.abs_err_improved
    picked, order = [], {"test": 0, "val": 1, "cv_train_val": 2}
    for src in ("test", "val", "cv_train_val"):
        sub = ea[ea["eval"] == src]
        fails = sub[(sub.e_b > 1) | (sub.e_i > 1)].copy()
        fails["k"] = fails[["e_b", "e_i"]].max(axis=1)
        picked += [(r, "failure (error > 1)") for _, r in fails.sort_values("k", ascending=False).iterrows()]
        if len(picked) >= 10:
            break
    test = ea[ea["eval"] == "test"]
    got = {(r.notebook_id, r.dimension, r["eval"]) for r, _ in picked}
    extra = []
    gap = (test.e_i - test.e_b)
    if len(test):
        b_better = test.loc[gap.idxmax()] if gap.max() > 0 else None
        i_better = test.loc[gap.idxmin()] if gap.min() < 0 else None
        for r, why in ((b_better, "largest gap: baseline better"), (i_better, "largest gap: improved better")):
            if r is not None and (r.notebook_id, r.dimension, r["eval"]) not in got:
                extra.append((r, why)); got.add((r.notebook_id, r.dimension, r["eval"]))
        ctrl = test[(test.e_b <= .5) & (test.e_i <= .5)].sort_values("notebook_id").head(2)
        extra += [(r, "control: both succeed") for _, r in ctrl.iterrows()]
    rows = []
    des = design.set_index("raw_path")
    raw = run.manifest.set_index("notebook_id")["raw_path"]
    for i, (r, why) in enumerate(picked + extra, 1):
        d = des.loc[raw[r.notebook_id]]
        rows.append({"case_id": f"E{i:02d}", "selection_reason": why, "source_split": r["eval"],
                     "anon_label": Path(raw[r.notebook_id]).stem, "notebook_id": r.notebook_id, "assignment_id": d.assignment_id,
                     "dimension": r.dimension, "human": r.human,
                     "baseline_week2": r["pred_week2_tfidf"], "improved_week3": r["pred_week3_combined"],
                     "baseline_error": round(r["pred_week2_tfidf"] - r.human, 2),
                     "improved_error": round(r["pred_week3_combined"] - r.human, 2),
                     "category_vs_pm1": r.category,
                     "planted_code": f"T1={d.code_T1}, T2={d.code_T2}, T3={d.code_T3}",
                     "planted_explanation": f"T2={d.expl_T2}, T3={d.expl_T3}",
                     **{c: r.get(c) for c in ("tfidf_cos_mean", "sem_cos_mean", "keyword_coverage", "sem_concept_coverage",
                                              "output_match_rate", "error_cell_rate") if c in r.index},
                     "evidence": "synthetic v2 (rule-derived label)"})
    df = pd.DataFrame(rows)
    diag = df.apply(diagnose, axis=1, result_type="expand")
    df["error_category"], df["root_cause"], df["suggested_mitigation"] = diag[0], diag[1], diag[2]
    return df


def run_stress(run):
    from src.evaluation.stress_tests import CRITERIA, fx, make_cases
    from src.preprocessing.code_checks import classify_error
    work = OUT / "stress_notebooks"
    work.mkdir(parents=True, exist_ok=True)
    cases = make_cases()
    scored = {}
    for name, (desc, nb) in cases.items():
        nb = fx.execute(nb, work)
        p = work / f"{name}.ipynb"
        nbformat.write(nb, p)
        scored[name] = (desc, run.score_notebook(p, "LAB_SYN1"))
    base = scored["BASE"][1]["predictions"]
    rows = []
    for name, (desc, s) in scored.items():
        crit = CRITERIA.get(name)
        for v in ("week2_tfidf", "week3_combined"):
            p = s["predictions"][v]
            f = s["features"]
            rows.append({"test": name, "input_condition": desc, "model": v,
                         **{f"pred_{d}": round(p[d], 2) for d in DIMS + ["overall"]},
                         **{f"base_{d}": round(base[v][d], 2) for d in DIMS},
                         "criterion": crit[0] if crit else "reference",
                         "passed": (bool(crit[1](p, base[v])) if crit else None),
                         "tfidf_cos_mean": round(f.get("tfidf_cos_mean", np.nan), 3),
                         "sem_cos_mean": round(f.get("sem_cos_mean", np.nan), 3),
                         "keyword_coverage": round(f.get("keyword_coverage", np.nan), 3),
                         "output_match_rate": round(f.get("output_match_rate", np.nan), 3),
                         "error_cell_rate": round(f.get("error_cell_rate", np.nan), 3),
                         "env_errors_detected": sum("[env error]" in e for e in s["diagnostics"]["errors"]),
                         "code_errors_detected": sum("[code error]" in e or "SyntaxError" in e for e in s["diagnostics"]["errors"]),
                         "consistency_flag": "not implemented (no code-explanation consistency check exists)",
                         "evidence": "controlled synthetic stress test"})
    # P: invalid notebook must be skipped, not crash the pipeline
    from src.pipeline import Pipeline
    bad = ROOT / "data" / "synthetic_v2" / "raw" / "INVALID_stress.ipynb"
    bad.write_text("{this is not a notebook")
    man = pd.read_csv(ROOT / "data/synthetic_v2/manifest.csv")
    tmp_man = ROOT / "data/synthetic_v2/manifest_with_invalid.csv"
    pd.concat([man, pd.DataFrame([{**man.iloc[0].to_dict(), "raw_path": "raw/INVALID_stress.ipynb"}])]).to_csv(tmp_man, index=False)
    try:
        r2 = Pipeline(ROOT / "configs/week5_v2.yaml", overrides={"manifest": "manifest_with_invalid.csv", "semantic": {"enabled": False},
                                                                  "model": {"variants": ["week2_tfidf"], "primary_variant": "week2_tfidf",
                                                                            "compare": None, "evaluate_test": False},
                                                                  "results_dir": str(OUT / "_invalid_input_check")}).run()
        ok, note = len(r2.skipped) == 1, r2.skipped[0]["error"] if r2.skipped else "not skipped"
    except Exception as e:
        ok, note = False, f"pipeline crashed: {type(e).__name__}: {e}"
    finally:
        bad.unlink(missing_ok=True)
        tmp_man.unlink(missing_ok=True)
    rows.append({"test": "P_invalid_notebook_file", "input_condition": "File that is not valid notebook JSON added to the manifest",
                 "model": "pipeline", "criterion": "skipped with a logged error; run completes", "passed": ok,
                 "consistency_flag": None, "evidence": "controlled synthetic stress test", "note": note})
    return pd.DataFrame(rows)


def paired_comparison(run, base="week2_tfidf", imp="week3_combined", n_boot=5000, seed=0):
    """Paired analysis on the SAME notebooks: MAE difference (improved - baseline) with a paired
    bootstrap 95% CI, per-notebook wins/losses, and an exact sign test on |error|."""
    from scipy.stats import binomtest
    pr = pd.read_csv(run.out / "predictions.csv")
    rows = []
    views = {"test": pr[pr.split == "test"], "val": pr[pr.split == "val"]}
    for view, df in views.items():
        for d in DIMS + ["overall"]:
            if d == "overall":
                g = df.groupby(["variant", "notebook_id"]).agg(human=("human", "sum"), predicted=("predicted", "sum")).reset_index()
            else:
                g = df[df.dimension == d]
            a = g[g.variant == base].set_index("notebook_id")
            b = g[g.variant == imp].set_index("notebook_id").loc[a.index]
            rows.append(_paired_row(view, d, a.human.to_numpy(), a.predicted.to_numpy(), b.predicted.to_numpy(), n_boot, seed, binomtest))
    for d in DIMS + ["overall"]:
        y = run.dev[DIMS].sum(axis=1).to_numpy(float) if d == "overall" else run.dev[d].to_numpy(float)
        pa = sum(run.oof[base][k] for k in DIMS) if d == "overall" else run.oof[base][d]
        pb = sum(run.oof[imp][k] for k in DIMS) if d == "overall" else run.oof[imp][d]
        rows.append(_paired_row("cv_train_val", d, y, np.asarray(pa), np.asarray(pb), n_boot, seed, binomtest))
    return pd.DataFrame(rows)


def _paired_row(view, d, y, pa, pb, n_boot, seed, binomtest):
    ea, eb = np.abs(pa - y), np.abs(pb - y)
    diff = eb.mean() - ea.mean()
    rng = np.random.default_rng(seed)
    boots = [eb[i].mean() - ea[i].mean() for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    wins, losses = int((eb < ea - 1e-9).sum()), int((eb > ea + 1e-9).sum())
    p = binomtest(wins, wins + losses).pvalue if wins + losses else None
    return {"view": view, "dimension": d, "n": int(len(y)), "MAE_baseline": float(ea.mean()), "MAE_improved": float(eb.mean()),
            "delta_MAE_improved_minus_baseline": float(diff),
            "relative_change_pct": float(100 * diff / ea.mean()) if ea.mean() else None,
            "delta_ci95_low": float(np.percentile(boots, 2.5)), "delta_ci95_high": float(np.percentile(boots, 97.5)),
            "improved_closer": wins, "baseline_closer": losses, "ties": int(len(y) - wins - losses),
            "sign_test_p": p}


def root_cause(run, design):
    """Out-of-fold (train+val) signed errors of the explanation models grouped by planted
    explanation kinds; and in-sample (train) vs out-of-fold MAE to check overfitting."""
    des = design.set_index("raw_path")
    raw = run.manifest.set_index("notebook_id")["raw_path"]
    rows = []
    for v in run.oof:
        for nid, y, p in zip(run.oof[v]["notebook_id"], run.dev.explanation, run.oof[v]["explanation"]):
            d = des.loc[raw[nid]]
            for t in ("T2", "T3"):
                rows.append({"variant": v, "notebook_id": nid, "task": t, "planted_kind": d[f"expl_{t}"],
                             "human_explanation": y, "pred": p, "signed_error": p - y})
    df = pd.DataFrame(rows)
    by_kind = df.groupby(["variant", "planted_kind"]).agg(n_task_slots=("signed_error", "size"),
                                                          mean_signed_error=("signed_error", "mean")).reset_index()
    tr = run.df[run.df.split == "train"]
    fit = []
    for v, ms in run.variant_models.items():
        for d in DIMS:
            ins = float(np.mean(np.abs(ms[d].predict(tr) - tr[d].to_numpy(float))))
            cv = json.loads((run.out / "metrics.json").read_text())["variants"][v]["grouped_cv_train_val"][d]["model"]["MAE"]
            fit.append({"variant": v, "dimension": d, "train_in_sample_MAE": ins, "cv_out_of_fold_MAE": cv})
    return by_kind, pd.DataFrame(fit)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from src.models.semantic_similarity import OnnxSentenceEncoder, verify_reference
    from src.pipeline import Pipeline
    from src.evaluation.design_check import check
    prov = verify_reference(OnnxSentenceEncoder(str(ROOT / "models/all-MiniLM-L6-v2-onnx")))
    assert prov["passed"], prov
    design_chk = check(ROOT / "data/synthetic_v2")
    design_chk.to_csv(OUT / "design_check.csv", index=False)
    # official runs: each split/test scored exactly once per dataset
    v1 = Pipeline(ROOT / "configs/week5_v1.yaml").run()
    v2r = Pipeline(ROOT / "configs/week5_v2.yaml").run()
    design = pd.read_csv(ROOT / "data/synthetic_v2/design.csv")
    m = pd.DataFrame(metrics_long(v2r, "synthetic_v2", "synthetic development, rule-derived labels")
                     + metrics_long(v1, "synthetic_v1", "synthetic development, simulated annotators"))
    m.to_csv(OUT / "metrics.csv", index=False)
    pd.concat([predictions_long(v2r, "synthetic_v2", "synthetic"), predictions_long(v1, "synthetic_v1", "synthetic")]).to_csv(OUT / "predictions.csv", index=False)
    select_error_cases(v2r, design).to_csv(OUT / "error_cases.csv", index=False)
    paired_comparison(v2r).to_csv(OUT / "paired_comparison.csv", index=False)
    rc_kind, rc_fit = root_cause(v2r, design)
    rc_kind.to_csv(OUT / "root_cause_by_explanation_kind.csv", index=False)
    rc_fit.to_csv(OUT / "overfitting_check.csv", index=False)
    rob = run_stress(v2r)
    rob.to_csv(OUT / "robustness_results.csv", index=False)
    pd.read_csv(v2r.out / "robustness.csv").to_csv(OUT / "text_robustness_results.csv", index=False)
    pd.read_csv(v2r.out / "probe_results.csv").to_csv(OUT / "probe_results.csv", index=False)
    # dataset audit
    feats = v2r.df
    ex_tr = {t for n in feats[feats.split == "train"].notebook_id for t in v2r.expl[n].values() if t}
    ex_te = [t for n in feats[feats.split == "test"].notebook_id for t in v2r.expl[n].values() if t]
    audit = {
        "v2": {"notebooks": int(len(v2r.manifest)), "groups": int(v2r.manifest.group_key.nunique()),
               "cells": int(len(v2r.cells)), "cells_by_type": v2r.cells.cell_type.value_counts().to_dict(),
               "split_counts": v2r.manifest.split.value_counts().to_dict(),
               "split_groups": v2r.manifest.groupby("split").group_key.nunique().to_dict(),
               "label_distribution": {d: v2r.y[d].value_counts().sort_index().to_dict() for d in DIMS},
               "label_missing": int(v2r.y[DIMS].isna().sum().sum()),
               "duplicate_raw_files": int(pd.Series([sh(ROOT / "data/synthetic_v2" / p) for p in v2r.manifest.raw_path]).duplicated().sum()),
               "near_duplicate_groups_merged": int(v2r.split_info["merged_groups"]),
               "test_explanations_seen_verbatim_in_train": f"{sum(t in ex_tr for t in ex_te)}/{len(ex_te)}",
               "planted_vs_observed_inconsistent": int((~design_chk.consistent).sum()),
               "wrong_logic_but_output_matches": int(design_chk.wrong_but_output_matches.sum()),
               "label_source": "RULE-derived from planted properties (docs/week5_protocol.md)",
               "pii_residual_matches": v2r.pii_residual},
        "v1": {"notebooks": int(len(v1.manifest)), "groups": int(v1.manifest.group_key.nunique()), "cells": int(len(v1.cells)),
               "split_counts": v1.manifest.split.value_counts().to_dict(),
               "label_distribution": {d: v1.y[d].value_counts().sort_index().to_dict() for d in DIMS},
               "label_source": "SIMULATED annotators A1-A3 (authored)"},
        "feature_correlations_train_v2": {
            c: round(float(feats[feats.split == "train"][[c, "explanation"]].corr(method="spearman").iloc[0, 1]), 3)
            for c in ("expl_tokens_log", "tfidf_cos_mean", "sem_cos_mean", "keyword_coverage", "sem_concept_coverage")},
        "md_tokens_equals_expl_tokens_v2": bool(np.allclose(feats.md_tokens_log, feats.expl_tokens_log)),
    }
    (OUT / "dataset_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "tests"], cwd=ROOT, capture_output=True, text=True)
    (OUT / "test_summary.txt").write_text("$ python -m pytest -q -rs tests\n\n" + test.stdout[-6000:] + test.stderr[-2000:])
    cfg = {"datasets": {"v2": "configs/week5_v2.yaml", "v1": "configs/week5_v1.yaml"},
           "encoder": {"name": "sentence-transformers/all-MiniLM-L6-v2 (ONNX export)", "provenance_check": prov,
                       "onnx_sha256_16": sh(ROOT / "models/all-MiniLM-L6-v2-onnx/onnx/model.onnx")},
           "week4_nli": "not evaluated: pretrained cross-encoder unavailable in this environment",
           "split_v2": json.loads((v2r.out / "split.json").read_text()),
           "alpha_selected_v2": {v: {d: r["holdout"][d]["alpha"] for d in DIMS} for v, r in json.loads((v2r.out / "metrics.json").read_text())["variants"].items()},
           "concept_threshold_v2": v2r.sem_info["concept_threshold"],
           "python": platform.python_version(), "git_free_snapshot": True}
    (OUT / "experiment_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print("done:", OUT)


if __name__ == "__main__":
    main()
