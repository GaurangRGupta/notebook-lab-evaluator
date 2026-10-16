"""Build notebooks/week4_report.ipynb (17-section Week 4 deliverable: baseline vs improved).

  python scripts/build_week4_report.py            # write the notebook (not executed)
  python scripts/build_week4_report.py --execute  # write AND execute (needs both pretrained models)

All numbers are computed live. Refuses to run with stub scorers unless NB_EVAL_ALLOW_STUB=1
(used only to smoke-test the notebook code; outputs are then NOT model results).
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
M, C = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

cells = [
M("""# Week 4: Improved Model - Baseline vs Improved Comparison
**Group 3 - Jupyter Notebook Lab Evaluator** · Team J013, J015, J016, J017, J018

> ⚠️ **Synthetic development results.** 10 team-authored notebooks with simulated labels (validation 2, test 2,
> grouped CV on train+validation 8). These numbers test the pipeline and the experimental protocol. They are
> **not** evidence of performance on real student submissions, and differences between models on this data are
> not statistically meaningful."""),
C("""import json, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from src.pipeline import Pipeline
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 80)
stub = os.environ.get("NB_EVAL_ALLOW_STUB") == "1"
overrides = {"semantic": {"encoder": "stub"}, "nli": {"encoder": "stub"}} if stub else None
run = Pipeline(ROOT / "configs" / "synthetic_dev_week4.yaml", overrides=overrides).run()
OUT = run.out
met = json.loads((OUT / "metrics.json").read_text())
L = met["variant_labels"]
if stub: print("!!! STUB SCORERS: notebook smoke test only. Numbers are NOT model results. !!!")
print("WARNING:", met["WARNING"])"""),

M("""## 1. Week 4 Objective
Official requirement (Section 14): *implement and compare an improved approach*; deliverable: **baseline vs improved
model comparison**. Question: does the improved NLP approach evaluate notebooks better than the Week 2 baseline,
using the same data, labels, split, preprocessing and metrics?"""),

M("""## 2. Previous Pipeline
* **Week 2 Baseline** (`week2_tfidf`): anonymise -> nbformat parse -> task alignment -> template removal ->
  code track (AST, errors, outputs, constructs) + text track (TF-IDF cosine, keyword coverage) -> 15 features ->
  Ridge per dimension.
* **Week 3 Semantic Model** (`week3_combined`): Week 2 + frozen MiniLM sentence embeddings (semantic cosine,
  semantic concept coverage) in the explanation model. Already a lexical + semantic hybrid."""),

M("""## 3. Baseline Definition (frozen)
The baseline is the Week 2 model, unchanged. Its predictions were saved to `results/frozen_week2_baseline/`
**before** any Week 4 model was evaluated; a test re-runs Week 2 and requires identical predictions."""),
C("""frozen = pd.read_csv(ROOT / "results/frozen_week2_baseline/predictions.csv")
cur = pd.read_csv(OUT / "predictions.csv"); cur = cur[cur.variant == "week2_tfidf"].reset_index(drop=True)
print("Week 2 baseline predictions identical to frozen snapshot:", np.allclose(frozen.predicted, cur.predicted))"""),

M("""## 4. Limitations of the Baseline (and of Week 3)
* TF-IDF and keyword coverage are lexical: correct paraphrases are missed and wrong answers reusing the reference
  vocabulary are rewarded (Week 3 probes: correct paraphrases 0.02-0.09 TF-IDF cosine, wrong-but-similar 0.27-0.84).
* Week 3's bi-encoder embeds student and reference **separately**; a single vector per text measures topical
  closeness, not whether the statement is **supported or contradicted** (negation, inverted relationships).
* No overall-score evaluation existed before Week 4."""),

M("""## 5. Proposed Improvement
**Week 4 Improved Model** (`week4_improved`) = Week 3 features **+ NLI consistency features** from a frozen,
locally-run cross-encoder (`cross-encoder/nli-deberta-v3-xsmall`):

| Feature | Definition | Question it answers |
|---|---|---|
| `nli_support` | mean P(entailment \\| premise = reference, hypothesis = student sentence) | Is what the student says backed by the reference? |
| `nli_coverage` | mean P(entailment \\| premise = student text, hypothesis = reference sentence) | Does the student's text imply the reference content? |
| `nli_contradiction` | max P(contradiction \\| reference, student sentence) | Does any statement conflict with the reference? |

Scoring estimator: Ridge by default; a shallow random forest only if it clears a rule fixed in advance (section 9)."""),

M("""## 6. Why This Improvement Was Chosen
* Week 3 is already a hybrid, so "TF-IDF + embeddings" would be a renaming, not an improvement.
* The largest weakness measured so far is **similarity is not correctness**. A cross-encoder reads both texts
  **jointly**, so the comparison can attend to negation and to which quantity goes up or down; NLI training
  (SNLI/MultiNLI) is about exactly entailment vs contradiction.
* It is free, local, CPU-sized, explainable ("the reference supports / contradicts this sentence") and leaves the
  code track untouched. The Week 1 proposal's other options (CodeBERT, LLM judge) are heavier and less explainable.
* It is a hypothesis, tested below - not assumed to help."""),

M("""## 7. Architecture
```
.ipynb -> anonymise -> parse -> tasks -> template removal
   CODE TRACK (unchanged)                 TEXT TRACK
   AST, errors, outputs, constructs       TF-IDF cosine                [W2]
                                          MiniLM bi-encoder cosine     [W3]
                                          lexical+semantic concepts    [W3]
                                          NLI cross-encoder:           [W4]
                                            support / coverage / contradiction
   -> features -> estimator per dimension (Ridge, or forest if pre-registered rule passes)
   -> code /8, explanation /8, documentation /4, overall /20
   -> report (+ "possible conceptual error" flags from NLI)
```"""),

M("""## 8. Implementation
New: `src/models/nli_consistency.py`, `src/models/improved.py`, `src/evaluation/error_analysis.py`,
`src/evaluation/robustness.py`, `configs/synthetic_dev_week4.yaml`, `data/synthetic/week4/robustness_cases.json`,
`tests/test_week4.py`, `scripts/run_week4.py`, this report builder. Modified: `src/pipeline.py`
(NLI step, behaviour checks, estimator selection, overall-score metrics, error analysis), `src/models/features.py`
(Week 4 variants + labels). `src/models/baseline.py` is untouched."""),
C("""pd.Series(met["nli_model"]).to_frame("NLI model")"""),

M("""## 9. Experimental Setup
* Same anonymisation, parsing, cleaning, split (StratifiedGroupKFold by student; 6/2/2 notebooks), labels, metrics.
* Pretrained encoders are frozen feature extractors (no fine-tuning). TF-IDF, imputation and estimators fitted on train.
* Ridge alpha chosen on validation (as in Week 2). Estimator rule, fixed before results: use a random forest
  (300 trees, depth 3, fixed) for a dimension only if its leave-one-student-out CV MAE on **train+validation** is
  >= 10% lower than Ridge's. Test never used for any choice; scored once after freezing.
* Two views: hold-out (val n=2, test n=2) and grouped CV on train+validation (n=8)."""),
C("""pd.DataFrame(met["estimator_selection"]).T"""),

M("""## 10. Baseline Results (Week 2, frozen) and 11. Improved Model Results
One table, all models, all dimensions plus overall. *Synthetic development results.*"""),
C("""ORDER = ["week2_tfidf", "week3_combined", "week4_improved"]
def rows(view):
    out = []
    for v in ORDER:
        r = met["variants"][v]
        blocks = {**{d: (r["grouped_cv_train_val"][d] if view == "cv" else r["holdout"][d].get(view)) for d in ("code", "explanation", "documentation")},
                  "overall": r["grouped_cv_overall"] if view == "cv" else r["holdout_overall"].get(view)}
        for d, b in blocks.items():
            if not b: continue
            for k, name in (("mean_predictor", "Mean predictor"), ("model", L[v])):
                if k == "mean_predictor" and v != ORDER[0]: continue
                x = b[k]
                out.append({"Model": name, "Dimension": d, "n": x["n"], "MAE": x["MAE"], "RMSE": x["RMSE"],
                            "Pearson": x["pearson_r"], "Spearman": x["spearman_rho"], "±1": x["within_1"], "Exact": x["exact"]})
    return pd.DataFrame(out).round(3)
print("Grouped CV on train+validation (n=8)"); display(rows("cv"))
print("Hold-out validation (n=2)"); display(rows("val"))
print("Hold-out TEST (n=2), scored once after freezing"); display(rows("test"))"""),

M("""## 12. Comparison (measured differences, Week 4 minus Week 2)
Negative MAE/RMSE difference = Week 4 lower error; positive = higher error. Reported as measured, no claim of
significance."""),
C("""def diff(view):
    out = []
    for d in ("code", "explanation", "documentation", "overall"):
        get = (lambda v: met["variants"][v]["grouped_cv_train_val"][d]["model"]) if (view == "cv" and d != "overall") else \\
              (lambda v: met["variants"][v]["grouped_cv_overall"]["model"]) if view == "cv" else \\
              (lambda v: (met["variants"][v]["holdout_overall"] if d == "overall" else met["variants"][v]["holdout"][d])[view]["model"])
        a, b = get("week2_tfidf"), get("week4_improved")
        out.append({"view": view, "dimension": d, "MAE W2": a["MAE"], "MAE W4": b["MAE"], "Δ MAE": b["MAE"] - a["MAE"],
                    "RMSE W2": a["RMSE"], "RMSE W4": b["RMSE"], "Δ RMSE": b["RMSE"] - a["RMSE"],
                    "±1 W2": a["within_1"], "±1 W4": b["within_1"]})
    return pd.DataFrame(out)
D = pd.concat([diff("cv"), diff("val"), diff("test")]).round(3); display(D)
for _, r in D.iterrows():
    word = "lower" if r["Δ MAE"] < 0 else "higher" if r["Δ MAE"] > 0 else "identical"
    print(f"{r['view']:4s} {r['dimension']:13s}: Week 4 MAE {word} by {abs(r['Δ MAE']):.3f} marks")"""),

M("""## 13. Error Analysis
Per notebook and dimension, baseline vs improved; "succeeds" = rounded prediction within ±1 mark. CV rows use
out-of-fold predictions on train+validation; val/test rows use the frozen hold-out predictions."""),
C("""ea = pd.read_csv(OUT / "error_analysis.csv")
display(ea.groupby(["eval", "category"]).size().unstack(fill_value=0))
for cat in ["baseline fails, improved succeeds", "baseline succeeds, improved fails", "both fail"]:
    sub = ea[ea.category == cat]
    print(f"\\n=== {cat}: {len(sub)} rows ===")
    if len(sub): display(sub.drop(columns=["notebook_id"]).head(12))"""),
M("""Reading guide: for explanation rows, compare `tfidf_cos_mean` (lexical), `sem_cos_mean` (semantic) and
`nli_contradiction`/`nli_support` against the human mark. Code rows depend on the code track only, which is
identical in both models unless the estimator changed, so code differences come from the estimator, not from NLP."""),
C("""pr = pd.read_csv(OUT / "probe_results.csv")
pr[["id", "category", "student_is_correct", "tfidf_cos", "semantic_cos", "nli_support", "nli_contradiction"]]"""),

M("## 14. Robustness and Ablation"),
C("""rb = pd.read_csv(OUT / "robustness_summary.csv", index_col=0); rb"""),
C("""abl = []
for v in ["week2_tfidf", "week3_semantic", "week3_combined", "abl_nli_only", "week4_improved", "abl_week4_plus_code"]:
    x = met["variants"][v]["grouped_cv_train_val"]["explanation"]["model"]
    abl.append({"variant": L[v], "explanation features": len(met["variants"][v]["holdout"]["explanation"]["features"]),
                "CV MAE": x["MAE"], "CV RMSE": x["RMSE"], "Spearman": x["spearman_rho"], "±1": x["within_1"]})
print("Explanation dimension, grouped CV on train+validation (n=8). Too small for conclusions about components.")
pd.DataFrame(abl).round(3)"""),

M("""## 15. Synthetic Data Limitation
* 10 notebooks / 5 students; validation and test have 2 notebooks each; CV has 8. A difference of a fraction of a
  mark can come from one notebook. Leave-one-student-out also makes the mean predictor anti-correlated by
  construction (leaving out a high scorer lowers the training mean), so its correlations are not meaningful.
* Synthetic explanations are clear-cut; probe and robustness sets are small and team-authored.
* The same code runs on real data with `configs/class_data.yaml`; only that run can answer the Week 4 question."""),

M("## 16. Conclusion"),
C("""print("Measured on synthetic development data (see section 12 for every difference).")
cv = D[D.view == "cv"].set_index("dimension")["Δ MAE"]
for d, x in cv.items(): print(f"  CV {d:13s} Δ MAE (W4 - W2) = {x:+.3f}")
print("No claim of improvement on real student submissions is made.")"""),

M("""## 17. Week 5 Plan
1. Evaluation & error analysis on the **real** annotated notebooks (test scored once), with bootstrap CIs.
2. 5-10 documented failure cases from real data (assignment requirement), using the error-analysis tooling.
3. Robustness on real explanations (paraphrase, truncation, irrelevant text) and NLI flag precision checked by annotators.
4. Calibrate the contradiction flag threshold on validation data; decide whether it belongs in feedback (Week 6)."""),
]

nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "week4_report.ipynb"
if "--execute" in sys.argv:
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
nbformat.write(nb, out)
print("wrote", out)
