"""Build notebooks/week3_report.ipynb (16-section Week 3 deliverable).

  python scripts/build_week3_report.py            # write the notebook (not executed)
  python scripts/build_week3_report.py --execute  # write AND execute it (needs the real MiniLM model)

Every number in the executed report is computed live. The report refuses to run with the
test-only stub encoder unless NB_EVAL_ALLOW_STUB=1 (used only to smoke-test the notebook code).
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
M, C = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

cells = [
M("""# Week 3: Core NLP Model (Version 1) - Group 3 Jupyter Notebook Lab Evaluator
**Team:** J013, J015, J016, J017, J018 · NLP Course Project 2026

> ⚠️ **Synthetic development run: results are not evidence of real student performance.**
> The class notebooks, real annotations and faculty-approved dataset are still pending. This report runs the
> Week 3 pipeline on the Week 2 **synthetic fixtures (10 notebooks, simulated labels)** plus small team-authored
> probe and calibration sets. It shows that the semantic component works end to end. Differences between
> TF-IDF and embeddings on 2 validation / 2 test notebooks are **not** statistically meaningful."""),
C("""import json, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from src.pipeline import Pipeline
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 90)
overrides = {"semantic": {"encoder": "stub"}} if os.environ.get("NB_EVAL_ALLOW_STUB") == "1" else None
run = Pipeline(ROOT / "configs" / "synthetic_dev_week3.yaml", overrides=overrides).run()
OUT = run.out
met = json.loads((OUT / "metrics.json").read_text())
if run.sem_info["is_stub"]:
    print("!!! STUB ENCODER: this execution only smoke-tests the notebook code. Numbers are NOT semantic results. !!!")
print("\\nWARNING:", met["WARNING"])"""),

M("""## 1. Project Overview
The Jupyter Notebook Lab Evaluator takes a lab submission (`.ipynb`) and returns a structured evaluation:
code correctness, output match, explanation score, documentation quality, overall score, errors, feedback and
confidence (Week 1). Week 1 framed it as a composite NLP task: (a) cell segmentation, (b) code summarisation,
(c) semantic textual similarity, (d) concept extraction, (e) rubric scoring as regression, (f) code-explanation
consistency. **Week 3 delivers (c)** and uses it in (d) and (e)."""),

M("""## 2. Week 2 Baseline Recap
Week 2 built: anonymisation, nbformat parsing, task alignment, template removal, AST/syntax/error checks, output
comparison, required-construct coverage, markdown cleaning, lexical concept coverage, **TF-IDF cosine** (fit on
train only), grouped notebook-level split, one Ridge model per rubric dimension, a mean-predictor floor and the
metric suite. The `week2_tfidf` variant below reproduces the Week 2 run exactly (checked by a unit test)."""),
C("""def holdout_table(variants=("week2_tfidf",), dims=("code", "explanation", "documentation")):
    rows = []
    for v in variants:
        for d in dims:
            for s in ("val", "test"):
                r = met["variants"][v]["holdout"][d].get(s)
                if not r: continue
                for m in ("ridge", "mean_predictor"):
                    x = r[m]
                    rows.append({"variant": v if m == "ridge" else "mean predictor", "dimension": d, "split": s,
                                 "n": x["n"], "MAE": x["MAE"], "RMSE": x["RMSE"], "pearson": x["pearson_r"],
                                 "spearman": x["spearman_rho"], "within_1": x["within_1"], "exact": x["exact"]})
    return pd.DataFrame(rows).drop_duplicates().round(3)
holdout_table()"""),

M("""## 3. Week 3 Objective
Replace *lexical* matching of student explanations with a *semantic* representation, and plug it into the
existing rubric-scoring model:

`student explanation → sentence encoder → embedding`, `reference explanation → sentence encoder → embedding`,
`cosine(embeddings) → features → Ridge → explanation score`.

Out of scope this week (Week 4+): CodeBERT, KeyBERT, LLM/rubric-conditioned scoring, generated feedback,
confidence, REST API."""),

M("""## 4. Why TF-IDF Is Insufficient
TF-IDF represents a text by weighted counts of the exact tokens it contains. Two explanations score as
similar only if they share words (or bigrams). A correct paraphrase with different wording therefore gets a
low score, while a wrong explanation that reuses the reference vocabulary can get a high one. The table shows
TF-IDF cosine for the Week 3 brief's Viterbi example (probe P01) and its greedy-decoding counterpart (P02)."""),
C("""probes = pd.read_csv(OUT / "probe_results.csv")
probes[probes.id.isin(["P01", "P02"])][["id", "category", "student_is_correct", "tfidf_cos", "student"]]"""),

M("""## 5. Semantic Embedding Model
| Property | Value |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Architecture | 6-layer MiniLM transformer (distilled from a larger BERT-style model), mean pooling, L2 normalisation |
| Size | ~22M parameters, ~90 MB; CPU inference on a laptop |
| Output | 384-dimensional unit vector per text |
| Input window | 256 word-piece tokens; longer texts are split into sentence windows and averaged |
| Training | contrastive learning on >1B sentence pairs, so paraphrases map to nearby vectors |
| Licence | Apache-2.0 |
| Use here | **frozen, pretrained feature extractor**: no fine-tuning, `eval()` mode, `inference_mode()`, gradients disabled |
| Privacy | runs locally; no student text is sent to any API (Week 1 requirement) |

Why this model: it is the standard lightweight general-purpose sentence encoder, small enough for every team
member's laptop and for batch grading on CPU. A larger encoder (e.g. `all-mpnet-base-v2`, 768-d, ~110M params)
can be swapped in via `semantic.model_name`; that comparison is a Week 4 option, not a Week 3 requirement."""),
C("""info = run.sem_info
enc = run.sem.encoder
n_params = sum(p.numel() for p in enc.model.parameters()) if hasattr(enc, "model") else None
pd.Series({**info, "parameters": n_params})"""),

M("""## 6. Embedding Generation
Text → WordPiece tokens (with `[CLS]`/`[SEP]`) → 6 transformer layers → token vectors → **mean pooling** over
non-padding tokens → **L2 normalisation**. Empty or placeholder explanations are not encoded: they get a zero
vector and similarity 0 (the same policy as TF-IDF in Week 2). Embeddings are cached by text."""),
C("""demo = ["TF-IDF down-weights words that appear in most documents.",
        "Words that occur everywhere get a small weight.",
        "The cat sat on the mat."]
E = enc.encode(demo)
print("shape:", E.shape, "| L2 norms:", np.round(np.linalg.norm(E, axis=1), 4))
pd.DataFrame(E[:, :8], index=demo, columns=[f"d{i}" for i in range(8)]).round(3)"""),

M("""## 7. Semantic Similarity
For unit vectors, cosine similarity is the dot product: `cos(u, v) = u·v / (‖u‖‖v‖) = u·v`, in [-1, 1].
Per notebook and per task with a reference explanation:
* `sem_cos_mean`: mean over tasks of cos(student explanation, reference explanation) - overall conceptual closeness;
* `sem_cos_max`: best task - mirrors `tfidf_cos_max` so the two representations are directly comparable;
* `sem_concept_coverage`: share of reference concepts matched **lexically (Week 2 rule) or semantically**
  (best cosine between any phrasing of the concept and any student sentence ≥ a threshold chosen on an
  authored calibration set, section 11).

Not added: a "semantic minus TF-IDF" feature. For a linear model it is an exact linear combination of two
features already present, so it adds collinearity and no information. Per-task similarities are kept in the
per-notebook report (tasks differ between labs, so they are not fixed-width model features)."""),
C("""S = enc.encode(demo)
pd.DataFrame(S @ S.T, index=[d[:40] for d in demo], columns=["s0", "s1", "s2"]).round(3)"""),

M("""## 8. Integration With Existing Features
Semantic features join the existing notebook feature table. Code correctness stays on the deterministic code
track; documentation keeps its Week 2 features. **Only the explanation model changes.** Below: explanation
features per notebook with the (simulated) human mark. Correlations use **train+validation only** (test excluded)
and n = 8 synthetic notebooks, so they are descriptive, not evidence."""),
C("""F = run.df.merge(run.manifest[["notebook_id", "raw_path"]], on="notebook_id")
F["nb"] = F.raw_path.str.extract(r"(S0\\d_LAB_SYN\\d)")[0]
cols = ["tfidf_cos_mean", "tfidf_cos_max", "keyword_coverage", "sem_cos_mean", "sem_cos_max", "sem_concept_coverage"]
display(F.set_index("nb").sort_index()[["split"] + cols + ["explanation"]].round(3))
dev = F[F.split != "test"]
dev[cols].corrwith(dev["explanation"], method="spearman").round(3).rename("spearman_with_human_explanation (train+val, n=8)")"""),

M("""## 9. Model Architecture
Three explanation-model variants, same Ridge estimator, same split, same alpha grid (chosen on validation):
* **A `week2_tfidf`** - Week 2 baseline, unchanged;
* **B `week3_semantic`** - semantic similarity and semantic concept coverage *replace* their lexical counterparts;
* **C `week3_combined`** (Version 1, used for reports) - lexical and semantic evidence together."""),
C("""pd.DataFrame({v: {"explanation features": ", ".join(r["holdout"]["explanation"]["features"]),
                  "alpha (val)": r["holdout"]["explanation"]["alpha"]} for v, r in met["variants"].items()}).T"""),
C("""pd.Series(met["variants"]["week3_combined"]["holdout"]["explanation"]["coefficients"]).round(3).rename("week3_combined explanation coefficients (standardised features)")"""),

M("""## 10. Evaluation
Same framework as Week 2: MAE, RMSE, Pearson, Spearman, within ±1, exact agreement, and the mean predictor.
Leakage controls: the encoder is pretrained and frozen (nothing fitted on our data); TF-IDF, imputation and Ridge
are fitted on train only; alpha on validation; the concept threshold on an authored calibration set; **test is
scored once, after all variants and settings were frozen.** Code and documentation results are identical across
variants by construction (verified by a test), so only explanation is compared."""),
C("""holdout_table(("week2_tfidf", "week3_semantic", "week3_combined"), ("explanation",))"""),
M("""Grouped cross-validation (leave one synthetic student out) on **train+validation only**: 8 out-of-fold
predictions, fixed alpha = 1. More predictions than the 2-notebook validation set, still tiny and synthetic."""),
C("""rows = []
for v, r in met["variants"].items():
    cv = r["grouped_cv_train_val"]["explanation"]
    for m in ("ridge", "mean_predictor"):
        x = cv[m]
        rows.append({"variant": v if m == "ridge" else "mean predictor", "n": x["n"], "MAE": x["MAE"], "RMSE": x["RMSE"],
                     "pearson": x["pearson_r"], "spearman": x["spearman_rho"], "within_1": x["within_1"], "exact": x["exact"]})
pd.DataFrame(rows).drop_duplicates().round(3)"""),

M("""## 11. TF-IDF vs Semantic Embeddings
**(a) Explanation-to-reference similarity.** Probe pairs (section 12) and notebook features (section 8).

**(b) Concept matching.** 41 authored concept/sentence pairs; the threshold is chosen on the *tune* half and
precision/recall/F1 reported on the *eval* half, whose concepts never appear in the tune half."""),
C("""cal = json.loads((OUT / "concept_calibration.json").read_text())
print("threshold:", cal["threshold"], "| chosen on:", cal["chosen_on"], "| n_tune:", cal["n_tune"], "| n_eval:", cal["n_eval"])
pd.DataFrame(cal["eval"]).T[["precision", "recall", "f1", "n", "support_positive"]].round(3)"""),
C("""pd.DataFrame(cal["eval_rows"])"""),
C("""reps = [json.loads(p.read_text()) for p in sorted((OUT / "reports").glob("*.json"))]
pd.DataFrame([{"notebook_id": r["notebook_id"], "split": r["split"],
               "concepts matched only semantically": r["semantic"]["concepts_matched_semantically"],
               "per-task similarity": r["semantic"]["per_task_similarity"]} for r in reps])"""),

M("""## 12. Error Analysis
Authored probe pairs: correct paraphrases, a technically equivalent definition, wrong explanations reusing the
reference vocabulary, inverted relationships, a one-word negation, a vague answer, an unrelated text sharing a
keyword, and a verbatim copy. **High semantic similarity means "talks about the same thing", not "is correct".**"""),
C("""probes[["id", "category", "student_is_correct", "tfidf_cos", "semantic_cos", "note"]]"""),
C("""sep = json.loads((OUT / "probe_summary.json").read_text())["separation"]
print("Share of (correct, incorrect) pairs on the same reference where the CORRECT explanation scores higher:")
for k, v in sep.items(): print(f"  {k:13s} {v['correct_ranked_higher']:.2f}  over {v['pairs']} pairs  (0.5 = no separation)")
print()
p = probes.set_index("id")
def line(a, b, why):
    print(f"{why}: {a} tfidf={p.loc[a,'tfidf_cos']:.2f} sem={p.loc[a,'semantic_cos']:.2f} | "
          f"{b} tfidf={p.loc[b,'tfidf_cos']:.2f} sem={p.loc[b,'semantic_cos']:.2f}")
line("P01", "P02", "Viterbi: correct paraphrase vs greedy (wrong)")
line("P03", "P04", "TF-IDF: correct paraphrase vs inverted relationship")
line("P11", "P12", "Perplexity: equivalent definition vs one-word negation")
line("P09", "P10", "Smoothing: correct paraphrase vs wrong mechanism")
line("P06", "P07", "Cosine: correct paraphrase vs trigonometry")
print()
wins = probes[probes.student_is_correct & (probes.semantic_cos > probes.tfidf_cos)]
print("Correct explanations where semantic similarity exceeds TF-IDF:", ", ".join(wins.id))
risky = probes[~probes.student_is_correct].sort_values("semantic_cos", ascending=False)
print("Incorrect explanations ranked by semantic similarity (highest = most dangerous):")
print(risky[["id", "category", "semantic_cos", "tfidf_cos"]].to_string(index=False))"""),
M("""How to read this: rows where a wrong explanation (P02, P04, P08, P10, P12) keeps a high semantic similarity are
failure cases for *any* similarity-to-reference score, because embeddings encode topic and wording more than
logical relations, negation or direction of an effect. P13 (verbatim copy) scores maximally with both methods,
so similarity cannot distinguish understanding from copying. These cases motivate Week 4 (a model that looks at
relations and at the code, e.g. code-explanation consistency) rather than a larger similarity model."""),

M("""## 13. Synthetic Data Limitation
* 10 synthetic notebooks, 5 synthetic students, simulated labels; validation and test hold **2 notebooks each**.
* Correlations on n = 2 are undefined; the CV comparison has n = 8. No difference between variants here can be
  called an improvement.
* Synthetic explanations were written with clear-cut quality levels; real explanations are longer, noisier and
  mixed-quality.
* Probe and calibration sets are small and authored by the team, so they illustrate behaviour; they are not a
  benchmark.
* All of this is re-run unchanged on real data with `configs/class_data.yaml` once it is available."""),

M("""## 14. Version 1 Pipeline
```
.ipynb
  -> anonymise -> nbformat parser -> cell segmentation, task alignment, template removal
  -> CODE TRACK                              TEXT TRACK
       AST / syntax checks                     markdown cleaning, placeholder removal
       runtime vs environment errors           TF-IDF cosine (fit on train)          [Week 2]
       output vs expected                      MiniLM sentence embeddings (frozen)   [Week 3]
       required-construct coverage             semantic cosine to reference          [Week 3]
                                               lexical + semantic concept coverage   [Week 3]
  -> feature aggregation (code | explanation | documentation)
  -> Ridge rubric model per dimension (variant week3_combined)
  -> structured JSON report (scores, errors, missing concepts, per-task semantic similarity)
```
Reusable component: `src/models/semantic_similarity.py` (`SentenceTransformerEncoder`, `SemanticSimilarity.similarity`,
`.concept_coverage`, `.notebook_features`), independent of the pipeline and ready for the Week 6 API."""),

M("## 15. Week 3 Conclusion"),
C("""ex = {v: met["variants"][v]["grouped_cv_train_val"]["explanation"]["ridge"]["MAE"] for v in met["variants"]}
print("Explanation MAE, grouped CV on train+val (n=8, synthetic):", {k: round(x, 3) for k, x in ex.items()})
print("Encoder:", run.sem_info["model"], "| revision:", run.sem_info["revision"], "| stub:", run.sem_info["is_stub"])
print("Interpretation: pipeline check only. Differences on 8 synthetic notebooks are not evidence that one variant is better.")"""),
M("""**Delivered:** a frozen, locally-run sentence-embedding component; semantic similarity and semantic concept
coverage features; three comparable scoring variants in the existing Ridge framework; leakage controls; a
concept-matching calibration and evaluation; an error-analysis probe set; tests (stub tests everywhere,
real-model tests where the model is available). **Not yet shown:** any real-data result."""),

M("""## 16. Week 4 Plan (Improved Model)
1. Hybrid explanation scorer: keep semantic features; add a **code-explanation consistency** signal (Week 1 task f),
   e.g. CodeBERT/UniXcoder embeddings of code cells vs the explanation, to catch "correct code, wrong explanation".
2. Relation-aware checks for the failure cases above: negation/direction cues, and an optional locally-run or
   rubric-conditioned LLM judge for the explanation component, compared against the embedding scorer
   (no identifiable student text to external APIs).
3. KeyBERT concept extraction compared with the lexical+semantic concept matcher (P/R/F1 on annotated concepts).
4. Confidence estimate (e.g. agreement between scorers / distance to training data).
5. Formal baseline-vs-improved comparison on the **real** annotated data, test scored once."""),
]

nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "week3_report.ipynb"
if "--execute" in sys.argv:
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
nbformat.write(nb, out)
print("wrote", out)
