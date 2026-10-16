# Group 3: Jupyter Notebook Lab Evaluator

The repository root is the notebook lab evaluator. Retired notebooks, the hello package, faculty handouts, and the Foundations of Machine Learning folder are in `local_archive/`. That folder is gitignored. The record of the move and the code corrections is `docs/cleanup_log.md`. Limits of the data and the frozen models are in `docs/limitations.md`.

## Week 6
The fitted models are served by a Python function and a local HTTP route. Documentation: `docs/API.md`. One-page note: `results/week6/Group3_Week6_Asset.docx`.

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows; use .venv/bin/pip on other systems
python scripts/fetch_models.py                   # verified ONNX export, skipped if already checked
python scripts/fit_week6_asset.py                # writes results/week6 and does not rewrite results/week5
python -m src.api.app                            # http://127.0.0.1:8000  POST /evaluate-notebook
python -m pytest -q tests/test_week6_asset.py
```

Week 6 scoring uses the ONNX encoder. It does not call an external language model. Student notebooks are not executed.

## Week 5
Protocol: `docs/week5_protocol.md` (written before any v2 model run). Report: `results/week5/Group3_Week5_Evaluation_Report.docx`. Addendum: `results/week5/Group3_Week5_Addendum.docx`.
```bash
python scripts/fetch_models.py        # verified ONNX export of all-MiniLM-L6-v2 (offline-capable)
python scripts/run_week5.py           # v1 + v2 evaluation, error cases, stress tests -> results/week5/
python scripts/export_week5_bundle.py && node scripts/build_week5_report.js
```
All data are synthetic (v1: simulated annotators; v2: rule-derived labels). The Week 4 NLI model is not evaluated
(pretrained cross-encoder unavailable in the development environment).


NLP Course Project 2026, AI University Platform. Team: J013, J015, J016, J017, J018.

> **Current status: development build on SYNTHETIC data.** The class notebooks, reference
> solutions/explanations and faculty-approved dataset are not yet available. Everything under
> `data/synthetic/` is team-authored fixture data with SIMULATED labels. It exists to build and
> test the pipeline. No result in `results/synthetic_dev/` is evidence of how the evaluator
> performs on real student submissions.

## Week 4: baseline vs improved model

* **Week 2 Baseline** (`week2_tfidf`), frozen in `results/frozen_week2_baseline/` (a test enforces it).
* **Week 3 Semantic Model** (`week3_combined`).
* **Week 4 Improved Model** (`week4_improved`): adds NLI consistency features (support, coverage, contradiction)
  from a frozen local cross-encoder, plus overall-score metrics, ablations, robustness suite and error analysis.
  Details: `docs/week4_model_card.md`.

```bash
python scripts/run_week4.py      # downloads two models once, runs tests, pipeline and the Week 4 report
```
Synthetic development results: not evidence of real student performance.

## Week 3 (Version 1): semantic similarity of explanations

New core NLP component: `src/models/semantic_similarity.py`, a frozen, locally-run
`sentence-transformers/all-MiniLM-L6-v2` encoder (384-d) that scores each task's student explanation against the
reference explanation and extends concept coverage with semantic matching. It feeds the existing Ridge
explanation model in three comparable variants (`week2_tfidf`, `week3_semantic`, `week3_combined` = Version 1).
Code and documentation scoring are unchanged. Details: `docs/week3_model_card.md`.

```bash
python scripts/run_week3.py      # downloads the model once, runs tests, pipeline and the Week 3 report
```
Outputs: `results/synthetic_dev_week3/` and `notebooks/week3_report.ipynb`.
Synthetic development run: results are not evidence of real student performance.

Week 2 is preserved: `configs/synthetic_dev.yaml` reproduces the Week 2 numbers exactly (unit-tested).

## Run (Week 2)

```bash
pip install -r requirements.txt
python -m ipykernel install --user               # needed only to regenerate fixtures
python scripts/make_synthetic_fixtures.py        # (re)build SYNTHETIC notebooks, executed for real outputs
python -m src.pipeline configs/synthetic_dev.yaml
python -m pytest -q tests
python scripts/build_week2_report.py             # executes notebooks/week2_report.ipynb
```

## Switching to real class data (no code changes)

1. Create `data/class/` with the same layout as `data/synthetic/`:
   `raw/<LAB>/*.ipynb`, `references/<LAB>/{template.ipynb, reference_solution.ipynb,
   reference_explanations.json, expected_outputs.json, rubric.json}`, `manifest.csv`,
   `annotations/raw_annotations.csv`, `class_list.txt`.
2. In `manifest.csv`, `group_key` must be a salted hash of the student (or repo/author for public
   notebooks), so one student's labs stay in the same split. Record `license` per file.
3. `export NB_EVAL_SALT=...` (kept outside the repo), then `python -m src.pipeline configs/class_data.yaml`.
4. Notebooks under `data/class/raw/`, hashed copies under `data/class/interim/`, and `id_map.csv` are already listed in `.gitignore`. The empty `class_list.txt` template is tracked. Do not commit a copy that contains student names.

## Pipeline

```
raw .ipynb ─► anonymise ─► parse (nbformat) ─► align cells to tasks ─► flag template cells
                                                     │
               ┌─────────────────────────────────────┴───────────────────────────────┐
          CODE TRACK                                                              TEXT TRACK
  syntax check (ast, magics stripped)                         markdown → text (keep tf-idf, word2vec …)
  runtime/env errors from outputs                             placeholder removal, template-line removal
  executed? in order?                                         concept coverage vs reference concepts
  output vs expected (numeric/contains/exact)                 TF-IDF cosine vs reference explanation (fit on TRAIN only)
  required-construct coverage vs reference AST                comment density, docstring rate, markdown length
               └──────────────────────► notebook feature row ◄───────────────────────┘
                                             │
                     grouped notebook-level split (~60/20/20)
                                             │
             Ridge regression per rubric dimension (+ mean-predictor floor)
                                             │
                MAE, RMSE, Pearson/Spearman, within ±1, exact  +  per-notebook JSON report
```

| Module | Role |
|---|---|
| `src/preprocessing/anonymize.py` | roll numbers, e-mails, class-list names, user paths, metadata |
| `src/preprocessing/parse_notebook.py` | cell records (source, execution count, outputs, errors, images flag) |
| `src/preprocessing/code_checks.py` | magics, `ast` syntax check, tokenizer-based comments, AST summary, output comparison |
| `src/preprocessing/text_clean.py` | markdown cleaning, tokenisation, concept coverage |
| `src/preprocessing/align_tasks.py` | `## Task N` alignment, template-cell detection |
| `src/data/references.py` | loads template, reference solution, explanations, expected outputs, rubric |
| `src/data/split.py` | near-duplicate grouping, grouped stratified split, leakage check |
| `src/models/features.py` | notebook features; TF-IDF similarity fit on train only |
| `src/models/baseline.py` | Ridge per dimension, alpha chosen on validation; mean predictor |
| `src/evaluation/metrics.py`, `iaa.py` | regression/binary metrics, bootstrap CI; Krippendorff, ICC, QWK |
| `src/models/semantic_similarity.py` | Week 3: frozen sentence encoder, cosine, semantic features, semantic concept matching |
| `src/evaluation/semantic_eval.py` | Week 3: concept-threshold calibration, error-analysis probes |
| `src/models/nli_consistency.py` | Week 4: frozen NLI cross-encoder, support/coverage/contradiction features |
| `src/models/improved.py` | Week 4: estimator factory, pre-registered Ridge vs forest rule |
| `src/evaluation/error_analysis.py`, `robustness.py` | Week 4: baseline-vs-improved error categories, controlled variations |
| `src/pipeline.py` | config-driven orchestration, feature variants, grouped CV, overall score, writes `results/<run>/` |
| `src/reporting.py` | one mark scale, one feedback list, and confidence. Used by the pipeline and by `src/asset.py` |
| `src/asset.py`, `src/api/app.py` | score one uploaded notebook; `POST /evaluate-notebook` on `127.0.0.1` |

## Plan corrections applied (after Week 2 planning review)

- Public-notebook licensing is **verified per source** (the licence on each repository or notebook page), never assumed per platform.
- Inter-annotator MAE is described as an **estimate of human annotation variability**, not a hard performance ceiling.
- The split is **notebook-level and group-preserving, targeting approximately 60/20/20**. StratifiedGroupKFold does not guarantee exact proportions; realised proportions are reported in `split.json`.

## Out of scope so far

CodeBERT, KeyBERT, and LLM scoring stay out of this package.
Week 6 fills feedback from the reference ideas already extracted. When the encoder is loaded, that list is the semantic one. Confidence is agreement between `week2_tfidf` and `week3_combined` only on dimensions whose feature lists differ. Code and documentation confidence are null for that reason. The route is `POST /evaluate-notebook` on localhost. See `docs/API.md` and `docs/limitations.md`. The Week 4 NLI check is still not part of the served model.
