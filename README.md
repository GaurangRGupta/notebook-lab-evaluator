# Group 3: Jupyter Notebook Lab Evaluator

Score one Jupyter notebook by sending the `.ipynb` file and a lab id. The reply is JSON: code, explanation, and documentation marks, overall out of 20, output match, errors, missing ideas, a short feedback sentence, and confidence. The student's code is not executed. The answer card stays in this package. You do not send it with the call.

The usage manual is `docs/manual.md`. The same manual in Word is `results/week6/Group3_Week6_Asset.docx`. Both have the function signatures, the arguments, the return fields, and runnable examples. `docs/API.md` is the short call.

## Call it

`LAB_SYN2` is the language-model lab (add-one smoothing and perplexity). The example file is `examples/language_model_lab.ipynb`. `LAB_SYN1` is the TF-IDF lab.

If `results/week6/asset/` and `models/all-MiniLM-L6-v2-onnx/` are already in the folder, install and start:

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
python -m src.api.app
```

On Windows the call from a second window is:

```bash
curl.exe -s -X POST http://127.0.0.1:8000/evaluate-notebook -F "assignment_id=LAB_SYN2" -F "file=@examples/language_model_lab.ipynb"
```

From Python, in this folder:

```python
from src.asset import evaluate_notebook, load_asset
asset = load_asset()
notebook = open("examples/language_model_lab.ipynb", "rb").read()
report = evaluate_notebook(notebook, "LAB_SYN2", asset)
```

The server listens on `127.0.0.1:8000` only. For that example the live marks are code 7.87, explanation 5.13, documentation 2.14, overall 15.14. The saved output matches, and the write-up for task T3 does not cover "inverse probability" or "surprise".

If the asset folder or the encoder folder is missing, run `python scripts/fetch_models.py` and then `python scripts/fit_week6_asset.py` before starting the server. The asset check is `python -m pytest -q tests/test_week6_asset.py`.

A new lab is a new folder of five files (handout, solution, explanations, expected outputs, rubric) under `data/synthetic_v2/references/<lab id>/`. Restart the server and pass that id. Scoring one notebook does not retrain the model. The two cards in this package were prepared for the project, and the marks follow that rubric.

The repository also keeps earlier notes. Retired material is in `local_archive/`, which is gitignored. `docs/cleanup_log.md` records the cleanup. `docs/limitations.md` records the limits of the fitted models.

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
