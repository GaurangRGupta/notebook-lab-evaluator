"""Build and execute notebooks/week2_report.ipynb (16-section Week 2 deliverable).
All numbers in the report are computed live by the pipeline when the notebook runs."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
M, C = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

cells = [
M("""# Week 2: Dataset & Baseline: Group 3 Jupyter Notebook Lab Evaluator
**Team:** J013, J015, J016, J017, J018 · NLP Course Project 2026

> ⚠️ **SYNTHETIC DEVELOPMENT RUN.** The class notebooks, reference solutions and faculty-approved dataset
> are not yet available. This report runs the complete Week 2 pipeline on **team-authored synthetic fixtures
> with simulated annotations**. It demonstrates that the infrastructure works end to end. **None of the
> numbers below are evidence of how the evaluator performs on real student submissions.**"""),
C("""import json, sys, warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from src.pipeline import Pipeline
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
run = Pipeline(ROOT / "configs" / "synthetic_dev.yaml").run()
OUT = run.out
print("\\ndata_status =", run.status)"""),

M("""## 1. Project Overview
Programming labs are submitted as `.ipynb` files mixing code, markdown explanations, comments and outputs.
Auto-graders usually check only execution or output. Week 1 defined a component that evaluates **both** code
correctness and the natural-language content, returning a structured report (code correctness, output match,
explanation score, documentation quality, overall score, errors, feedback, confidence).

**Week 2 scope (assignment Section 14):** data collection, cleaning, preprocessing, annotation, split, baseline,
initial evaluation. Improved models (embeddings, CodeBERT, KeyBERT, LLM) are Weeks 3–4."""),

M("""## 2. Dataset Description
| | Planned (Week 1) | Used in this run |
|---|---|---|
| Source | Anonymised class notebooks (MS Teams) + licensed GitHub/Kaggle notebooks | **Synthetic fixtures** authored by the team |
| Size | ~150–200 notebooks, ~3,000–4,000 cells | 10 notebooks, 2 labs × 5 quality profiles |
| Labels | ~100 notebooks × 3 annotators | **Simulated** marks from 3 fictitious annotators |

Synthetic labs: `LAB_SYN1` (TF-IDF & cosine similarity) and `LAB_SYN2` (bigram LM with add-one smoothing).
Notebooks were executed with a real kernel, so outputs, tracebacks and execution counts are genuine.
Each lab has a template, reference solution, reference explanations with concept lists, expected outputs
(derived by executing the reference) and a rubric. Public-notebook licences will be verified per source."""),
C("""man = pd.read_csv(ROOT / "data/synthetic/manifest.csv")
man[["raw_path", "assignment_id", "group_key", "profile"]]"""),

M("## 3. Dataset Statistics"),
C("""st = json.loads((OUT / "dataset_stats.json").read_text())
{k: v for k, v in st.items() if k != "label_distribution"}"""),
C("""pd.DataFrame(st["label_distribution"]).fillna(0).astype(int).sort_index().rename_axis("mark")"""),

M("""## 4. Notebook Parsing
`nbformat.read(as_version=4)` + `validate`. Each cell becomes a record with: type, source, execution count,
output types, text output (stream + `text/plain`, ANSI stripped), error name/value/traceback, an image flag
(bytes dropped), line/char counts. Cells are aligned to tasks via `## Task N` / `Q N` headers; unaligned
cells are left unassigned rather than guessed."""),
C("""cells = pd.read_csv(OUT / "cells.csv")
ex = cells[cells.notebook_id == cells.notebook_id.iloc[26]]
ex[["cell_index", "cell_type", "task_id", "is_template", "execution_count", "has_error", "error_name", "output_text"]]"""),

M("""## 5. Data Cleaning
* **Anonymisation first** (no external APIs): roll-number regex, e-mails, class-list names, user names in paths,
  all notebook/cell metadata. Files renamed to salted hashes. Two synthetic notebooks contain planted fake PII.
* **Template removal:** cells matching the handout (similarity ≥ 0.9) are flagged and excluded from student
  features, so prompt text cannot inflate similarity to the reference.
* **Placeholders** (`-`, "Your answer here", empty) are treated as no answer."""),
C("""print("Residual PII pattern matches after anonymisation:", run.pii_residual)
raw = (ROOT / "data/synthetic/raw/LAB_SYN1/SYN_S05_LAB_SYN1.ipynb")
nid = run.manifest.set_index("raw_path").loc["raw/LAB_SYN1/SYN_S05_LAB_SYN1.ipynb", "notebook_id"]
import nbformat
print("BEFORE:", nbformat.read(raw, 4).cells[1].source.replace(chr(10), " "))
print("AFTER: ", nbformat.read(run.interim / f"{nid}.ipynb", 4).cells[1].source.replace(chr(10), " "))"""),

M("""## 6. Preprocessing
**Code:** IPython magics/shell lines commented out → `ast.parse` (syntax check) → AST summary (calls, defs,
docstrings, loops, imports) → comments via `tokenize` (so `#` inside strings is ignored) → error type from
outputs, split into *environment* vs *code* errors → output normalisation (whitespace, case, memory addresses)
and comparison with expected outputs (numeric tolerance / contains / exact).

**Markdown:** images, links, HTML, header/emphasis marks removed; inline code and LaTeX content kept;
lowercased; token pattern keeps `tf-idf`, `word2vec`, `bi-lstm`; **no stemming or lemmatisation** so
technical terms survive; concept matching on token sequences with listed variants."""),
C("""from src.preprocessing.text_clean import markdown_to_text, tokenize
from src.preprocessing import code_checks as cc
s = "## Why?\\n**TF-IDF** down-weights `the`; see [docs](http://x) $P(w|c)$ and bi-LSTM"
print(markdown_to_text(s)); print(tokenize(markdown_to_text(s)))
print(cc.syntax_check("%time x = 1\\nfor i in range(3)\\n    print(i)"))
print(cc.extract_comments('s = "a # b"  # real comment'))"""),

M("""## 7. Annotation Schema
Scales from the Week 1 example rubric: **Code 0–8, Explanation 0–8, Documentation 0–4**, marked per notebook.
Level descriptors are a **draft** (`docs/annotation_guidelines.md`) that the team must approve; Week 1 does not
define them. Procedure: joint calibration on ~10 notebooks → independent blind marking by 3 annotators →
items with annotator range > 1 discussed and adjudicated, others take the median. Agreement is reported on
pre-discussion marks. The mean pairwise annotator MAE estimates **human annotation variability**; it is a
reference point, not a hard ceiling. Below, agreement is computed on **simulated** annotations to test the code."""),
C("""iaa = json.loads((OUT / "iaa.json").read_text())
pd.DataFrame({d: {k: r[k] for k in ["krippendorff_alpha_ordinal", "icc_2_1", "mean_pairwise_qwk",
                                     "mean_pairwise_mae", "mean_pairwise_within_1"]}
              for d, r in iaa["iaa"].items()}).round(3)"""),
C("""lab = pd.read_csv(OUT / "labels_final.csv")
lab[["notebook_id", "code", "explanation", "documentation", "code_needs_discussion",
     "explanation_needs_discussion", "documentation_needs_discussion"]]"""),

M("""## 8. Train/Validation/Test Split
Split at **notebook level**; all cells inherit their notebook's split. Groups are never divided: same student
(`group_key`), plus near-duplicate clusters (TF-IDF cosine ≥ 0.95, for copied/forked notebooks).
StratifiedGroupKFold (5 folds, stratified by assignment/source): 1 fold test, 1 validation, 3 train, targeting
**approximately** 60/20/20. Because whole groups are assigned, exact proportions are not guaranteed; realised
proportions are reported. TF-IDF vocabulary and imputation statistics are fit on train only.

*Why:* cells from one notebook share variable names, phrasing and style, and one student's labs share style.
Splitting cells randomly would let the model see near-copies of test material in training and inflate scores."""),
C("""sp = json.loads((OUT / "split.json").read_text())
print(sp["method"]); print("target  ", sp["target"]); print("realised", sp["realised"], "| groups:", sp["n_groups"])
run.manifest.merge(pd.read_csv(ROOT/"data/synthetic/manifest.csv")[["raw_path"]], on="raw_path")\\
   .groupby(["split", "group_key"]).assignment_id.apply(list)"""),

M("""## 9. Code-Based Baseline
Deterministic checks per notebook: share of student code cells that parse, share executed, error-cell rate,
out-of-order execution, output-match rate vs expected outputs, and required-construct coverage (non-generic
library calls in the reference solution, e.g. `TfidfVectorizer`, `cosine_similarity`; helper names defined by
the reference are excluded because students may name helpers differently)."""),
C("""F = run.df.merge(run.manifest[["notebook_id", "raw_path"]], on="notebook_id")
F["nb"] = F.raw_path.str.extract(r"(S0\\d_LAB_SYN\\d)")[0]
from src.models.features import CODE_FEATURES, EXPLANATION_FEATURES, DOCUMENTATION_FEATURES
F.set_index("nb").sort_index()[CODE_FEATURES + ["code"]].round(2)"""),

M("""## 10. Text-Based Baseline
TF-IDF (unigrams + bigrams, sublinear tf, English stop words, custom technical tokeniser) **fit on training
explanations + reference texts only**; cosine similarity between each task's student explanation and the
reference explanation (mean and max over tasks). Concept coverage = share of reference concepts (with
variants) found in the explanation."""),
C("""F.set_index("nb").sort_index()[EXPLANATION_FEATURES + ["explanation"]].round(2)"""),

M("""## 11. Feature Engineering
Documentation features: comment density (comment lines / code lines), docstring rate (docstrings / defined
functions), markdown length (log tokens), markdown-to-code cell ratio. Feature groups per dimension are set
in `src/models/features.py` and can be overridden from the config."""),
C("""F.set_index("nb").sort_index()[DOCUMENTATION_FEATURES + ["documentation"]].round(2)"""),

M("""## 12. Baseline Model
One **Ridge (linear) regression per rubric dimension** on standardised, median-imputed features;
alpha ∈ {0.1, 1, 10} chosen on validation MAE; predictions clipped to [0, max]. Linear rather than logistic
because Week 1 frames scoring as regression against human marks, the marks are ordinal, and a 9-class
logistic model would be unstable on ~60 training notebooks. A **mean predictor** is the reference floor."""),
C("""met = json.loads((OUT / "metrics.json").read_text())
pd.DataFrame({d: r["coefficients"] for d, r in met["results"].items()}).round(3)"""),

M("""## 13. Evaluation
Per dimension: MAE, RMSE, Pearson and Spearman correlation, agreement within ±1 and exact agreement
(predictions rounded), bootstrap 95% CI for MAE when n ≥ 10. Correlations are reported as undefined when
n < 3 or a series is constant. Precision/recall/F1 are used only where a binary gold label exists (none in this
synthetic run). Concept-extraction F1 belongs with the Week 4 KeyBERT comparison."""),

M("""## 14. Initial Results (SYNTHETIC: pipeline check only)
> ⚠️ Validation and test each contain **2 synthetic notebooks** with **simulated** labels. These numbers only show
> that training, prediction and scoring run correctly. They say nothing about accuracy on real submissions."""),
C("""rows = []
for d, r in met["results"].items():
    for s in ("val", "test"):
        if s in r:
            for m in ("ridge", "mean_predictor"):
                x = r[s][m]
                rows.append({"dimension": d, "split": s, "model": m, "n": x["n"], "MAE": x["MAE"], "RMSE": x["RMSE"],
                             "pearson_r": x["pearson_r"], "within_1": x["within_1"], "exact": x["exact"]})
print(met["WARNING"])
pd.DataFrame(rows).round(2)"""),
C("""rep = json.loads(next((OUT / "reports").glob("*.json")).read_text())
print(json.dumps({k: rep[k] for k in list(rep)[:14]}, indent=1)[:1500])"""),

M("""## 15. Limitations
* **No real data yet.** All results come from 10 synthetic notebooks with simulated labels; they validate code paths only.
* Rubric level descriptors are a draft awaiting team approval.
* Output comparison assumes tasks are marked with `## Task N` headers and that expected outputs exist.
* Keyword coverage is lexical: paraphrases outside the variant lists are missed and a wrong explanation that uses
  the right words can still score (the Week 1 stretch goal about code–explanation inconsistency needs Week 3–4 models).
* TF-IDF similarity rewards vocabulary overlap, not correctness.
* Notebooks are not re-executed by default (stored outputs are used); re-execution needs a sandbox and lab data files.
* With ~100 annotated notebooks, validation/test sets of ~20 give wide confidence intervals."""),

M("""## 16. Week 2 Conclusion
**Done (infrastructure):** anonymiser, nbformat parser, task alignment and template removal, code checks
(AST, errors, output comparison, construct coverage), markdown cleaning and concept coverage, TF-IDF similarity
fit on train only, grouped notebook-level split, Ridge baseline with a mean-predictor floor, metrics and
agreement statistics, per-notebook JSON reports, 29 passing tests, config-driven so real data replaces
synthetic data without code changes.

**Not done (blocked on data):** real data collection, real annotation and agreement, and the real baseline
evaluation. These run with `python -m src.pipeline configs/class_data.yaml` once `data/class/` is populated."""),
]

nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "week2_report.ipynb"
NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
nbformat.write(nb, out)
print("wrote", out)
