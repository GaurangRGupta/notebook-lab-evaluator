# Class data layout

This folder has the same shape as `data/synthetic/`. It has no notebooks yet. Foundations of Machine Learning files stay in `local_archive/sample_FOML_documents/`. The inventory of that folder is `docs/foml_pilot_inventory.md`.

`configs/class_data.yaml` scores three variants into `results/real_class/`:

- `week2_tfidf` (frozen Week 2 baseline)
- `week3_combined` (Week 3 Version 1)
- `week4_improved` (Week 4 NLI model; this is the primary variant for per-notebook reports)

The error-analysis file compares `week2_tfidf` with `week4_improved`, because that report takes two names. Ridge stays the estimator. The synthetic config's forest rule is left off here.

Run `python -m src.pipeline configs/class_data.yaml` only after the steps below are done. An empty folder will not produce a result. Set `NB_EVAL_SALT` outside the repo first. The pipeline stops if that variable is missing.

## When the class files are in hand

1. Put the original notebooks in `raw/<LAB_ID>/*.ipynb`. Notebook files under `raw/` are gitignored. This note is tracked.
2. For each lab, add `references/<LAB_ID>/` with these files, written from the official question sheet:
   - `template.ipynb`
   - `reference_solution.ipynb`
   - `reference_explanations.json`
   - `expected_outputs.json`
   - `rubric.json`
   The reference explanation is the assignment. A student answer is not the reference.
3. Add one row per notebook to `manifest.csv`. Columns are `raw_path`, `assignment_id`, `group_key`, `source`, `license`. `group_key` is a salted hash of the student, so one student's labs stay in one split.
4. Human marks go in `annotations/raw_annotations.csv`. Columns are `raw_path`, `annotator`, `code`, `explanation`, `documentation`, `status`. Two people mark. Nobody marks their own notebook. Agree `docs/annotation_guidelines.md` first. It is still a draft.
5. If two marks on one notebook differ by more than 1, discuss and write the agreed mark in `annotations/adjudicated.csv`. Columns are `raw_path`, `code`, `explanation`, `documentation`. Agreement is computed on the marks from before that discussion.
6. `class_list.txt` in this folder is an empty template and is tracked. One name per line. Lines that start with `#` are ignored. Do not commit a copy that contains student names.
7. The anonymiser writes hashed notebooks into `interim/`. Those notebooks and `id_map.csv` are gitignored. Markers read the hashed copies. This note is tracked.
8. Before the run, copy `data/synthetic/week3/` to `data/class/week3/`. Those files are authored calibration pairs for the concept threshold. They are method checks, not class submissions.

`results/real_class/` stays empty until that run. Synthetic metrics stay in their own results folders.
