# Datasheet (assignment Section 8 items)

## A. Current development data: SYNTHETIC

| Item | Synthetic development set |
|---|---|
| 1. Source | Authored by the team (`scripts/make_synthetic_fixtures.py`). Not student work. |
| 2. Size | 10 notebooks (2 labs x 5 quality profiles), 130 cell records. |
| 3. Format | nbformat v4 `.ipynb`, executed with a real Python kernel so outputs, errors and execution counts are genuine. |
| 4. Labels | SIMULATED marks from 3 fictitious annotators (code 0–8, explanation 0–8, documentation 0–4). |
| 5. Split | Notebook-level, grouped by synthetic student key; target ~60/20/20; realised proportions in `results/synthetic_dev/split.json`. |
| 6. Preprocessing | See README "Pipeline". |
| 7. Quality issues | Tiny; profiles are deliberately clear-cut; labels are simulated; both labs have the same structure. |
| 8. Privacy | Contains only fictitious names/roll numbers (J098, J099) planted to test the anonymiser. |
| 9. Licensing | Team-authored. |

**Use:** testing the pipeline only. Metrics on this set are not evidence of evaluator performance.

## B. Real dataset: TO BE COMPLETED when available

| Item | To record |
|---|---|
| 1. Source | Class lab submissions (MS Teams), with faculty authorisation documented; public notebooks with URL per file. |
| 2. Size | Notebooks, cells, per assignment and per source. |
| 3. Format | `.ipynb` v4 (convert older versions with nbformat). |
| 4. Labels | Annotator marks + adjudicated marks; agreement statistics. |
| 5. Split | Realised proportions and group counts. |
| 6. Preprocessing | As implemented. |
| 7. Quality issues | Unexecuted notebooks, environment errors, copied submissions, missing references. |
| 8. Privacy | Anonymisation steps, residual-PII check, where the id map and salt are stored. |
| 9. Licensing | **Verified per source**: record the licence stated on each GitHub repository or Kaggle notebook page. Do not assume a platform-wide licence; exclude a notebook if its licence cannot be confirmed. |
