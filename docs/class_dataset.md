# Class dataset (Managing Uncertainty): keep/drop and feasibility

Source: the class folder `Class_dataset/MU` (8 student folders plus root-level files), copied read-only. Decisions per file are in `data/class/manifest.csv` (`python scripts/build_class_manifest.py`). Student names, rolls and paths are replaced by salted ids; raw copies, the id map and the name list are gitignored.

## Keep / drop (88 files)

**Kept: 62 notebooks.** Per lab: MU_L01 8, MU_L02 8, MU_L03 7, MU_L04 6, MU_L05 7, MU_L06 8, MU_L07_08 7, MU_L09 7, MU_L10 4.

**Dropped: 26.**

| reason | files |
|---|---|
| root-level copy, byte-identical to a file in a student folder | 17 |
| duplicate of the same student's file (identical code) | 3 |
| data file, not a submission (kept aside as reference data) | 3 |
| Word document, not a notebook | 1 |
| blank handout (no code); used as the MU_L02 template | 1 |
| byte-identical to another student's file (attribution conflict) | 1 |

Flags kept on rows (not dropped): one MU_L03 pair is byte-identical across two students, one MU_L09 pair has identical code, 37 of the 62 kept notebooks are by project team members (5 of the 8 students), and most notebooks have no markdown cells, so the explanation cannot be assessed.

## Feasibility verdict

- **Feasible for**: a working demonstration on real coursework, descriptive statistics and a systems check (62 of 62 scored, 0 failures), error and output-mismatch detection, and as the pool for human marking.
- **Not feasible for**: training or validating the marks. There are no human marks, 4-8 notebooks per lab, almost no written explanations (explanation is 0 for all 62), no faculty reference solutions (references are team-authored), and a conflict of interest because most authors are on the team.
- **So**: report class results as out-of-domain, unvalidated predictions (`docs/evaluation.md`). Agreement numbers only after independent markers fill `data/class/annotations/marking_sheet.csv` and `scripts/eval_class.py` is run.
