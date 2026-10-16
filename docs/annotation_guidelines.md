# Annotation Guidelines v0.1 (DRAFT, needs team approval)

**Status:** draft written for Week 2 development. The maximum marks (code 8, explanation 8,
documentation 4) come from the example input in the Week 1 proposal. The level descriptors
below are **proposed**, not taken from the Week 1 proposal or the assignment handout, which
do not define them. The team must review, edit and approve them before real annotation starts.

## 1. Unit of annotation

One **notebook** receives three marks. Annotators read the whole notebook, including outputs.
Instructor template text (task prompts, given code) is never credited.

## 2. Scales and anchors

Anchors are given for even marks; odd marks sit between neighbouring anchors.

### Code Correctness (0–8)

| Mark | Descriptor |
|---|---|
| 8 | Every task produces the correct output with the intended method; no errors; cells run in order. |
| 6 | All tasks essentially correct, with one minor issue (e.g. a cell not executed, a non-intended method that still gives the right output, cosmetic output differences). |
| 4 | About half the tasks correct; at least one task has a runtime/syntax error or a wrong output. |
| 2 | Most tasks fail (errors or wrong outputs), but there is a relevant attempt. |
| 0 | No code, nothing relevant, or nothing runs. |

Rules: judge the code, not the explanation. Failures caused only by the environment
(e.g. a data file the student could not have had) are not penalised if the code is otherwise correct.

### Explanation Quality (0–8)

| Mark | Descriptor |
|---|---|
| 8 | All explanation prompts answered; conceptually correct; covers the key concepts in the student's own words. |
| 6 | Correct, but misses one key concept or is shallow in one task. |
| 4 | Partly correct; generic in places; several concepts missing or one explanation missing. |
| 2 | Mostly vague, generic statements with little conceptual content, or contains a clear conceptual error. |
| 0 | Missing, placeholder text, or fundamentally wrong (describes a different algorithm). |

Rules: length is not quality. A fluent but wrong explanation scores below a short correct one.
Do not credit text copied from the prompt.

### Documentation Quality (0–4)

| Mark | Descriptor |
|---|---|
| 4 | Meaningful comments where logic is non-obvious, docstrings for functions, clear markdown structure. |
| 3 | Good comments or docstrings with minor gaps. |
| 2 | Some comments, but sparse or partly trivial. |
| 1 | Minimal (one or two trivial comments). |
| 0 | No documentation. |

## 3. Procedure

1. **Calibration:** all three annotators mark the same ~10 notebooks together, discuss, and revise this document (v0.2).
2. **Independent marking:** each annotator marks every notebook in the annotation set alone,
   without seeing other annotators' marks or any automated scores.
3. **Recording:** one row per annotator per notebook in `annotations/raw_annotations.csv`:
   `raw_path, annotator, code, explanation, documentation, status`.
4. **Disagreements:** the Week 1 proposal says disagreements are resolved by discussion.
   Proposed (team to confirm): items where the annotators' range exceeds 1 mark on any dimension
   are discussed and the agreed mark is written to `annotations/adjudicated.csv`; other items take the median.
5. **Agreement reporting** (on pre-discussion marks, `src/evaluation/iaa.py`): Krippendorff's alpha
   (ordinal and interval), ICC(2,1), mean pairwise quadratic-weighted kappa, mean pairwise MAE and
   within ±1 agreement. The mean pairwise MAE is an estimate of human annotation variability on this
   rubric; it is a reference point for the model's MAE, not a hard performance ceiling.

## 4. Privacy

Annotators only see anonymised notebooks (`interim/`). The mapping from hashes to original files
stays offline with one team member. No notebook is pasted into any external AI service.
