# Week 5 evaluation protocol (written BEFORE any model was run on the v2 data)

Status: SYNTHETIC DEVELOPMENT EVALUATION. No real student notebooks or faculty marks are available.

## 1. Why a second synthetic set
The Week 2 set (v1: 10 notebooks, 5 synthetic students) gives test n = 2, which cannot support a
test-set evaluation. v2 is a larger, *pre-specified* synthetic benchmark. It is still synthetic: the
labels are derived by fixed rules from planted properties, so results measure whether the evaluator
recovers planted quality differences, not agreement with real graders.

## 2. v2 design (`scripts/make_synthetic_v2.py`, seed 2026)
* 30 synthetic students x 2 labs (LAB_SYN1 TF-IDF/cosine, LAB_SYN2 bigram LM) = 60 notebooks,
  same templates, reference solutions, reference explanations and expected outputs as v1.
* Student traits, shared across both labs (so a student is a real group for splitting):
  code skill (high/mid/low), explanation skill (high/mid/low), comment level (0/1/2),
  helper-function style (none / function without docstring / function with docstring).
* Per task the planted code variant is one of: correct_a, correct_b (alternative correct
  implementation), syntax error, runtime error, wrong logic, unexecuted (correct code, never run).
  Tasks are self-contained so one failure does not cascade.
* Per explanation prompt the planted kind is one of: correct (reference-like wording), paraphrase
  (correct, different wording), partial, vague, wrong (reuses reference vocabulary), off-topic, empty.
  Texts come from fixed pools written before any model was run on v2.
* All notebooks are executed with a real kernel; outputs and errors are genuine.

## 3. Label rules (fixed in advance; single "RULE" annotator)
* Code (0-8): task points correct_a/correct_b = 1, unexecuted = 0.5, syntax/runtime/wrong = 0;
  code = round_half_up(8 x points / 3).
* Explanation (0-8): per prompt correct = 4, paraphrase = 4, partial = 2, vague = 1,
  wrong/off-topic/empty = 0; explanation = sum over the 2 prompts.
* Documentation (0-4): comment level (0-2) + 1 if helper function has a docstring
  + 1 if at least one explanation prompt has a non-empty, non-placeholder answer.
* Overall (0-20) = sum of the three.

## 4. Split
Notebook-level StratifiedGroupKFold (k = 5, seed 42), groups = student, strata = assignment:
one fold test, one fold validation, three folds train (target ~60/20/20). Realised counts are
reported. The split is written to `split.json` before models are fitted and is never changed.

## 5. Models compared (same split, labels, preprocessing, metrics)
* Mean predictor (training mean per dimension).
* Week 2 Baseline (`week2_tfidf`): frozen feature set, Ridge per dimension.
* Week 3 Semantic-only (`week3_semantic`, ablation).
* Week 3 Semantic Model (`week3_combined`) = the IMPROVED model evaluated in Week 5.
  Encoder: all-MiniLM-L6-v2 ONNX export, provenance verified against the published
  sentence-transformers README similarities.
* Week 4 NLI model (`week4_improved`): NOT evaluated. Its pretrained cross-encoder could not be
  obtained in this environment (huggingface.co blocked; no verifiable mirror). Declared here,
  before any result, so its absence cannot be a post-hoc choice.

Only choice made with data: Ridge alpha in {0.1, 1, 10} per dimension on VALIDATION MAE (as in Week 2).
Concept threshold: calibrated on the authored tune pairs (Week 3 rule). Test is scored once.

## 6. Metrics
Per dimension and overall: MAE (primary), RMSE, Pearson, Spearman, within +-1, exact
(rounded half up), bootstrap 95% CI for MAE (2,000 resamples, n >= 10). Also leave-one-student-out
CV on train+validation as a stability check (test untouched).

## 7. Error-case selection (fixed rule, no cherry-picking)
From TEST predictions of week2_tfidf vs week3_combined, explanation and code dimensions:
take all rows where either model's absolute error > 1, sorted by the larger error; then add the
largest-gap rows where the baseline is better and where the improved model is better, and two
controls where both errors <= 0.5. If fewer than 10 rows exist, extend with validation rows, then
CV out-of-fold rows, each labelled with its source.

## 8. Robustness / stress tests
Controlled notebooks (empty/short/paraphrased/wrong-but-similar explanations, correct code + wrong
explanation, wrong code + convincing explanation, missing output, runtime error, output mismatch,
unrelated markdown, reordered cells, unusual term formatting, missing package/file, template text
copied as answer, no documentation, invalid notebook file) are scored by the frozen models
trained on v2 train. They are reported separately and never mixed into test metrics.

## Amendment A1 (before any model was run on v2)
The first v2 generation placed the helper function inside the Task 1 cell; when that cell had a
planted syntax error the helper was never defined and later, correctly implemented tasks failed with
NameError (found by the planted-vs-observed consistency check: 1 notebook, 2 tasks). This violated
the "tasks are self-contained" design rule, so the helper now sits in its own Task 1 cell and the set
was regenerated with the same seed. No label rule, split rule or model choice was changed.
