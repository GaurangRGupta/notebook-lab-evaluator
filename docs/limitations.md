# Limits of the current evaluator

These are properties of the data and of the frozen models. This cleanup does not treat them as defects to remove by refitting. The concept threshold stays 0.275. The split seed stays 42. The synthetic salt stays `SYNTHETIC-DEV-SALT-NOT-SECRET`. `nli.enabled` stays false. The v2 notebooks were not regenerated.

## What the marks are

The v2 marks are rule labels on notebooks the team planted. The annotator id is `RULE`. They are not faculty marks. `data/class/` is the empty place for class notebooks. It has no submissions yet.

## Code score

Code features read syntax, execution counts, and saved output. The code label on a v2 notebook is built from those same signals. A high correlation on that dimension is partly the feature and the label describing one planted fact.

## Explanation text

17 of 20 test explanations appear verbatim in the training notebooks. On every v2 notebook the markdown token count equals the explanation token count, so those two features do not carry separate information.

## Semantic similarity

On the robustness sentences, the frozen MiniLM cosine is 0.424 for a correct paraphrase and 0.861 for a wrong answer that reuses the reference words. A high cosine here means the words are close to the reference. It does not mean the explanation is right. The threshold 0.275 was set on the authored calibration pairs and was not moved after these numbers were seen.

## Week 3 against Week 2

On the v2 test set, explanation MAE is 1.346 for `week3_combined` and 1.512 for `week2_tfidf`. The 95% interval on the difference includes zero. Week 3 is not shown to beat Week 2. Code MAE and documentation MAE are the same for `week2_tfidf`, `week3_semantic`, and `week3_combined`, because those two dimensions use the same features in every variant.

## What the served response does with that fact

Code and documentation confidence are null, because the two variants share those feature lists and their agreement would always be 1. Explanation confidence is `1 - min(1, abs(week2_tfidf - week3_combined) / 8)`. Overall confidence is that explanation value. The method string says this is not a calibrated probability.

Feedback uses the semantic missing-idea list when the encoder is loaded. With the encoder off, it uses the lexical list.

## Files this note does not rewrite

`results/week5/` is the submitted evaluation. Its report, addendum, and `metrics.csv` stay as they were. Per-notebook JSON already in `results/week5/` and the fit-time copies in `results/week6/reports/` still store code as a fraction of 8 and leave feedback and confidence empty. A new call to `evaluate_notebook`, and a future pipeline run, store code as a mark from 0 to 8.

Student code is not executed. Scoring reads the outputs saved in the file.

## Optional Grok grader

`NB_EVAL_LLM=1` adds an `llm` object from `grok-4.20-0309-non-reasoning` (default; `NB_EVAL_LLM_MODEL` overrides it) through the xAI API. Its evaluation is `docs/llm_evaluation.md`. The OpenAI Python package is only the client. The key is `XAI_API_KEY` in a gitignored `.env`. A SuperGrok subscription does not pay for the call. The prompt's only standard is the reference notebook stored for that lab, `reference_solution.ipynb`, plus that card's written explanation and expected output. The model is told not to grade from its own knowledge of the subject. With the switch off, notebook text stays on this machine. The Ridge marks are unchanged either way. The two included reference notebooks were prepared for this project. A faculty notebook replaces the file for that lab.
