# Evaluation

All numbers below are read from files in `results/`. Re-generate them with the scripts named in each section.

## Metric definitions

For each dimension (code 0-8, explanation 0-8, documentation 0-4, overall 0-20), with human (or reference-label) mark *h* and predicted mark *p* over *n* notebooks:

- **MAE**: mean of |p − h|. Lower is better. The 95% CI is a bootstrap over notebooks.
- **RMSE**: square root of the mean of (p − h)². It penalises large misses more than MAE.
- **Pearson r**: linear correlation of p and h. Undefined when either side is constant.
- **Spearman ρ**: correlation of the ranks. It asks whether the model orders notebooks like the marker.
- **within ±1**: share of notebooks with |p − h| ≤ 1 mark.
- **QWK** (quadratic weighted kappa): agreement on integer marks (rounded), corrected for chance, with disagreements weighted by squared distance. 1 = perfect, 0 = chance.

## Synthetic data, Week 5 (validation split)

Source: `results/week5/metrics.csv`. Data: `synthetic_v2`, rule-derived labels. This is development evidence only.

| model | split | dimension | n | MAE (95% CI) | Pearson | within ±1 |
|---|---|---|---|---|---|---|
| week2_tfidf | val | code | 12 | 0.658 (0.280–1.136) | 0.924 | 0.833 |
| mean_predictor | val | code | 12 | 1.963 |  | 0.583 |
| week2_tfidf | val | explanation | 12 | 1.223 (0.765–1.712) | 0.799 | 0.667 |
| mean_predictor | val | explanation | 12 | 2.537 |  | 0.167 |
| week2_tfidf | val | documentation | 12 | 0.516 (0.276–0.778) | 0.778 | 0.917 |
| mean_predictor | val | documentation | 12 | 1.019 |  | 0.667 |
| week2_tfidf | val | overall | 12 | 2.119 | 0.763 | 0.333 |
| mean_predictor | val | overall | 12 | 3.213 |  | 0.333 |
| week2_tfidf | test | code | 12 | 0.252 (0.136–0.383) | 0.989 | 1.000 |
| mean_predictor | test | code | 12 | 1.847 |  | 0.667 |
| week2_tfidf | test | explanation | 12 | 1.512 (0.814–2.321) | 0.776 | 0.583 |
| mean_predictor | test | explanation | 12 | 3.051 |  | 0.167 |
| week2_tfidf | test | documentation | 12 | 0.463 (0.275–0.640) | 0.647 | 1.000 |
| mean_predictor | test | documentation | 12 | 0.657 |  | 1.000 |
| week2_tfidf | test | overall | 12 | 1.666 | 0.818 | 0.417 |
| mean_predictor | test | overall | 12 | 2.750 |  | 0.417 |
| week2_tfidf | cv_train_val | code | 48 | 0.543 | 0.948 | 0.917 |
| mean_predictor | cv_train_val | code | 48 | 2.195 | -0.820 | 0.542 |
| week2_tfidf | cv_train_val | explanation | 48 | 1.305 | 0.744 | 0.625 |
| mean_predictor | cv_train_val | explanation | 48 | 2.196 | -0.942 | 0.229 |
| week2_tfidf | cv_train_val | documentation | 48 | 0.579 | 0.661 | 0.958 |
| mean_predictor | cv_train_val | documentation | 48 | 0.826 | -1.000 | 0.708 |
| week2_tfidf | cv_train_val | overall | 48 | 1.670 | 0.830 | 0.458 |
| mean_predictor | cv_train_val | overall | 48 | 3.355 | -0.913 | 0.271 |
| week3_semantic | val | code | 12 | 0.658 (0.280–1.136) | 0.924 | 0.833 |
| week3_semantic | val | explanation | 12 | 1.424 (0.985–1.875) | 0.769 | 0.500 |
| week3_semantic | val | documentation | 12 | 0.516 (0.276–0.778) | 0.778 | 0.917 |
| week3_semantic | val | overall | 12 | 2.243 | 0.758 | 0.250 |
| week3_semantic | test | code | 12 | 0.252 (0.136–0.383) | 0.989 | 1.000 |
| week3_semantic | test | explanation | 12 | 1.618 (1.046–2.262) | 0.803 | 0.333 |
| week3_semantic | test | documentation | 12 | 0.463 (0.275–0.640) | 0.647 | 1.000 |
| week3_semantic | test | overall | 12 | 1.798 | 0.817 | 0.417 |
| week3_semantic | cv_train_val | code | 48 | 0.543 | 0.948 | 0.917 |
| week3_semantic | cv_train_val | explanation | 48 | 1.377 | 0.712 | 0.583 |
| week3_semantic | cv_train_val | documentation | 48 | 0.579 | 0.661 | 0.958 |
| week3_semantic | cv_train_val | overall | 48 | 1.863 | 0.803 | 0.500 |
| week3_combined | val | code | 12 | 0.658 (0.280–1.136) | 0.924 | 0.833 |
| week3_combined | val | explanation | 12 | 1.217 (0.814–1.620) | 0.804 | 0.583 |
| week3_combined | val | documentation | 12 | 0.516 (0.276–0.778) | 0.778 | 0.917 |
| week3_combined | val | overall | 12 | 2.114 | 0.776 | 0.250 |
| week3_combined | test | code | 12 | 0.252 (0.136–0.383) | 0.989 | 1.000 |
| week3_combined | test | explanation | 12 | 1.346 (0.738–2.043) | 0.848 | 0.583 |
| week3_combined | test | documentation | 12 | 0.463 (0.275–0.640) | 0.647 | 1.000 |
| week3_combined | test | overall | 12 | 1.477 | 0.852 | 0.667 |
| week3_combined | cv_train_val | code | 48 | 0.543 | 0.948 | 0.917 |
| week3_combined | cv_train_val | explanation | 48 | 1.245 | 0.797 | 0.562 |
| week3_combined | cv_train_val | documentation | 48 | 0.579 | 0.661 | 0.958 |
| week3_combined | cv_train_val | overall | 48 | 1.626 | 0.852 | 0.458 |
| week2_tfidf | val | code | 2 | 0.931 |  | 1.000 |
| mean_predictor | val | code | 2 | 2.500 |  | 0.000 |
| week2_tfidf | val | explanation | 2 | 1.292 |  | 0.500 |
| mean_predictor | val | explanation | 2 | 2.500 |  | 0.000 |
| week2_tfidf | val | documentation | 2 | 0.103 |  | 1.000 |
| mean_predictor | val | documentation | 2 | 0.000 |  | 1.000 |
| week2_tfidf | val | overall | 2 | 2.314 |  | 0.500 |
| mean_predictor | val | overall | 2 | 0.500 |  | 1.000 |
| week2_tfidf | test | code | 2 | 0.759 |  | 1.000 |
| mean_predictor | test | code | 2 | 2.000 |  | 0.500 |
| week2_tfidf | test | explanation | 2 | 0.957 |  | 1.000 |
| mean_predictor | test | explanation | 2 | 1.000 |  | 1.000 |
| week2_tfidf | test | documentation | 2 | 0.893 |  | 1.000 |
| mean_predictor | test | documentation | 2 | 1.500 |  | 0.500 |
| week2_tfidf | test | overall | 2 | 1.555 |  | 0.500 |
| mean_predictor | test | overall | 2 | 2.000 |  | 0.500 |
| week2_tfidf | cv_train_val | code | 8 | 1.886 | 0.643 | 0.500 |
| mean_predictor | cv_train_val | code | 8 | 3.167 | -0.957 | 0.125 |
| week2_tfidf | cv_train_val | explanation | 8 | 1.210 | 0.890 | 0.625 |
| mean_predictor | cv_train_val | explanation | 8 | 4.167 | -0.997 | 0.000 |
| week2_tfidf | cv_train_val | documentation | 8 | 0.542 | 0.829 | 0.750 |
| mean_predictor | cv_train_val | documentation | 8 | 1.333 | -1.000 | 0.500 |
| week2_tfidf | cv_train_val | overall | 8 | 3.435 | 0.726 | 0.250 |
| mean_predictor | cv_train_val | overall | 8 | 6.458 | -0.991 | 0.250 |
| week3_semantic | val | code | 2 | 0.931 |  | 1.000 |
| week3_semantic | val | explanation | 2 | 1.714 |  | 0.500 |
| week3_semantic | val | documentation | 2 | 0.103 |  | 1.000 |
| week3_semantic | val | overall | 2 | 2.736 |  | 0.000 |
| week3_semantic | test | code | 2 | 0.759 |  | 1.000 |
| week3_semantic | test | explanation | 2 | 1.212 |  | 1.000 |
| week3_semantic | test | documentation | 2 | 0.893 |  | 1.000 |
| week3_semantic | test | overall | 2 | 1.676 |  | 0.500 |
| week3_semantic | cv_train_val | code | 8 | 1.886 | 0.643 | 0.500 |
| week3_semantic | cv_train_val | explanation | 8 | 2.868 | 0.227 | 0.250 |
| week3_semantic | cv_train_val | documentation | 8 | 0.542 | 0.829 | 0.750 |
| week3_semantic | cv_train_val | overall | 8 | 5.195 | 0.295 | 0.250 |
| week3_combined | val | code | 2 | 0.931 |  | 1.000 |
| week3_combined | val | explanation | 2 | 1.411 |  | 0.500 |
| week3_combined | val | documentation | 2 | 0.103 |  | 1.000 |
| week3_combined | val | overall | 2 | 2.434 |  | 0.000 |
| week3_combined | test | code | 2 | 0.759 |  | 1.000 |
| week3_combined | test | explanation | 2 | 0.140 |  | 1.000 |
| week3_combined | test | documentation | 2 | 0.893 |  | 1.000 |
| week3_combined | test | overall | 2 | 1.456 |  | 0.500 |
| week3_combined | cv_train_val | code | 8 | 1.886 | 0.643 | 0.500 |
| week3_combined | cv_train_val | explanation | 8 | 1.506 | 0.848 | 0.625 |
| week3_combined | cv_train_val | documentation | 8 | 0.542 | 0.829 | 0.750 |
| week3_combined | cv_train_val | overall | 8 | 3.731 | 0.687 | 0.125 |

## Synthetic data, held-out test split and Week 8 LLM grader

Source: `results/week8_llm/metrics.csv`. `week3_combined` is the served variant. `week8_llm` is the optional xAI grader, run on synthetic notebooks only. Total API cost of all cached calls: $0.2664 (`results/week8_llm/usage.json`).

| view | variant | dimension | n | MAE (95% CI) | RMSE | Pearson | Spearman | within ±1 |
|---|---|---|---|---|---|---|---|---|
| test | week2_tfidf | code | 12 | 0.252 (0.136–0.383) | 0.335 | 0.989 | 0.929 | 1.000 |
| test | week2_tfidf | explanation | 12 | 1.512 (0.814–2.321) | 2.022 | 0.776 | 0.797 | 0.583 |
| test | week2_tfidf | documentation | 12 | 0.463 (0.275–0.640) | 0.568 | 0.647 | 0.586 | 1.000 |
| test | week2_tfidf | overall | 12 | 1.666 (0.989–2.436) | 2.125 | 0.818 | 0.655 | 0.417 |
| test | week3_combined | code | 12 | 0.252 (0.136–0.383) | 0.335 | 0.989 | 0.929 | 1.000 |
| test | week3_combined | explanation | 12 | 1.346 (0.738–2.043) | 1.779 | 0.848 | 0.840 | 0.583 |
| test | week3_combined | documentation | 12 | 0.463 (0.275–0.640) | 0.568 | 0.647 | 0.586 | 1.000 |
| test | week3_combined | overall | 12 | 1.477 (0.838–2.177) | 1.914 | 0.852 | 0.751 | 0.667 |
| test | week8_llm | code | 12 | 1.750 (1.167–2.333) | 2.062 | 0.595 | 0.661 | 0.500 |
| test | week8_llm | explanation | 12 | 1.250 (0.833–1.667) | 1.500 | 0.943 | 0.901 | 0.500 |
| test | week8_llm | documentation | 12 | 0.500 (0.167–0.917) | 0.816 | 0.569 | 0.475 | 0.917 |
| test | week8_llm | overall | 12 | 2.500 (1.667–3.417) | 2.944 | 0.716 | 0.842 | 0.250 |
| all_v2_llm_only | week8_llm | code | 60 | 1.450 (1.233–1.667) | 1.688 | 0.767 | 0.760 | 0.617 |
| all_v2_llm_only | week8_llm | explanation | 60 | 1.300 (1.083–1.500) | 1.538 | 0.821 | 0.811 | 0.533 |
| all_v2_llm_only | week8_llm | documentation | 60 | 0.467 (0.333–0.617) | 0.730 | 0.645 | 0.620 | 0.967 |
| all_v2_llm_only | week8_llm | overall | 60 | 2.083 (1.717–2.467) | 2.553 | 0.766 | 0.768 | 0.383 |

## Class dataset (MU course, 62 notebooks): out-of-domain, unvalidated

Source: `results/class_mu/summary.json` (`python scripts/run_class.py`). The models were trained on synthetic labs and applied without retraining. The reference for each lab is **team-authored, not faculty-issued**. **No human marks exist**, so the table holds descriptive statistics of the predictions only. It is not accuracy.

| lab | n | overall mean (sd) | code mean | explanation mean | documentation mean | output match rate (mean) |
|---|---|---|---|---|---|---|
| MU_L01 | 8 | 6.768 (3.595) | 5.261 | 0.0 | 1.505 | 0.625 |
| MU_L02 | 8 | 8.812 (1.291) | 7.359 | 0.0 | 1.455 | 0.938 |
| MU_L03 | 7 | 7.681 (1.732) | 6.164 | 0.0 | 1.516 | 0.762 |
| MU_L04 | 6 | 5.822 (0.078) | 4.35 | 0.0 | 1.473 | None |
| MU_L05 | 7 | 9.3 (0.0) | 7.86 | 0.0 | 1.44 | 1.0 |
| MU_L06 | 8 | 7.705 (3.305) | 6.01 | 0.0 | 1.695 | 0.75 |
| MU_L07_08 | 7 | 8.32 (2.3) | 6.027 | 0.0 | 2.293 | 0.762 |
| MU_L09 | 7 | 9.13 (0.84) | 7.591 | 0.0 | 1.541 | 0.952 |
| MU_L10 | 4 | 9.203 (1.404) | 6.957 | 0.0 | 2.242 | 0.875 |

What this shows and what it does not:

- 62 of 62 kept notebooks were scored with 0 pipeline failures.
- Explanation is 0.0 for every notebook. Almost no class notebook has markdown explanations, so the explanation dimension cannot be assessed on this data. TF-IDF and MiniLM variants give identical marks for the same reason.
- MU_L04 has no expected outputs (random simulation), so output match is not available there.
- Known failure: a notebook that only prints the expected numbers gets the same code mark as a real solution (`demo/samples/failure_hardcoded_output_MU_L01.ipynb`: overall 10.89 vs 9.31 for a real class solution). The code is not executed and output matching cannot tell hard-coded output from computed output.

### Agreement with human marks (when they exist)

`data/class/annotations/marking_sheet.csv` has one row per kept notebook (`status=unmarked`). A marker fills the three marks, a justification and `status=marked`. Then:

```
python scripts/eval_class.py
```

This writes `results/class_mu/eval.json` with all the metrics above, per variant. With no marked rows it reports `"no human marks yet"` and no numbers. Five of the eight students are project team members, so marks from team members on their own work are a conflict of interest; `team_member_author` is kept in the sheet so those rows can be excluded.
