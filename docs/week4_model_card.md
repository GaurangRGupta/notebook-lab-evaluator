# Week 4 Improved Model: NLI consistency features

| | |
|---|---|
| Versions | **Week 2 Baseline** (`week2_tfidf`, frozen) · **Week 3 Semantic Model** (`week3_combined`) · **Week 4 Improved Model** (`week4_improved`) |
| Change | Explanation model gains `nli_support`, `nli_coverage`, `nli_contradiction` from a frozen NLI cross-encoder |
| Model | `cross-encoder/nli-deberta-v3-xsmall` (pretrained on SNLI + MultiNLI), local CPU, inference only; label order read from its config |
| Estimator | Ridge per dimension; random forest only if a pre-registered train+val CV rule passes (>= 10% lower MAE) |
| Code track | Unchanged; code correctness still comes from deterministic checks |
| Leakage controls | frozen models, train-only fitting, val-only alpha, train+val-only estimator rule, test scored once, Week 2 predictions frozen and tested |
| Report additions | per-task NLI scores; "possible conceptual error" flags when P(contradiction) >= 0.5 (not feedback text) |
| Status | Code complete and tested with stubs; model results via `scripts/run_week4.py`. Synthetic development data only. |

Known risks: NLI models are trained on everyday sentences, not technical NLP explanations; long references are
truncated by the model; a contradiction score is a signal for the scorer and for review, not a verdict.
