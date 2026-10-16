# Week 4 model-free results (SYNTHETIC development results)

Computed in the development workspace, where the pretrained models could not be downloaded:
* `../week2_baseline_week4_protocol/`: frozen Week 2 Baseline evaluated with the Week 4 protocol
  (hold-out val/test, grouped CV on train+val, overall score). Hold-out predictions identical to the frozen snapshot.
* `robustness_tfidf_only*.csv`: the robustness suite scored with TF-IDF only.
Week 3 / Week 4 model numbers are produced by `python scripts/run_week4.py` (needs internet once).
