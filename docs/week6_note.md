# Week 6 note

Group 3. Rolls J013, J015, J016, J017, J018.

This page is the Week 6 hand-in note. The callable asset is documented in `docs/API.md`.

The asset scores one `.ipynb` file against a lab card. The call is `evaluate_notebook(notebook_bytes, assignment_id)` or `POST /evaluate-notebook` on `127.0.0.1:8000`. The body is the notebook file plus `assignment_id` (`LAB_SYN1` or `LAB_SYN2`). The response is JSON: code, explanation, and documentation as marks, overall out of 20, output match, errors, missing ideas, a short feedback sentence, and a confidence object.

The served score is `week3_combined`. Confidence is the agreement between that model and `week2_tfidf` on dimensions whose feature lists differ. Code and documentation use the same features in both, so those confidences are null. Overall confidence is the explanation value. It is not a calibrated probability.

Feedback names reference ideas the write-up does not cover. When the encoder is loaded, the list is the semantic one. For `data/synthetic_v2/raw/LAB_SYN1/V2_S06_LAB_SYN1.ipynb` the live response is code 7.94, explanation 7.81, documentation 2.43, overall 18.17, and explanation confidence 0.976. The lexical check still names "direction" for task T3. The encoder accepts that idea, so the served missing list is empty and the flag is false.

The Week 6 refit matches the Week 5 predictions. The largest absolute difference across 216 cells is 0.000001 marks. `results/week5/` was not rewritten. The Week 4 NLI model is not in this asset. The notebooks are team-written and the marks are rule labels, not faculty marks. Student code is not executed.
