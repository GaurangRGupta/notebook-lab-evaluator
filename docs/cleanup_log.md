# Cleanup log

16 October 2026. Nothing was deleted. Retired material sits in `local_archive/`, which is gitignored. This file is the tracked record.

## Directories moved

| Old path | New path |
| --- | --- |
| `docs/` (course notes that were at the root before this phase) | `local_archive/fml_baseline/docs/` |
| `data/` (Foundations of Machine Learning notebooks, grades, splits) | `local_archive/fml_baseline/data/` |
| `results/` (old Week 2 metrics) | `local_archive/fml_baseline/results/` |
| `scripts/` (old baseline scripts) | `local_archive/fml_baseline/scripts/` |
| `src/speech_and_natural_language_processing_project/` | `local_archive/course_skeleton/src/speech_and_natural_language_processing_project/` |
| `uv.lock` | `local_archive/course_skeleton/uv.lock` |
| `README.md` (previous root readme, copied before the root file was replaced) | `local_archive/course_skeleton/README.md` |
| `sample_FOML_documents/` | `local_archive/sample_FOML_documents/` |
| `submission_details/` | `local_archive/submission_details/` |
| `J016_work/` | `local_archive/j016_work/` |
| `.venv` (the uv hello environment) | `local_archive/course_skeleton/.venv` |

`local_archive/j016_work/` was moved last. Its zips, virtual environments, `__MACOSX` folders, and older packages were not edited. The Week 6 zip inside it was built before the corrections below. It is a historical hand-in.

## Files copied into the tracked tree

The Week 6 package was copied from `J016_work/J016_Submission_5_nb_lab_evaluator 2/nb_lab_evaluator 2` onto the repository root. The copy left out `.venv`, `__pycache__`, `.pytest_cache`, and that package's `.gitignore`. Robocopy reported 581 files copied, about 90 MB, exit code 3 (files copied, and the destination already had a few files). The original tree stayed inside `J016_work` and then moved with that folder.

After that copy:

- `docs/week1_proposal.md` was copied back from `local_archive/fml_baseline/docs/week1_proposal.md`.
- The empty class layout was copied from the Week 4 package into `data/class/`.
- `docs/foml_pilot_inventory.md` was copied from the Week 4 package. Its path line now says `local_archive/sample_FOML_documents/`.

`class_list.txt` and the annotation files are empty templates. They have no student names. Notebooks later placed in `data/class/raw/` and hashed copies in `data/class/interim/` are gitignored.

## Code changes

Shared rules are in `src/reporting.py`. `src/asset.py` and `Pipeline.reports` call it. The sentences are not copied into both files.

| Change | Module | Test |
| --- | --- | --- |
| Code, explanation, and documentation are marks. Overall is the rounded sum of the raw marks. Code is no longer stored as a fraction of 8. | `src/reporting.py` `marks`, used by `src/pipeline.py` and `src/asset.py` | `tests/test_reporting.py::test_marks_share_one_scale_and_sum_to_overall` |
| Feedback uses the semantic missing-idea list when the encoder ran, and the lexical list when it did not. | `choose_missing`, `feedback_and_flag` | `test_feedback_names_the_semantic_list_when_the_encoder_ran`, `test_feedback_names_the_lexical_list_when_the_encoder_is_off` |
| Confidence is null where `week2_tfidf` and `week3_combined` share a feature list. Overall confidence is the explanation value. | `agreement_confidence` | `test_confidence_is_null_where_feature_lists_match`, `tests/test_week6_asset.py::test_response_ranges_and_feedback` |
| Diagnostic confidence remains for a run with the encoder off. | `diagnostic_confidence` | `test_diagnostic_confidence_when_the_encoder_is_off` |
| `Pipeline.reports` fills feedback and confidence with those helpers. | `src/pipeline.py` | covered by the reporting tests and by the synthetic end-to-end run, which writes `results/synthetic_dev/reports/` |
| A placeholder salt `${NB_EVAL_SALT}` with the variable unset stops the pipeline before scoring. | `Pipeline.anonymise` (already present) | `tests/test_pipeline_units.py::test_placeholder_salt_stops_before_scoring` |
| Asset and API source do not import `nbclient`. | `src/asset.py`, `src/api/app.py` | `tests/test_week6_asset.py::test_asset_does_not_import_a_kernel` |
| `POST /evaluate-notebook` returns 400 for an unknown lab id and 200 for `LAB_SYN1`. | `src/api/app.py` | `test_http_evaluate_and_reject` |
| A one-name `compare` list no longer indexes a missing second name. | `Pipeline.model` | no separate test. The Week 5 and Week 6 configs pass two names. |
| The frozen Week 2 prediction check allows a difference below 1e-9. A fresh run differed by about 1e-15, which is the last bit of a float. | `tests/test_week4.py` | `test_week2_baseline_frozen` |

`results/week6/asset/meta.json` now stores the same confidence sentence the function returns. The joblib files were not refit. `scripts/fit_week6_asset.py` reads that sentence from `src/reporting.py`.

On `data/synthetic_v2/raw/LAB_SYN1/V2_S06_LAB_SYN1.ipynb` the live response is code 7.94, explanation 7.81, documentation 2.43, overall 18.17. Those marks match the previous response. The lexical check still names `direction` for task T3. The encoder accepts that idea, so the served missing list is empty and the flag is false. Explanation confidence is 0.976. Code and documentation confidence are null. Overall confidence is 0.976.

`docs/API.md`, `docs/week6_note.md`, `results/week6/example_response.json`, and `results/week6/Group3_Week6_Asset.docx` were rewritten from that live call. The docx passed `validate.py`.

`docs/limitations.md` records the limits this phase did not treat as fitting bugs: rule labels, circular code features, 17 of 20 test explanations copied into training, equal markdown and explanation token counts, paraphrase cosine 0.424 against 0.861 for a wrong answer that reuses the reference words, and a Week 3 versus Week 2 difference whose interval includes zero. The concept threshold, the split seed, and the synthetic salt were not changed. `nli.enabled` stays false. The v2 notebooks were not regenerated. NLI weights were not downloaded.

## Commands

| Command | Result |
| --- | --- |
| `python -m pytest -q tests` with `HF_HUB_OFFLINE=1`, first run | 1 failed, 81 passed, 4 skipped. The failure was `test_week2_baseline_frozen`: `DataFrame.equals` rejected a 1e-15 difference. |
| same command after the tolerance change | 82 passed, 4 skipped, 64.20 seconds. |
| `POST /evaluate-notebook` for `LAB_SYN1` through `TestClient` | HTTP 200. The body matched `evaluate_notebook` on the same file. |
| same route with assignment id `NO_SUCH_LAB` | HTTP 400. |
| `validate.py` on `results/week6/Group3_Week6_Asset.docx` | All validations passed. |

Week 5 files were hashed after the tests. They still match the copies made at the start of this phase:

| File | SHA256 | Bytes | Last write (UTC) |
| --- | --- | --- | --- |
| `results/week5/metrics.csv` | `B2A8F894E961E42CC066612A48EFCCB3ABDA9C75DBD1E1211CD512C67C4B2EE2` | 80527 | 2026-10-09T16:26:56.9024567Z |
| `results/week5/Group3_Week5_Evaluation_Report.docx` | `2DC9F8D533D458639F3807378E5543FC29EDA564570D2039C7E184924FD06BAA` | 46201 | 2026-10-09T16:26:56.7485370Z |
| `results/week5/Group3_Week5_Addendum.docx` | `48074A879D3FE75FDC647A81A80B9F8E2405BECEC131CDF44F46E278B53B4371` | 12024 | 2026-10-09T16:49:03.5412194Z |
| `configs/week5_v2.yaml` | `7AD925EAA1283DED07A083FEBED657BE98B46E602BECFD0040B81FA8ABB3D28C` | 1536 | 2026-10-09T16:27:01.8829714Z |

The end-to-end test rewrote `results/synthetic_dev/` because that config is what the test runs. `results/week5/` was not rewritten. `results/week6/reports/` is still the fit-time JSON: code stored as a fraction of 8, feedback and confidence empty. A new pipeline run would write the corrected fields. This phase did not refit, because the response builder does not change the Ridge models.

`uv.lock` was not regenerated. The hello lockfile is in `local_archive/course_skeleton/uv.lock`. `pyproject.toml` now describes the evaluator and lists the same libraries as `requirements.txt`. The hello console script is not registered.

## Unfinished

There is no git remote. This phase makes one local commit and does not push. A push needs a repository URL.

Real class notebooks are still absent. `data/class/` is empty. The Week 4 NLI model is still an incomplete download, and this phase did not fetch it. The Grok Bot migration was left for later.
