# Notebook lab evaluator API

Group 3. Rolls J013, J015, J016, J017, J018.

Another project can call this asset without reading the training code. The call sends a Jupyter notebook and a lab id. The response is one JSON object. The notebook is not executed. Scoring uses the saved outputs already in the file.

The served model is `week3_combined` from the Week 5 fit: TF-IDF plus a frozen ONNX copy of `sentence-transformers/all-MiniLM-L6-v2`. A second frozen model, `week2_tfidf`, is loaded only to compute confidence. The Week 4 NLI model is not part of this asset.

Marks are code 0-8, explanation 0-8, documentation 0-4. Overall is the sum, out of 20. These are the same scales as the Week 5 tables.

## Setup

From the package directory:

```bash
python -m venv .venv
pip install -r requirements.txt
python scripts/fetch_models.py
python scripts/fit_week6_asset.py
python -m src.api.app
```

The server listens on `127.0.0.1:8000` only. `fit_week6_asset.py` writes `results/week6/` and does not rewrite `results/week5/`.

On this machine the refit matched the Week 5 predictions. Across 216 prediction cells the largest absolute difference was 0.000001 marks.

## Function

```python
from src.asset import evaluate_notebook, load_asset

asset = load_asset()
notebook = open("data/synthetic_v2/raw/LAB_SYN1/V2_S06_LAB_SYN1.ipynb", "rb").read()
report = evaluate_notebook(notebook, "LAB_SYN1", asset)
```

`assignment_id` is `LAB_SYN1` or `LAB_SYN2`. Those ids name the lab cards already in `data/synthetic_v2/references/`. An unknown id raises `ValueError`.

## HTTP

`GET /health`

Returns the scoring variant, the encoder name, and the data-status string.

`POST /evaluate-notebook`

| Parameter | Where | Meaning |
| --- | --- | --- |
| file | multipart file | A `.ipynb` file |
| assignment_id | form field | `LAB_SYN1` or `LAB_SYN2` |

```bash
curl -s -X POST http://127.0.0.1:8000/evaluate-notebook ^
  -F "assignment_id=LAB_SYN1" ^
  -F "file=@data/synthetic_v2/raw/LAB_SYN1/V2_S06_LAB_SYN1.ipynb"
```

An empty upload, a file that is not a notebook, or an unknown `assignment_id` returns HTTP 400.

## Example response

This is the live response for `data/synthetic_v2/raw/LAB_SYN1/V2_S06_LAB_SYN1.ipynb`.

```json
{
  "assignment_id": "LAB_SYN1",
  "scoring_variant": "week3_combined",
  "code": 7.94,
  "explanation": 7.81,
  "documentation": 2.43,
  "overall": 18.17,
  "max_score": 20,
  "output_match_rate": 1.0,
  "errors": [],
  "missing_concepts": {},
  "feedback": "The saved output matches, and the write-up covers the reference ideas this check looks for.",
  "confidence": {
    "code": null,
    "explanation": 0.976,
    "documentation": null,
    "overall": 0.976,
    "method": "Agreement between week2_tfidf and week3_combined is computed only where their feature lists differ. This is not a calibrated probability."
  },
  "flag_output_ok_explanation_weak": false,
  "data_status": "SYNTHETIC_DEVELOPMENT_DATA_V2_RULE_LABELS"
}
```

`flag_output_ok_explanation_weak` is true when every expected output matches and the chosen missing-idea list is not empty.

## Confidence

Code and documentation use the same features in `week2_tfidf` and `week3_combined`, so those two confidences are null. Explanation uses

```text
1 - min(1, abs(week2_tfidf - week3_combined) / 8)
```

Overall confidence is that explanation value. In the example above it is 0.976. The value is agreement between two frozen models on the one dimension where their feature lists differ. It is not a probability from a calibration set.

## Feedback

Feedback is built from reference ideas the write-up does not cover. When the encoder is loaded, that list is the semantic one. When the encoder is off, the list is the lexical one. No external model is called.

On this notebook the lexical check still names `direction` for task T3. The encoder accepts that idea, so the served list is empty, the flag is false, and the feedback says the write-up covers the ideas this check looks for.

## Limits

- The numbers are on team-written notebooks. v2 marks come from planted rules. They are not faculty marks.
- Code is judged from saved outputs, syntax, and expected constructs. A kernel is not started.
- The run stays on this machine. Student text is not sent to an external API.
- Real class notebooks are not in this asset yet. The two lab ids above are the synthetic cards.
- Week 5 did not show that `week3_combined` beats `week2_tfidf`. The test explanation MAE was 1.346 against 1.512, and the 95% interval on the difference included zero. See `docs/limitations.md`.
- Files already in `results/week5/`, and the fit-time copies in `results/week6/reports/`, still store code as a fraction of 8 and leave feedback empty. A new call stores code as a mark from 0 to 8.
