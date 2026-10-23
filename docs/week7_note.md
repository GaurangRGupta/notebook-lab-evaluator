# Week 7: demo, API handoff, and responsible AI

Group 3. Rolls J013, J015, J016, J017, J018.

This week puts a page on the server that already scores notebooks, and a second route that reads one score and says whether a person should review it. The served marks stay the Ridge marks from `week3_combined`. The optional Grok grader stays off. No model was refit.

The Word copy of this note is `results/week7/Group3_Week7_Integration.docx`.

## What this week adds

1. A demo page at `http://127.0.0.1:8000/`.
2. `POST /support-signal`, a reader for a student-support tool.
3. The architecture picture in `docs/architecture.png` and `docs/architecture.md`.
4. The responsible-AI page in `docs/responsible_ai.md`.

## How to run the demo

From the package directory, when `results/week6/asset/` and `models/all-MiniLM-L6-v2-onnx/` are already present:

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
python -m src.api.app
```

Leave that window open. In a browser on this machine, open `http://127.0.0.1:8000/`.

Choose a file and a lab id, then submit. The page sends the file to `POST /evaluate-notebook`. It shows code, explanation, documentation, overall, output match, errors, missing ideas, feedback, the weak-explanation flag, and confidence. It then posts that JSON to `POST /support-signal` and shows the review record. The second call does not score the notebook again.

The page says the student's code is not executed and the mark is a suggestion for the instructor.

If the asset folder or the encoder folder is missing, run `python scripts/fetch_models.py` and then `python scripts/fit_week6_asset.py` before starting the server.

## Three notebooks to try

| File | Lab id | Overall | What the page should show |
| --- | --- | --- | --- |
| `examples/language_model_lab.ipynb` | `LAB_SYN2` | 15.14 | Output match 1.0, flag true, review needed. Task T3 misses inverse probability and surprise. |
| `examples/tfidf_lab.ipynb` | `LAB_SYN1` | 18.17 | Output match 1.0, flag false, no errors, review not requested. |
| `examples/language_model_weak.ipynb` | `LAB_SYN2` | 7.28 | Output match 0, syntax errors in the file, review needed. |

The first row is the one to show if there is time for only one file. The saved output matches, and the write-up still misses reference ideas, so the flag and the support signal both ask for a person.

## Language-model example

Marks for `examples/language_model_lab.ipynb` and `LAB_SYN2`:

- code 7.87
- explanation 5.13
- documentation 2.14
- overall 15.14 out of 20
- output match 1.0
- `flag_output_ok_explanation_weak` true
- explanation confidence 0.986

The support signal for that score:

```json
{
  "source": "notebook-lab-evaluator",
  "assignment_id": "LAB_SYN2",
  "overall": 15.14,
  "max_score": 20,
  "needs_human_review": true,
  "reason": "The saved output matches. The write-up does not cover the reference ideas named below. The write-up for T3 does not cover: inverse probability, surprise."
}
```

## Where this component sits

The vertical picture is in `docs/architecture.png`. In words:

1. Input. One `.ipynb` file and a lab id. The answer card stays in the package.
2. Preprocessing. Anonymise, parse, align tasks, drop template cells.
3. NLP model. TF-IDF plus the frozen MiniLM encoder, then Ridge. The served variant is `week3_combined`.
4. Post-processing. Marks, missing ideas, a feedback sentence, confidence, and the weak-explanation flag.
5. API. `POST /evaluate-notebook` on this machine, port 8000.
6. Application. The demo page, and `POST /support-signal`.

In the course picture this is the Labs box under Academic Services. It uses semantic similarity and an evaluation service. The support signal is what an Early Warning reader under Student Services would store.

This is not a live call to another group's server. We do not have their URL, and this hand-in does not build their component. Another group calls us with a notebook and a lab id. A student-support tool calls `/support-signal` with the JSON we returned.

## How another application calls the two routes

Score one notebook:

```bash
curl.exe -s -X POST http://127.0.0.1:8000/evaluate-notebook -F "assignment_id=LAB_SYN2" -F "file=@examples/language_model_lab.ipynb"
```

Ask whether a person should review a score already in hand. `report` is the JSON from the call above.

```python
import httpx

signal = httpx.post("http://127.0.0.1:8000/support-signal", json=report)
signal.raise_for_status()
print(signal.json()["needs_human_review"])
```

For the language-model example that prints `True`.

Field list, error cases, and the answer-card files are in `docs/manual.md`. The short call is `docs/API.md`.

## Responsible AI

The full page is `docs/responsible_ai.md`. Short form:

- Privacy. Roll-number patterns are removed before scoring. The salt stays outside the repo. The server stays on this machine. Class notebooks are not in this package.
- Bias. A wrong answer that reuses the reference words can look close. A correct paraphrase can score too low. The recorded MiniLM means are 0.861 and 0.424. The test labels are rules, not faculty marks.
- Over-confidence. Confidence is agreement between Week 2 and Week 3 on the explanation. It is not a probability.
- Human oversight. The instructor sets the mark. The flag and `needs_human_review` mean a person should read the notebook.
- Security. Student code is not executed.
- Limits. On 12 synthetic test notebooks, Week 3 is not shown to beat Week 2. A new lab needs a new five-file card.

## Left for 31 October

The class demo is 31 October 2026, 9:00 to 13:00, 15 minutes. Still to write: the 10-page report, the slides, and a faculty answer card when one exists. This week does not add those.
