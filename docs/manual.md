# Notebook grader manual

Group 3. Rolls J013, J015, J016, J017, J018.

This is the page to keep open while you call the grader. The same manual in Word is `results/week6/Group3_Week6_Asset.docx`. Limits of the fitted models are in `docs/limitations.md`.

Two functions do the work. Both are in `src.asset`.

- `load_asset` reads the fitted files and the answer cards.
- `evaluate_notebook` scores one notebook and returns a dict.

## 1. What it does

You give it one Jupyter notebook and a lab id. It returns one JSON object: marks, whether the saved output matches, errors found in the file, ideas the write-up does not cover, a short feedback sentence, and a confidence value.

The student's code is not executed. Nothing in the notebook is run. The grader reads the cells and the outputs that are already saved in the file, and it compares those with an answer card stored in this package.

You do not send the answer card with the call. The lab id is the name of the card folder.

Two cards are included:

| Lab id | Lab | Example notebook |
| --- | --- | --- |
| `LAB_SYN2` | Language model: add-one smoothing and perplexity | `examples/language_model_lab.ipynb` |
| `LAB_SYN1` | TF-IDF and cosine similarity | `examples/tfidf_lab.ipynb` |

The two cards were prepared for this project. The marks follow that rubric: code 0 to 8, explanation 0 to 8, documentation 0 to 4. Overall is the sum, out of 20.

## 2. What you need

- Python 3.12 or newer
- The libraries in `requirements.txt`
- The fitted files in `results/week6/asset/`
- The encoder files in `models/all-MiniLM-L6-v2-onnx/`

From the package directory:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\pip install -r requirements.txt
```

Other systems:

```bash
.venv/bin/pip install -r requirements.txt
```

If those two folders are already in the package, skip the next two commands and start the server or call the function. If either folder is missing:

```bash
python scripts/fetch_models.py
python scripts/fit_week6_asset.py
```

`fetch_models.py` downloads the checked encoder and verifies three published similarities. `fit_week6_asset.py` writes `results/week6/asset/`. You do not run either command again for each student notebook.

## 3. Quick start

Run this from the package directory, so `import src.asset` can see the `src` folder.

```python
from pathlib import Path
from src.asset import evaluate_notebook, load_asset

asset = load_asset()
notebook = Path("examples/language_model_lab.ipynb").read_bytes()
report = evaluate_notebook(notebook, "LAB_SYN2", asset)

print(report["code"], report["explanation"], report["documentation"], report["overall"])
print(report["feedback"])
```

That program prints:

```text
7.87 5.13 2.14 15.14
The saved output matches. The write-up does not cover the reference ideas named below. The write-up for T3 does not cover: inverse probability, surprise.
```

Call `load_asset()` once and reuse `asset` for every notebook. Each call to `evaluate_notebook` is independent.

## 4. load_asset

```text
load_asset(asset_dir=None)
```

`asset_dir` is the folder that holds `meta.json`, `tfidf.joblib`, and `variant_models.joblib`. The default is `results/week6/asset`. Pass a `pathlib.Path`, or leave it out.

The function also reads the paths stored in `meta.json`: the encoder in `models/all-MiniLM-L6-v2-onnx/`, and every card folder under `data/synthetic_v2/references/`.

It returns an `Asset`. You pass that object into `evaluate_notebook`.

If the folder has no `meta.json`, it raises `FileNotFoundError`. A missing folder named `results/week6/missing` raises:

```text
No fitted asset at results\week6\missing. Run: python scripts/fit_week6_asset.py
```

The folder in that message is the path you passed. The command at the end is the fit command.

## 5. evaluate_notebook

```text
evaluate_notebook(notebook, assignment_id, asset=None)
```

| Argument | What to pass |
| --- | --- |
| notebook | The file bytes, or a `pathlib.Path`. A plain string is parsed as the notebook text, so pass `Path("file.ipynb")` or the bytes from `read_bytes()`. |
| assignment_id | `LAB_SYN1` or `LAB_SYN2`, or the folder name of a card you have added |
| asset | The object from `load_asset()`. If you omit it, the function loads the asset itself. |

The return value is a dict. Section 8 lists every key. Over HTTP the same dict is sent as JSON, with `None` written as `null` and `True` / `False` written as `true` / `false`.

An unknown lab id raises `ValueError`:

```text
Unknown assignment_id 'NO_SUCH_LAB'. Known ids: LAB_SYN1, LAB_SYN2.
```

## 6. Score one notebook over HTTP

Start the server from the package directory and leave that window open:

```bash
python -m src.api.app
```

It listens on `http://127.0.0.1:8000` only. Another machine cannot call it. Notebook text is not sent off this computer.

Check that the models loaded:

```bash
curl.exe http://127.0.0.1:8000/health
```

A live health response is:

```json
{
  "status": "ok",
  "scoring_variant": "week3_combined",
  "encoder": "sentence-transformers/all-MiniLM-L6-v2 (ONNX export)",
  "data_status": "SYNTHETIC_DEVELOPMENT_DATA_V2_RULE_LABELS"
}
```

From a second window, `httpx` posts the same file. `httpx` is installed with `requirements.txt`. The server from the first window has to be running.

```python
from pathlib import Path
import httpx

path = Path("examples/language_model_lab.ipynb")
with path.open("rb") as handle:
    response = httpx.post(
        "http://127.0.0.1:8000/evaluate-notebook",
        data={"assignment_id": "LAB_SYN2"},
        files={"file": ("language_model_lab.ipynb", handle, "application/octet-stream")},
    )
response.raise_for_status()
print(response.json()["overall"])
```

That call prints `15.14`.

The same request with `curl.exe`. On Windows, use `curl.exe`. In PowerShell, `curl` is a different command and will not accept `-F`. This one line works in PowerShell and in the Command Prompt.

```bash
curl.exe -s -X POST http://127.0.0.1:8000/evaluate-notebook -F "assignment_id=LAB_SYN2" -F "file=@examples/language_model_lab.ipynb"
```

The same call for the TF-IDF example:

```bash
curl.exe -s -X POST http://127.0.0.1:8000/evaluate-notebook ^
  -F "assignment_id=LAB_SYN1" ^
  -F "file=@examples/tfidf_lab.ipynb"
```

| Piece | Where | Meaning |
| --- | --- | --- |
| file | multipart file | The student's `.ipynb` |
| assignment_id | form field | The lab id |

These three uploads return HTTP 400:

| What you sent | Response detail |
| --- | --- |
| An empty file | `The upload is empty.` |
| Bytes that are not a notebook | `The upload is not a readable notebook: Notebook does not appear to be JSON: 'not a notebook'` |
| Lab id `NO_SUCH_LAB` | `Unknown assignment_id 'NO_SUCH_LAB'. Known ids: LAB_SYN1, LAB_SYN2.` |

A successful call returns HTTP 200 and the same JSON object as `evaluate_notebook`.

### 6.1 Demo page

With the server running, open `http://127.0.0.1:8000/` in a browser on this machine. Choose the notebook file and the lab id, then submit. The page sends the file to `POST /evaluate-notebook` and shows the marks, the feedback, the flag, and the confidence. It then sends that JSON to `POST /support-signal`. The second call does not score the notebook again.

`POST /support-signal` returns a short record for a student-support reader:

| Field | Meaning |
| --- | --- |
| source | `notebook-lab-evaluator` |
| assignment_id | The lab id from the score |
| overall | The overall mark |
| max_score | 20 for the cards in this package |
| needs_human_review | True when a person should read the notebook |
| reason | The feedback sentence already on the score |

`needs_human_review` is true when the weak-explanation flag is true, or the saved output does not fully match, or the errors list is not empty. A body with no output match also asks for a review.

For `examples/language_model_lab.ipynb` with `LAB_SYN2` the flag is true, so `needs_human_review` is true. For `examples/tfidf_lab.ipynb` with `LAB_SYN1` the flag is false, the output match is 1.0, and the errors list is empty, so `needs_human_review` is false. For `examples/language_model_weak.ipynb` the output match is 0, so `needs_human_review` is true.

The page uses the served marks. It does not turn on the optional Grok grader. The page says the student's code is not executed and the mark is a suggestion for the instructor.

The same handoff from Python, using a score you already have in `report`:

```python
import httpx

signal = httpx.post("http://127.0.0.1:8000/support-signal", json=report)
signal.raise_for_status()
print(signal.json()["needs_human_review"])
```

## 7. What you send, and what you do not send

Send the notebook file and the lab id.

Do not send the answer card, the mark scheme as a separate file, or any other student's notebook. The card is already on disk. The fitted models are already on disk. One request does not look through the other notebooks in the package.

The notebook should be a normal `.ipynb` with the lab's task headings (`## Task 1`, `## Task 2`, `## Task 3`), code cells, and saved outputs. Cells that match the handout in the card are treated as starter text and left out of the student's writing.

## 8. What you get back

| Field | Meaning |
| --- | --- |
| assignment_id | The lab id you sent |
| scoring_variant | The model that produced the marks. This package serves `week3_combined`. |
| code | Code mark, from 0 to 8. Read from syntax, saved errors, execution counts, and saved output. |
| explanation | Explanation mark, from 0 to 8. Read from the markdown write-up compared with the card. |
| documentation | Documentation mark, from 0 to 4. Read from comments, docstrings, and the amount of markdown. |
| overall | The three raw marks added together, then rounded. Adding the three printed marks can differ by one hundredth. |
| max_score | 20 |
| output_match_rate | Share of tasks whose saved output matches the card. `1.0` means every compared task matched. `0.0` means none matched. |
| errors | A list. Each item is a syntax error, a runtime error saved in a cell, an output mismatch, or a code cell that was never executed. An empty list means none of those were found. |
| missing_concepts | Reference ideas the write-up does not cover, grouped by task. An empty object means this check found no gap. |
| feedback | One or more sentences built from `missing_concepts` and the output match. |
| confidence | Agreement between two frozen models. `code` and `documentation` are `null` in this package, because those two models use the same inputs for those marks. `explanation` and `overall` are the same number, from 0 to 1. |
| flag_output_ok_explanation_weak | `true` only when every saved output matches and the write-up still misses at least one reference idea. |
| data_status | The label stored with this fitted asset. |

`confidence.method` is this sentence: "Agreement between week2_tfidf and week3_combined is computed only where their feature lists differ. This is not a calibrated probability."

The explanation number is `1 - min(1, abs(week2_tfidf - week3_combined) / 8)`. A value near 1 means the two models gave close explanation marks. It is not a probability that the mark is correct.

## 9. Example: language-model lab, output matches, write-up has a gap

File: `examples/language_model_lab.ipynb`

Lab id: `LAB_SYN2`

The saved output matches the card on every task. The write-up for task T3 does not cover two ideas from the card, "inverse probability" and "surprise". The flag is true because the output is fine and the write-up is still missing those ideas.

```python
import json
from pathlib import Path
from src.asset import evaluate_notebook, load_asset

asset = load_asset()
report = evaluate_notebook(Path("examples/language_model_lab.ipynb"), "LAB_SYN2", asset)
print(json.dumps(report, indent=2))
```

`json.dumps` prints the object below. Saved copy: `results/week6/example_language_model.json`

```json
{
  "assignment_id": "LAB_SYN2",
  "scoring_variant": "week3_combined",
  "code": 7.87,
  "explanation": 5.13,
  "documentation": 2.14,
  "overall": 15.14,
  "max_score": 20,
  "output_match_rate": 1.0,
  "errors": [],
  "missing_concepts": {
    "T3": ["inverse probability", "surprise"]
  },
  "feedback": "The saved output matches. The write-up does not cover the reference ideas named below. The write-up for T3 does not cover: inverse probability, surprise.",
  "confidence": {
    "code": null,
    "explanation": 0.986,
    "documentation": null,
    "overall": 0.986,
    "method": "Agreement between week2_tfidf and week3_combined is computed only where their feature lists differ. This is not a calibrated probability."
  },
  "flag_output_ok_explanation_weak": true,
  "data_status": "SYNTHETIC_DEVELOPMENT_DATA_V2_RULE_LABELS"
}
```

How to read it: code 7.87 out of 8, explanation 5.13 out of 8, documentation 2.14 out of 4, overall 15.14 out of 20. No syntax errors and no output mismatches. The feedback names the gap. Confidence on explanation is 0.986.

## 10. Example: TF-IDF lab, output matches and the write-up covers the ideas

File: `examples/tfidf_lab.ipynb`

Lab id: `LAB_SYN1`

The card for this lab expects saved lines of this kind: vocabulary size, TF-IDF matrix shape, and a cosine similarity. On this notebook every compared output matches, and the idea check does not report a gap. The flag is false. The feedback says the output matches and the write-up covers the ideas this check looks for.

```python
from pathlib import Path
from src.asset import evaluate_notebook, load_asset

asset = load_asset()
report = evaluate_notebook(Path("examples/tfidf_lab.ipynb"), "LAB_SYN1", asset)
print("code", report["code"])
print("explanation", report["explanation"])
print("documentation", report["documentation"])
print("overall", report["overall"])
print("errors", report["errors"])
print("feedback", report["feedback"])
print("flag", report["flag_output_ok_explanation_weak"])
```

That program prints:

```text
code 7.94
explanation 7.81
documentation 2.43
overall 18.17
errors []
feedback The saved output matches, and the write-up covers the reference ideas this check looks for.
flag False
```

Adding the three printed marks gives 18.18. `overall` is 18.17, because it is the rounded sum of the raw marks, and the printed marks are already rounded.

Saved copy: `results/week6/example_tfidf.json`

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

Same call, different card. The lab id is the only change. The marks are not a second copy of the language-model result.

## 11. Example: language-model lab, broken code and output that does not match

File: `examples/language_model_weak.ipynb`

Lab id: `LAB_SYN2`

This is the same card as the language-model example above, and a different notebook. Two code cells have a syntax error saved in the file. None of the three saved outputs match the card, so `output_match_rate` is 0.0. The flag is false, because the flag requires a full output match. The feedback names the missing ideas. The first sentence is the task T2 gap.

```python
from pathlib import Path
from src.asset import evaluate_notebook, load_asset

asset = load_asset()
report = evaluate_notebook(Path("examples/language_model_weak.ipynb"), "LAB_SYN2", asset)
print("output_match_rate", report["output_match_rate"])
print("errors:")
for item in report["errors"]:
    print(" ", item)
print("feedback", report["feedback"])
```

That program prints:

```text
output_match_rate 0.0
errors:
  cell 7 (T2): SyntaxError line 8: '(' was never closed
  cell 11 (T3): SyntaxError line 10: '(' was never closed
  output mismatch in T1
  output mismatch in T2
  output mismatch in T3
feedback The write-up for T2 does not cover: zero probability, unseen bigrams, add one, vocabulary size, probability mass. The write-up for T3 does not cover: inverse probability.
```

Saved copy: `results/week6/example_language_model_weak.json`

```json
{
  "assignment_id": "LAB_SYN2",
  "scoring_variant": "week3_combined",
  "code": 0.14,
  "explanation": 4.77,
  "documentation": 2.37,
  "overall": 7.28,
  "max_score": 20,
  "output_match_rate": 0.0,
  "errors": [
    "cell 7 (T2): SyntaxError line 8: '(' was never closed",
    "cell 11 (T3): SyntaxError line 10: '(' was never closed",
    "output mismatch in T1",
    "output mismatch in T2",
    "output mismatch in T3"
  ],
  "missing_concepts": {
    "T2": ["zero probability", "unseen bigrams", "add one", "vocabulary size", "probability mass"],
    "T3": ["inverse probability"]
  },
  "feedback": "The write-up for T2 does not cover: zero probability, unseen bigrams, add one, vocabulary size, probability mass. The write-up for T3 does not cover: inverse probability.",
  "confidence": {
    "code": null,
    "explanation": 0.928,
    "documentation": null,
    "overall": 0.928,
    "method": "Agreement between week2_tfidf and week3_combined is computed only where their feature lists differ. This is not a calibrated probability."
  },
  "flag_output_ok_explanation_weak": false,
  "data_status": "SYNTHETIC_DEVELOPMENT_DATA_V2_RULE_LABELS"
}
```

How to read it: the code mark is near the bottom of the 0 to 8 scale because the file contains syntax errors and the saved outputs do not match. The explanation mark is separate. A notebook can fail the code check and still receive an explanation mark from the writing that is there. Documentation is separate again.

## 12. Score several notebooks

Reuse one `Asset` and loop. This scores the three example files:

```python
from pathlib import Path
from src.asset import evaluate_notebook, load_asset

asset = load_asset()
jobs = [
    ("examples/language_model_lab.ipynb", "LAB_SYN2"),
    ("examples/tfidf_lab.ipynb", "LAB_SYN1"),
    ("examples/language_model_weak.ipynb", "LAB_SYN2"),
]
for path, lab in jobs:
    report = evaluate_notebook(Path(path), lab, asset)
    print(lab, Path(path).name, report["overall"])
```

That program prints:

```text
LAB_SYN2 language_model_lab.ipynb 15.14
LAB_SYN1 tfidf_lab.ipynb 18.17
LAB_SYN2 language_model_weak.ipynb 7.28
```

## 13. The answer card

Each lab is one folder. The folder name is the lab id you pass in the call.

Language-model card: `data/synthetic_v2/references/LAB_SYN2/`

TF-IDF card: `data/synthetic_v2/references/LAB_SYN1/`

| File | Role |
| --- | --- |
| `template.ipynb` | The handout the class was given. Matching cells are starter text and are not scored as the student's writing. |
| `reference_solution.ipynb` | A worked solution. The code check looks for the same kinds of calls. A student may use different helper names. |
| `reference_explanations.json` | The written answer for each task, plus the idea words the write-up should cover. |
| `expected_outputs.json` | The saved output a correct notebook should contain, and how to compare it. |
| `rubric.json` | The ceilings: code 8, explanation 8, documentation 4. |

The language-model card compares saved output as a number within 0.001 of these lines:

| Task | Expected saved line |
| --- | --- |
| T1 | `count(the, cat) = 2` |
| T2 | `P(cat \| the) = 0.3` |
| T3 | `Perplexity: 4.472` |

The TF-IDF card uses the same kind of numeric comparison for vocabulary size, matrix shape, and cosine similarity.

The idea list for the language-model write-up includes, among others, "zero probability", "unseen bigrams", "add one", "vocabulary size", and "probability mass" on task T2, and "inverse probability" and "surprise" on task T3. Sections 9 and 11 show what the reply looks like when some of those ideas are absent.

## 14. Add a lab

Use this when the notebook is a new assignment and it still has the same shape: task headings, code cells, markdown, and saved outputs.

1. Make a folder `data/synthetic_v2/references/<lab id>/`. The folder name is the id you will send.
2. Put the five files from section 13 in that folder. Write them from the question sheet for that lab. A student notebook is not the card.
3. Stop the server and start it again, or call `load_asset()` again. The loader reads every folder under `data/synthetic_v2/references/`.
4. Call `evaluate_notebook(notebook_bytes, "<lab id>", asset)` or POST that id.

You do not refit the models to score one notebook of the same shape. The fitted files turn the comparison numbers into marks. The card supplies what "correct" means for that lab.

If the new lab has no saved output, no task headings, or a different mark ceiling, say so in `docs/limitations.md` before treating the number as a mark for that lab. The ceilings in a new `rubric.json` document the lab. The served marks in this package still come from the fitted models, which were built for ceilings of 8, 8, and 4.

## 15. Feedback sentences

The feedback is assembled from the missing-idea list. No external model writes it.

When every output matches and at least one idea is missing, the first sentence is:

`The saved output matches. The write-up does not cover the reference ideas named below.`

Then one sentence per task, for example:

`The write-up for T3 does not cover: inverse probability, surprise.`

When every output matches and the idea list is empty:

`The saved output matches, and the write-up covers the reference ideas this check looks for.`

When the output does not fully match, the missing-idea sentences are still included, and the opening sentence about a saved-output match is left off. Section 11 is that case.

When the encoder is loaded, the missing-idea list is the semantic one. An idea can be accepted because the write-up says the same thing in other words. When the encoder is off, the list is a word match against the card. This package loads the encoder, so the examples above use the semantic list.

## 16. Confidence

Two frozen models score the explanation. `week3_combined` is the mark you receive. `week2_tfidf` is loaded only so the two explanation marks can be compared.

Code and documentation use the same inputs in both models, so a comparison there would always be perfect. Those two confidences are `null` on purpose. Overall confidence copies the explanation confidence. It is not an average of three numbers.

## 17. Check that your copy works

From the package directory, with the encoder and the asset folder present:

```bash
python -m pytest -q tests/test_week6_asset.py
```

That check uses a TF-IDF notebook already in the package. It confirms the server address is `127.0.0.1`, the call returns marks, and an unknown lab id is rejected.

## 18. Optional Grok grader

The marks in the sections above stay as they are. This call adds a separate `llm` object. It does not replace `code`, `explanation`, `documentation`, or `overall`.

The client library is the OpenAI Python package, with `base_url` set to `https://api.x.ai/v1` and the model set to `grok-4.20-0309-non-reasoning` by default (set `NB_EVAL_LLM_MODEL` to change it). The evaluation of this grader is `docs/llm_evaluation.md`. The key is an xAI API key from the xAI console. That package speaks the OpenAI request shape, so it can call a vendor that offers the same shape. xAI does. Anthropic's own API uses a different request shape, so this setting does not send the notebook to Claude. A SuperGrok subscription does not pay for these calls. Put the key in a file named `.env` in the package directory. That file is gitignored.

```text
XAI_API_KEY=your-key-here
NB_EVAL_LLM=1
```

Leave `NB_EVAL_LLM` unset when you do not want the extra call. With the switch off, no notebook text leaves this machine.

The Grok grader scores the student notebook against the reference notebook stored for that lab id, `reference_solution.ipynb`, together with the written explanation and the expected output in the same card folder. The prompt tells the model that this reference is the only standard. It is told not to add requirements from its own knowledge of the subject. The two cards in this package were prepared for the project. A faculty reference notebook replaces `reference_solution.ipynb` for that lab. The student's code is still not executed.

A lab with no reference notebook does not call the API. The `llm` object then says there is no standard to apply. A missing key or a failed call leaves the Ridge marks in place and records the failure on `llm.status`.

## 19. Limits

- The server binds to this machine. Do not put it on an open address if the notebooks are student work.
- A kernel is not started. A notebook with no saved output cannot show a passing output match.
- `results/week6/reports/` holds older per-notebook files from the fit. Those files are not the response this manual describes. Use the function or the HTTP call.
- The two included cards were prepared for this project. The marks follow that rubric.
- `docs/limitations.md` records further limits of the fitted models for the group that maintains this package.
