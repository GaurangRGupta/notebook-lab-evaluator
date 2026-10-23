# Notebook lab evaluator API

Group 3. Rolls J013, J015, J016, J017, J018.

The manual is `docs/manual.md`. Use that page for the field list, the three worked examples, the answer-card files, and how to add a lab. This page is only the call.

Send one `.ipynb` file and a lab id. The reply is one JSON object. The student's code is not executed. Do not send the answer card with the call.

`LAB_SYN2` is the language-model lab. `LAB_SYN1` is the TF-IDF lab.

From the package directory, when `results/week6/asset/` and `models/all-MiniLM-L6-v2-onnx/` are present:

```bash
python -m venv .venv
pip install -r requirements.txt
python -m src.api.app
```

The server listens on `127.0.0.1:8000` only.

```python
from src.asset import evaluate_notebook, load_asset

asset = load_asset()
notebook = open("examples/language_model_lab.ipynb", "rb").read()
report = evaluate_notebook(notebook, "LAB_SYN2", asset)
```

On Windows:

```bash
curl.exe -s -X POST http://127.0.0.1:8000/evaluate-notebook ^
  -F "assignment_id=LAB_SYN2" ^
  -F "file=@examples/language_model_lab.ipynb"
```

An empty file, a file that is not a notebook, or an unknown lab id returns HTTP 400. The worked responses, including a notebook whose output does not match, are in `docs/manual.md`.
