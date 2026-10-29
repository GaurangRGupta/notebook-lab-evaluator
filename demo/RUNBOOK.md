# Live demo runbook (2-3 minutes, end of the presentation)

## Before the talk (Windows, in the project folder)

```bat
.venv\Scripts\activate
python scripts\prepare_demo_samples.py   REM optional: class samples + refreshes demo\fallback\
python -m src.api.app
```

Open `http://127.0.0.1:8000/` in the browser. Keep a second terminal open in the project folder.
Check: `curl.exe -s http://127.0.0.1:8000/health` prints `"status":"ok"`.

## Script

1. **Upload a notebook (~45 s).** Click *Choose File*, pick `examples\language_model_lab.ipynb`, Lab id `LAB_SYN2`, *Score notebook*.
   Point at: overall **15.14 / 20**, code / explanation / documentation, output match, the missing concepts for T3 ("inverse probability"), the feedback line, and the support signal (`needs_human_review: true`).
   Optional: pick the class sample *binomial / Poisson [MU_L07_08]* and *Score sample*: it shows the "out-of-domain, unvalidated" and "team-authored reference" badges and an output mismatch in T2.
2. **Call it from another app (~30 s).** In the second terminal:
   ```bat
   curl.exe -s -X POST http://127.0.0.1:8000/evaluate-notebook -F "file=@examples/tfidf_lab.ipynb" -F "assignment_id=LAB_SYN1"
   ```
   Overall is **18.17**. The page's *Call it from another app* panel shows the same call in curl, Python and PowerShell.
3. **Failure case (~45 s).** Pick the sample *FAILURE CASE: hard-coded prints, no logic [MU_L01]*, *Score sample*.
   It has three `print` lines with the right numbers and no computation, and still gets code **7.87 / 8** and overall **10.89**, higher than a real class solution (9.31). Say why: the code is not executed and output matching cannot tell hard-coded from computed output; that is why the mark is a suggestion and the support signal asks a person to review.
4. Close (~15 s): the marks are suggestions; class predictions are unvalidated until the marking sheet is filled (`python scripts\eval_class.py`).

## If something breaks

Saved JSON from the same build is in `demo\fallback\` (`<sample>.report.json` and `<sample>.support.json`), plus page captures in `demo\check\` (`*.html`, `*.png`). Open them instead of the live call.
- Server will not start: run `python scripts\fetch_models.py` then `python scripts\fit_week6_asset.py`.
- Port busy: stop the other process, or run `python -c "import uvicorn; from src.api.app import app; uvicorn.run(app, host='127.0.0.1', port=8001)"`.
- Unknown lab id gives a 400 with the list of known ids (`demo\check\curl_unknown_lab.json`).
