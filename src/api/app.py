"""Local API for the notebook lab evaluator.

Bind this process to 127.0.0.1. Another machine is not a client of this process.
Start it only after the asset has been fitted:

    python scripts/fit_week6_asset.py
    python -m src.api.app
"""
from __future__ import annotations

import html
from contextlib import asynccontextmanager
from pathlib import Path

import nbformat
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from ..asset import evaluate_notebook, load_asset
from ..integration import support_signal

HOST = "127.0.0.1"
PORT = 8000
_DEMO_TEMPLATE = Path(__file__).with_name("demo.html").read_text(encoding="utf-8")
SAMPLES_DIR = Path(__file__).resolve().parents[2] / "demo" / "samples"
# name -> (lab id, short description). Files that are missing on disk are not listed.
SAMPLES = {
    "tfidf_lab.ipynb": ("LAB_SYN1", "synthetic: strong TF-IDF lab"),
    "language_model_lab.ipynb": ("LAB_SYN2", "synthetic: language-model lab, one concept gap"),
    "language_model_weak.ipynb": ("LAB_SYN2", "synthetic: weak lab with errors"),
    "failure_hardcoded_output_MU_L01.ipynb": ("MU_L01", "FAILURE CASE: hard-coded prints, no logic"),
    "class/MU_L06_sample.ipynb": ("MU_L06", "class (anonymised): conditional probability"),
    "class/MU_L07_08_sample.ipynb": ("MU_L07_08", "class (anonymised): binomial / Poisson"),
    "class/MU_L01_sample.ipynb": ("MU_L01", "class (anonymised): car rental billing"),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.asset = load_asset()
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc
    yield


app = FastAPI(title="Jupyter notebook lab evaluator", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def demo_page():
    ids = sorted(app.state.asset.refs)
    options = "\n".join(
        f'<option value="{html.escape(lab_id)}">{html.escape(lab_id)}</option>'
        for lab_id in ids
    )
    page = _DEMO_TEMPLATE.replace("<!-- LAB_OPTIONS -->", options)
    return HTMLResponse(page)


@app.get("/samples")
def list_samples():
    """Sample notebooks for the demo picker. Class samples exist only on machines that ran
    scripts/prepare_demo_samples.py (they are gitignored)."""
    out = []
    for name, (lab, desc) in SAMPLES.items():
        if (SAMPLES_DIR / name).is_file():
            out.append({"name": name, "assignment_id": lab, "description": desc})
    return out


@app.get("/samples/{name:path}")
def get_sample(name: str):
    if name not in SAMPLES or not (SAMPLES_DIR / name).is_file():
        raise HTTPException(status_code=404, detail="Unknown sample.")
    return Response((SAMPLES_DIR / name).read_bytes(), media_type="application/json")


@app.post("/support-signal")
def support_signal_route(report: dict):
    try:
        return support_signal(report)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health():
    asset = app.state.asset
    return {
        "status": "ok",
        "scoring_variant": asset.primary,
        "encoder": asset.encoder_name,
        "data_status": asset.data_status,
    }


@app.post("/evaluate-notebook")
async def evaluate_notebook_route(
    file: UploadFile = File(...),
    assignment_id: str = Form(...),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The upload is empty.")
    try:
        nbformat.reads(raw.decode("utf-8"), as_version=4)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"The upload is not a readable notebook: {exc}") from exc
    known = app.state.asset.refs
    if assignment_id not in known:
        names = ", ".join(sorted(known)) or "(none)"
        raise HTTPException(
            status_code=400,
            detail=f"Unknown assignment_id {assignment_id!r}. Known ids: {names}.",
        )
    try:
        return evaluate_notebook(raw, assignment_id, app.state.asset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
