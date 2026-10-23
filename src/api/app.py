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
from fastapi.responses import HTMLResponse

from ..asset import evaluate_notebook, load_asset
from ..integration import support_signal

HOST = "127.0.0.1"
PORT = 8000
_DEMO_TEMPLATE = Path(__file__).with_name("demo.html").read_text(encoding="utf-8")


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
