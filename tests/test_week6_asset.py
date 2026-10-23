"""Week 6 asset checks. These tests do not treat rule labels as a target."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import HOST, app
from src.asset import evaluate_notebook, load_asset
from src.models import llm_evaluator

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "data" / "synthetic_v2" / "raw" / "LAB_SYN1" / "V2_S01_LAB_SYN1.ipynb"


@pytest.fixture(scope="module")
def asset():
    return load_asset()


def test_server_binds_localhost_only():
    assert HOST == "127.0.0.1"


def test_asset_does_not_import_a_kernel():
    for name in ("src/asset.py", "src/api/app.py", "src/reporting.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        assert "nbclient" not in source
        assert "ExecutePreprocessor" not in source


def test_response_ranges_and_feedback(asset, monkeypatch):
    monkeypatch.setattr(llm_evaluator, "ENV_FILE", ROOT / "missing.env")
    monkeypatch.delenv("NB_EVAL_LLM", raising=False)
    report = evaluate_notebook(NOTEBOOK.read_bytes(), "LAB_SYN1", asset)
    assert report["assignment_id"] == "LAB_SYN1"
    assert report["scoring_variant"] in {"week3_combined", "week2_tfidf"}
    assert 0 <= report["code"] <= 8
    assert 0 <= report["explanation"] <= 8
    assert 0 <= report["documentation"] <= 4
    assert 0 <= report["overall"] <= 20
    assert report["max_score"] == 20
    assert abs((report["code"] + report["explanation"] + report["documentation"]) - report["overall"]) <= 0.02
    assert isinstance(report["feedback"], str) and report["feedback"]
    assert report["confidence"]["code"] is None
    assert report["confidence"]["documentation"] is None
    assert 0 <= report["confidence"]["explanation"] <= 1
    assert report["confidence"]["overall"] == report["confidence"]["explanation"]
    assert "feature lists differ" in report["confidence"]["method"]
    assert "not a calibrated probability" in report["confidence"]["method"]
    assert report["data_status"].startswith("SYNTHETIC")
    assert isinstance(report["flag_output_ok_explanation_weak"], bool)
    assert "llm" not in report


def test_unknown_assignment(asset):
    with pytest.raises(ValueError, match="Unknown assignment_id"):
        evaluate_notebook(NOTEBOOK.read_bytes(), "NO_SUCH_LAB", asset)


def test_http_evaluate_and_reject(monkeypatch):
    monkeypatch.setattr(llm_evaluator, "ENV_FILE", ROOT / "missing.env")
    monkeypatch.delenv("NB_EVAL_LLM", raising=False)
    with TestClient(app) as client:
        _http_checks(client)


def _http_checks(client):
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["scoring_variant"]
    assert body["data_status"].startswith("SYNTHETIC")
    with NOTEBOOK.open("rb") as handle:
        ok = client.post(
            "/evaluate-notebook",
            files={"file": ("lab.ipynb", handle, "application/json")},
            data={"assignment_id": "LAB_SYN1"},
        )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["assignment_id"] == "LAB_SYN1"
    assert "feedback" in payload and "confidence" in payload
    with NOTEBOOK.open("rb") as handle:
        bad = client.post(
            "/evaluate-notebook",
            files={"file": ("lab.ipynb", handle, "application/json")},
            data={"assignment_id": "NO_SUCH_LAB"},
        )
    assert bad.status_code == 400
