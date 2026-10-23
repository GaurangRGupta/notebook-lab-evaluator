"""Week 7 demo page and the student-support handoff."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.integration import support_signal
from src.models import llm_evaluator

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SAVED = ROOT / "results" / "week6"


def _saved(name: str) -> dict:
    return json.loads((SAVED / name).read_text(encoding="utf-8"))


def test_support_signal_on_the_three_examples():
    gap = support_signal(_saved("example_language_model.json"))
    assert gap["source"] == "notebook-lab-evaluator"
    assert gap["assignment_id"] == "LAB_SYN2"
    assert gap["overall"] == 15.14
    assert gap["max_score"] == 20
    assert gap["needs_human_review"] is True
    assert "inverse probability" in gap["reason"]

    covered = support_signal(_saved("example_tfidf.json"))
    assert covered["assignment_id"] == "LAB_SYN1"
    assert covered["overall"] == 18.17
    assert covered["needs_human_review"] is False

    broken = support_signal(_saved("example_language_model_weak.json"))
    assert broken["overall"] == 7.28
    assert broken["needs_human_review"] is True


def test_missing_output_match_asks_for_review():
    signal = support_signal({"assignment_id": "LAB_SYN1", "overall": 10, "max_score": 20})
    assert signal["needs_human_review"] is True
    assert signal["reason"] == ""


def test_support_signal_rejects_a_non_object():
    with pytest.raises(TypeError):
        support_signal(["not", "a", "report"])


def test_demo_page_and_support_route(monkeypatch):
    monkeypatch.setattr(llm_evaluator, "ENV_FILE", ROOT / "missing.env")
    monkeypatch.delenv("NB_EVAL_LLM", raising=False)
    notebook = (EXAMPLES / "language_model_lab.ipynb").read_bytes()
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]
        text = page.text
        assert "LAB_SYN1" in text and "LAB_SYN2" in text
        assert "POST /evaluate-notebook" in text
        assert "POST /support-signal" in text
        assert "The student's code is not executed." in text

        empty = client.post(
            "/evaluate-notebook",
            files={"file": ("empty.ipynb", b"", "application/json")},
            data={"assignment_id": "LAB_SYN2"},
        )
        assert empty.status_code == 400
        assert empty.json()["detail"] == "The upload is empty."

        scored = client.post(
            "/evaluate-notebook",
            files={"file": ("language_model_lab.ipynb", notebook, "application/json")},
            data={"assignment_id": "LAB_SYN2"},
        )
        assert scored.status_code == 200
        report = scored.json()
        assert "llm" not in report
        assert report["overall"] == 15.14
        assert report["flag_output_ok_explanation_weak"] is True

        signal = client.post("/support-signal", json=report)
        assert signal.status_code == 200
        body = signal.json()
        assert body["needs_human_review"] is True
        assert body["source"] == "notebook-lab-evaluator"
        assert body["overall"] == 15.14
