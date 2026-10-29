"""The Grok grader is optional and judges only the stored reference notebook."""
import json

import pytest

from src.data.references import AssignmentReference
from src.models import llm_evaluator
from src.models.llm_evaluator import LlmGrade, MissingTask, build_messages, maybe_grade


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_evaluator, "ENV_FILE", tmp_path / "no.env")


def _ref() -> AssignmentReference:
    ref = AssignmentReference("LAB_SYN2")
    ref.solution_cells = [
        {
            "task_id": "T1",
            "cell_type": "code",
            "source": "count_the_cat = 2  # reference-only-line",
            "output_text": "count(the, cat) = 2",
        }
    ]
    ref.explanations = {
        "T1": {
            "text": "Count the word after the.",
            "concepts": [{"term": "add one"}],
        }
    }
    ref.expected_outputs = {"T1": {"value": "count(the, cat) = 2", "match": "numeric", "tol": 0.001}}
    return ref


def _cells() -> list[dict]:
    return [
        {
            "task_id": "T1",
            "cell_type": "markdown",
            "source": "I counted the word.",
            "output_text": "",
        }
    ]


def test_prompt_uses_the_reference_notebook_as_the_standard():
    messages = build_messages(_cells(), _ref(), ["output mismatch in T1"])
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "only marking standard is the reference material" in system
    assert "Do not add requirements from your own knowledge" in system
    assert "reference-only-line" in user
    assert "Reference ideas: add one" in user
    assert "count(the, cat) = 2" in user
    assert "output mismatch in T1" in user


def test_switch_off_does_not_call(monkeypatch):
    monkeypatch.delenv("NB_EVAL_LLM", raising=False)

    def complete(_messages):
        raise AssertionError("the switch is off")

    assert maybe_grade(_cells(), _ref(), [], complete=complete) is None


def test_missing_key(monkeypatch):
    monkeypatch.setenv("NB_EVAL_LLM", "1")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    report = maybe_grade(_cells(), _ref(), [], complete=lambda _messages: pytest.fail("called"))
    assert report["status"] == "missing_key"
    assert "XAI_API_KEY" in report["detail"]


def test_no_reference_notebook(monkeypatch):
    monkeypatch.setenv("NB_EVAL_LLM", "1")
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    empty = AssignmentReference("LAB_SYN2")
    report = maybe_grade(_cells(), empty, [], complete=lambda _messages: pytest.fail("called"))
    assert report["status"] == "no_reference"


def test_cache_reuses_the_parsed_grade(tmp_path, monkeypatch):
    monkeypatch.setenv("NB_EVAL_LLM", "1")
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    calls = {"n": 0}

    def complete(_messages):
        calls["n"] += 1
        return LlmGrade(
            code=7.87,
            explanation=5.13,
            documentation=2.14,
            feedback="Task T1 write-up misses add one from the reference.",
            missing=[MissingTask(task="T1", ideas=["add one"])],
        )

    first = maybe_grade(_cells(), _ref(), [], complete=complete, cache_dir=tmp_path)
    second = maybe_grade(_cells(), _ref(), [], complete=complete, cache_dir=tmp_path)
    assert calls["n"] == 1
    assert first["status"] == "ok"
    assert first["model"] == llm_evaluator.MODEL
    assert first["reference"] == "reference_solution.ipynb"
    assert first["overall"] == 15.14
    assert first["missing_concepts"] == {"T1": ["add one"]}
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["overall"] == first["overall"]
    saved = list(tmp_path.glob("*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text(encoding="utf-8"))["feedback"] == first["feedback"]


def test_call_failure_keeps_a_status(monkeypatch, tmp_path):
    monkeypatch.setenv("NB_EVAL_LLM", "1")
    monkeypatch.setenv("XAI_API_KEY", "xai-secret-value")

    def complete(_messages):
        raise RuntimeError("credit exhausted for xai-secret-value")

    report = maybe_grade(_cells(), _ref(), [], complete=complete, cache_dir=tmp_path)
    assert report["status"] == "error"
    assert "xai-secret-value" not in report["detail"]
    assert "credit exhausted" in report["detail"]
