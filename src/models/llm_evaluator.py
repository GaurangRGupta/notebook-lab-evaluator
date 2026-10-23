"""Optional Grok grader.

The OpenAI SDK is only the HTTP client. Requests go to the xAI API
(https://api.x.ai/v1) with XAI_API_KEY and the model grok-4.7.

The standard is the reference notebook stored for the lab, not the model's
own knowledge of the subject. The call is off unless NB_EVAL_LLM=1.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from pydantic import BaseModel, Field

from ..reporting import marks

ROOT = Path(__file__).resolve().parents[2]
MODEL = "grok-4.7"
BASE_URL = "https://api.x.ai/v1"
CACHE_DIR = ROOT / "results" / "week7" / "llm_cache"
ENV_FILE = ROOT / ".env"
CELL_LIMIT = 4000
OUTPUT_LIMIT = 1000
_KEY_RE = re.compile(r"xai-[A-Za-z0-9_\-]{8,}")

SYSTEM_PROMPT = (
    "You grade one student lab notebook. The only marking standard is the "
    "reference material in the user message. That material is the reference "
    "notebook for this assignment, plus the written explanation and the "
    "expected saved output for each task. "
    "Score the student by comparison with that reference. Do not add "
    "requirements from your own knowledge of the subject. Do not ignore a "
    "difference from the reference because you know another acceptable "
    "method, unless the reference itself allows it. If the reference is "
    "silent on a point, do not invent a penalty for it. "
    "Do not execute code. Read the saved source and the saved outputs only. "
    "A saved output that matches the expected line is not, by itself, "
    "evidence that the written explanation is right. "
    "Marks: code from 0 to 8, explanation from 0 to 8, documentation from "
    "0 to 4. feedback is a short paragraph that names the task and the "
    "reference idea or reference line the student missed. missing lists "
    "only ideas that are named in the reference and absent from the "
    "student write-up."
)


class MissingTask(BaseModel):
    task: str = Field(description="Task id from the reference, such as T1.")
    ideas: list[str] = Field(description="Reference ideas the write-up does not cover.")


class LlmGrade(BaseModel):
    code: float = Field(ge=0, le=8, description="Code mark from comparison with the reference notebook.")
    explanation: float = Field(ge=0, le=8, description="Explanation mark from comparison with the reference.")
    documentation: float = Field(ge=0, le=4, description="Documentation mark from comparison with the reference.")
    feedback: str = Field(description="Short paragraph naming the reference gap, if any.")
    missing: list[MissingTask] = Field(description="Reference ideas absent from the student write-up.")


def load_env_file(path: Path | None = None) -> None:
    """Load KEY=VALUE lines. An existing environment variable wins."""
    file = path if path is not None else ENV_FILE
    if not file.is_file():
        return
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def llm_enabled() -> bool:
    load_env_file()
    return os.environ.get("NB_EVAL_LLM", "").strip().lower() in {"1", "true", "yes"}


def _clip(text: str, limit: int) -> str:
    cleaned = _KEY_RE.sub("[redacted-key]", text or "")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + " [truncated]"


def _task_ids(ref, cells: list[dict]) -> list[str]:
    found = set(ref.explanations) | set(ref.expected_outputs)
    for cell in list(ref.solution_cells) + list(cells):
        task = cell.get("task_id")
        if isinstance(task, str) and task:
            found.add(task)
    return sorted(task for task in found if isinstance(task, str) and task)


def _cells_for(cells: list[dict], task: str) -> list[str]:
    lines = []
    for cell in cells:
        if cell.get("task_id") != task:
            continue
        kind = cell.get("cell_type", "cell")
        lines.append(f"[{kind}]")
        lines.append(_clip(cell.get("source", ""), CELL_LIMIT))
        output = cell.get("output_text") or ""
        if output.strip():
            lines.append("[saved output]")
            lines.append(_clip(output, OUTPUT_LIMIT))
    return lines


def build_messages(cells: list[dict], ref, errors: list[str]) -> list[dict]:
    """Prompt whose only standard is this lab's reference notebook."""
    blocks = [
        f"Lab id: {ref.assignment_id}",
        "Reference notebook (this is the standard):",
    ]
    if not ref.solution_cells:
        blocks.append("(no reference notebook was stored for this lab)")
    else:
        for task in _task_ids(ref, cells):
            chunk = _cells_for(ref.solution_cells, task)
            if not chunk:
                continue
            blocks.append(f"Reference task {task}")
            blocks.extend(chunk)
            info = ref.explanations.get(task) or {}
            if info.get("text"):
                blocks.append("Reference explanation:")
                blocks.append(_clip(info["text"], CELL_LIMIT))
            terms = [item.get("term", "") for item in info.get("concepts", []) if item.get("term")]
            if terms:
                blocks.append("Reference ideas: " + ", ".join(terms))
            spec = ref.expected_outputs.get(task) or {}
            if spec.get("value"):
                blocks.append(
                    f"Expected saved output ({spec.get('match', 'contains')}): {spec['value']}"
                )
    blocks.append("Student notebook:")
    for task in _task_ids(ref, cells):
        chunk = _cells_for(cells, task)
        if not chunk:
            continue
        blocks.append(f"Student task {task}")
        blocks.extend(chunk)
    if errors:
        blocks.append("Checks already found in the saved file:")
        blocks.extend(f"- {item}" for item in errors)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(blocks)},
    ]


def _cache_path(messages: list[dict], cache_dir: Path) -> Path:
    raw = json.dumps({"model": MODEL, "messages": messages}, ensure_ascii=True, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _public_grade(grade: LlmGrade, cached: bool) -> dict:
    code = min(8.0, max(0.0, float(grade.code)))
    explanation = min(8.0, max(0.0, float(grade.explanation)))
    documentation = min(4.0, max(0.0, float(grade.documentation)))
    scored = marks(
        {"code": code, "explanation": explanation, "documentation": documentation},
        {"code": 8, "explanation": 8, "documentation": 4},
    )
    missing = {
        item.task: [idea for idea in item.ideas if idea]
        for item in grade.missing
        if item.ideas
    }
    return {
        "status": "ok",
        "model": MODEL,
        "reference": "reference_solution.ipynb",
        "code": scored["code"],
        "explanation": scored["explanation"],
        "documentation": scored["documentation"],
        "overall": scored["overall"],
        "max_score": scored["max_score"],
        "feedback": grade.feedback,
        "missing_concepts": missing,
        "cached": cached,
    }


def openai_complete(messages: list[dict]) -> LlmGrade:
    """Call api.x.ai. The SDK package is openai. The model is Grok."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["XAI_API_KEY"], base_url=BASE_URL)
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        temperature=0,
        messages=messages,
        response_format=LlmGrade,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("The Grok response did not match the grade schema.")
    return parsed


def _safe_detail(exc: Exception) -> str:
    text = _KEY_RE.sub("[redacted-key]", str(exc))
    key = os.environ.get("XAI_API_KEY", "")
    if key:
        text = text.replace(key, "[redacted-key]")
    return text[:500]


def maybe_grade(cells: list[dict], ref, errors: list[str], complete=None, cache_dir: Path | None = None) -> dict | None:
    """Return the llm object, or None when the switch is off."""
    if not llm_enabled():
        return None
    if not ref.solution_cells:
        return {
            "status": "no_reference",
            "detail": "This lab has no reference notebook, so the Grok grader has no standard to apply.",
        }
    if not os.environ.get("XAI_API_KEY", "").strip():
        return {
            "status": "missing_key",
            "detail": "NB_EVAL_LLM is on and XAI_API_KEY is not set. The Ridge marks above are unchanged.",
        }
    messages = build_messages(cells, ref, errors)
    folder = cache_dir if cache_dir is not None else CACHE_DIR
    path = _cache_path(messages, folder)
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["cached"] = True
        return saved
    caller = complete if complete is not None else openai_complete
    try:
        grade = caller(messages)
    except Exception as exc:
        return {"status": "error", "detail": _safe_detail(exc)}
    report = _public_grade(grade, cached=False)
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
