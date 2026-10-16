"""Load per-assignment reference material.

Expected layout (identical for synthetic and real data):
  <references_dir>/<ASSIGNMENT_ID>/
      template.ipynb                  blank handout given to students
      reference_solution.ipynb        instructor/team solution, same task headers
      reference_explanations.json     {"tasks": {"T2": {"text": str, "concepts": [{"term", "variants"}]}}}
      expected_outputs.json           {"tasks": {"T1": {"value": str, "match": "numeric|contains|exact", "tol": float}}}
      rubric.json                     {"code": 8, "explanation": 8, "documentation": 4}
Missing optional files degrade gracefully (the corresponding features become NaN/0 and are logged).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..preprocessing.align_tasks import assign_tasks
from ..preprocessing.code_checks import ast_summary, required_constructs
from ..preprocessing.parse_notebook import parse_notebook_file


@dataclass
class AssignmentReference:
    assignment_id: str
    template_cells: list[dict] = field(default_factory=list)
    explanations: dict = field(default_factory=dict)       # task -> {"text", "concepts"}
    expected_outputs: dict = field(default_factory=dict)   # task -> spec
    required: dict = field(default_factory=dict)           # task -> set of call names
    rubric: dict = field(default_factory=lambda: {"code": 8, "explanation": 8, "documentation": 4})
    warnings: list[str] = field(default_factory=list)


def _load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def load_reference(ref_root, assignment_id: str) -> AssignmentReference:
    d = Path(ref_root) / assignment_id
    ref = AssignmentReference(assignment_id)
    if (d / "template.ipynb").exists():
        _, ref.template_cells = parse_notebook_file(d / "template.ipynb", f"{assignment_id}_template")
    else:
        ref.warnings.append("no template.ipynb: prompt text cannot be removed from student cells")

    if (d / "reference_solution.ipynb").exists():
        _, sol = parse_notebook_file(d / "reference_solution.ipynb", f"{assignment_id}_ref")
        sol = assign_tasks(sol)
        tmpl_code = {c["source"] for c in ref.template_cells if c["cell_type"] == "code"}
        # helper names the reference defines itself (e.g. p_add_one) are not required:
        # students may legitimately name their helpers differently
        own = set()
        for c in sol:
            if c["cell_type"] == "code":
                s = ast_summary(c["source"])
                own |= set(s["functions"]) | set(s["classes"])
        for c in sol:
            if c["cell_type"] == "code" and c["task_id"] and c["source"] not in tmpl_code:
                ref.required.setdefault(c["task_id"], set()).update(required_constructs(c["source"]) - own)
    else:
        ref.warnings.append("no reference_solution.ipynb: construct coverage unavailable")

    ex = _load_json(d / "reference_explanations.json")
    ref.explanations = (ex or {}).get("tasks", {})
    if not ex:
        ref.warnings.append("no reference_explanations.json: similarity/coverage unavailable")
    eo = _load_json(d / "expected_outputs.json")
    ref.expected_outputs = (eo or {}).get("tasks", {})
    if not eo:
        ref.warnings.append("no expected_outputs.json: output comparison unavailable")
    rb = _load_json(d / "rubric.json")
    if rb:
        ref.rubric = {k: rb[k] for k in ("code", "explanation", "documentation") if k in rb}
    return ref
