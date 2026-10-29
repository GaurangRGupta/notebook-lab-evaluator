"""Cell-to-task alignment and template (instructor prompt) removal."""
from __future__ import annotations

import re

from .text_clean import similarity

TASK_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*(?:task|q|question|exercise)\s*[-_ ]?(\d+)",
                            re.IGNORECASE | re.MULTILINE)


def task_of_header(md: str) -> str | None:
    m = TASK_HEADER_RE.search(md or "")
    return f"T{int(m.group(1))}" if m else None


def assign_tasks(cells: list[dict]) -> list[dict]:
    """Every cell after a '## Task N' header belongs to TN until the next header.
    Cells before the first header get task_id None (preamble). Unaligned cells are
    left as None rather than guessed."""
    current = None
    for c in cells:
        if c["cell_type"] == "markdown":
            t = task_of_header(c["source"])
            if t:
                current = t
        c["task_id"] = current
    return cells


def flag_template_cells(cells: list[dict], template_cells: list[dict], threshold: float = 0.9) -> list[dict]:
    """Mark cells that reproduce the instructor's template (prompt text, given code).
    These are excluded from student features, otherwise prompt text would inflate
    similarity to the reference explanation."""
    by_type = {"markdown": [], "code": []}
    for t in template_cells:
        if t["cell_type"] in by_type and t["source"].strip():
            by_type[t["cell_type"]].append(t["source"])
    for c in cells:
        cands = by_type.get(c["cell_type"], [])
        best = max((similarity(c["source"], t) for t in cands), default=0.0)
        c["template_similarity"] = round(best, 3)
        c["is_template"] = best >= threshold
    return cells


# ---------------------------------------------------------------- loose alignment (class data)
# Real class notebooks rarely carry '## Task N' headers. Students write '# Q1' as the first
# line of a code cell, or nothing at all. Loose alignment is opted into per lab in rubric.json
# ("alignment": {"mode": "loose", ...}). Synthetic labs keep the strict rule above.
LOOSE_Q_RE = re.compile(r"^\s*#*\s*(?:task|que|ques|question|q|exp|exercise)\s*[-_.:)]?\s*(\d+)\b",
                        re.IGNORECASE)


def _first_line(src: str) -> str:
    for line in (src or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def assign_tasks_loose(cells: list[dict], solution_cells: list[dict], valid_tasks: set[str],
                       markers: list | None = None, generic_q: bool = True,
                       min_similarity: float = 0.3,
                       expected_outputs: dict | None = None) -> tuple[list[dict], dict]:
    """Align cells to tasks with three rules, in this order:

    1. A lab-specific marker (regex searched in the first line of any cell) sets the task.
    2. A generic 'Q1' / '# Q.2' / 'Question 3' first line sets the task, if that task exists.
    3a. An unmarked code cell whose saved output matches exactly one task's expected output (or,
        when several match, the first such task not yet claimed) takes that task.
    3. An unmarked code cell is compared with each reference-solution code cell (difflib ratio).
       If the best ratio is at least ``min_similarity`` the cell takes that task.
    Otherwise the cell inherits the current task (cells after '# Q1' belong to Q1).
    4. Code cells that are still unaligned go to the nearest reference task ("nearest").
    Returns (cells, diagnostics). Diagnostics count how each cell was aligned."""
    compiled = [(re.compile(p, re.IGNORECASE), t) for p, t in (markers or [])]
    ref_code = [(c["source"], c["task_id"]) for c in solution_cells
                if c["cell_type"] == "code" and c.get("task_id") and c["source"].strip()]
    stats = {"marker": 0, "q_header": 0, "output": 0, "similarity": 0, "inherited": 0, "nearest": 0,
             "unaligned": 0}
    from .code_checks import compare_output
    order = sorted(valid_tasks, key=lambda t: int(t[1:]) if t[1:].isdigit() else 0)
    claimed: set[str] = set()
    current = None
    for c in cells:
        line = _first_line(c["source"])
        how = None
        for rx, task in compiled:
            if rx.search(line):
                current, how = task, "marker"
                break
        if how is None and generic_q:
            m = LOOSE_Q_RE.match(line)
            if m and f"T{int(m.group(1))}" in valid_tasks:
                current, how = f"T{int(m.group(1))}", "q_header"
        if how is None and c["cell_type"] == "code" and expected_outputs and (c.get("output_text") or "").strip():
            hits = [t for t in order if t in expected_outputs
                    and compare_output(c["output_text"], expected_outputs[t])]
            if hits:
                free = [t for t in hits if t not in claimed]
                current, how = (free or hits)[0], "output"
                claimed.add(current)
        if how is None and c["cell_type"] == "code" and c["source"].strip() and ref_code:
            best, task = max((similarity(c["source"], s), t) for s, t in ref_code)
            if best >= min_similarity:
                current, how = task, "similarity"
        c["task_id"] = current
        if c["source"].strip():
            stats[how or ("inherited" if current else "unaligned")] += 1
    # 4. Code cells still without a task (e.g. before any marker, below the similarity bar) go to
    #    the nearest reference task. Counted separately: this is the weakest evidence.
    if ref_code:
        for c in cells:
            if c["task_id"] is None and c["cell_type"] == "code" and c["source"].strip():
                c["task_id"] = max((similarity(c["source"], s), t) for s, t in ref_code)[1]
                stats["unaligned"] -= 1
                stats["nearest"] += 1
    return cells, stats


def align_for_reference(cells: list[dict], ref) -> tuple[list[dict], dict]:
    """Strict alignment unless the lab's rubric.json opts into loose alignment."""
    al = getattr(ref, "alignment", None) or {}
    if al.get("mode") == "loose":
        valid = {c["task_id"] for c in ref.solution_cells if c.get("task_id")}
        cells, stats = assign_tasks_loose(cells, ref.solution_cells, valid, al.get("markers"),
                                          al.get("generic_q", True), al.get("min_similarity", 0.3),
                                          getattr(ref, "expected_outputs", None))
        return cells, {"method": "loose", **stats}
    cells = assign_tasks(cells)
    n = sum(1 for c in cells if c["source"].strip())
    a = sum(1 for c in cells if c["source"].strip() and c.get("task_id"))
    return cells, {"method": "strict", "aligned": a, "unaligned": n - a}
