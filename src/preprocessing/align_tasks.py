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
