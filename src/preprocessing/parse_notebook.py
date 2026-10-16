"""Parse a .ipynb file into notebook- and cell-level records with nbformat.

Retained per cell (and why):
  cell_type, source          -> the content we evaluate
  execution_count            -> whether/in what order the cell was run
  output_text                -> stream + text/plain results, for output comparison
  output_types               -> stream / execute_result / display_data / error
  has_error, error_name/value, traceback -> runtime error detection
  has_image                  -> a plot was produced (bytes are NOT kept)
  n_lines, n_chars           -> length features
Dropped: widget state, image bytes, HTML renderings, timing, cell metadata.
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
MAX_OUTPUT_CHARS = 2000


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def _join(v) -> str:
    return "".join(v) if isinstance(v, list) else (v or "")


def read_notebook(path) -> nbformat.NotebookNode:
    nb = nbformat.read(str(path), as_version=4)
    nbformat.validate(nb)
    return nb


def parse_cells(nb: nbformat.NotebookNode, notebook_id: str) -> list[dict]:
    records = []
    for idx, cell in enumerate(nb.cells):
        src = cell.source or ""
        rec = {
            "notebook_id": notebook_id,
            "cell_index": idx,
            "cell_type": cell.cell_type,
            "source": src,
            "n_lines": len([l for l in src.splitlines() if l.strip()]),
            "n_chars": len(src),
            "execution_count": None,
            "output_types": "",
            "output_text": "",
            "has_error": False,
            "error_name": None,
            "error_value": None,
            "traceback": None,
            "has_image": False,
        }
        if cell.cell_type == "code":
            rec["execution_count"] = cell.get("execution_count")
            texts, types = [], []
            for out in cell.get("outputs", []):
                otype = out.get("output_type")
                types.append(otype)
                if otype == "stream":
                    texts.append(_join(out.get("text")))
                elif otype in ("execute_result", "display_data"):
                    data = out.get("data", {})
                    if "text/plain" in data:
                        texts.append(_join(data["text/plain"]))
                    if any(k.startswith("image/") for k in data):
                        rec["has_image"] = True
                elif otype == "error":
                    rec["has_error"] = True
                    rec["error_name"] = out.get("ename")
                    rec["error_value"] = strip_ansi(out.get("evalue", ""))
                    rec["traceback"] = strip_ansi("\n".join(out.get("traceback", [])))[:MAX_OUTPUT_CHARS]
            rec["output_types"] = ",".join(types)
            rec["output_text"] = strip_ansi("\n".join(t.rstrip("\n") for t in texts))[:MAX_OUTPUT_CHARS]
        records.append(rec)
    return records


def parse_notebook_file(path, notebook_id: str | None = None) -> tuple[dict, list[dict]]:
    path = Path(path)
    nb = read_notebook(path)
    nid = notebook_id or path.stem
    cells = parse_cells(nb, nid)
    meta = {
        "notebook_id": nid,
        "nbformat": f"{nb.nbformat}.{nb.nbformat_minor}",
        "language": nb.metadata.get("language", nb.metadata.get("kernelspec", {}).get("language", "python")),
        "n_cells": len(cells),
        "n_code_cells": sum(c["cell_type"] == "code" for c in cells),
        "n_markdown_cells": sum(c["cell_type"] == "markdown" for c in cells),
        "synthetic_development_data": bool(nb.metadata.get("synthetic_development_data", False)),
    }
    return meta, cells
