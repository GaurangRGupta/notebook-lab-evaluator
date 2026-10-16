"""Planted-vs-observed consistency check for the synthetic v2 set: every planted code variant
must produce the behaviour it is supposed to produce when actually executed."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd

from ..preprocessing.align_tasks import assign_tasks
from ..preprocessing.code_checks import compare_output
from ..preprocessing.parse_notebook import parse_cells

EXPECT = {"correct_a": "output matches, no error", "correct_b": "output matches, no error",
          "unexecuted": "cell not executed", "syntax": "error output", "runtime": "error output",
          "wrong": "no error (output may or may not match)"}


def check(data_dir) -> pd.DataFrame:
    data_dir = Path(data_dir)
    d = pd.read_csv(data_dir / "design.csv")
    rows = []
    for _, r in d.iterrows():
        exp = json.loads((data_dir / "references" / r.assignment_id / "expected_outputs.json").read_text())["tasks"]
        cells = assign_tasks(parse_cells(nbformat.read(str(data_dir / r.raw_path), 4), "x"))
        for t in ("T1", "T2", "T3"):
            cc = [c for c in cells if c["cell_type"] == "code" and c["task_id"] == t]
            out = "\n".join(c["output_text"] for c in cc)
            err = any(c["has_error"] for c in cc)
            ran = all(c["execution_count"] for c in cc)
            v = r[f"code_{t}"]
            match = compare_output(out, exp[t])
            ok = {"correct_a": match and not err, "correct_b": match and not err, "unexecuted": not ran,
                  "syntax": err, "runtime": err, "wrong": not err}[v]
            rows.append({"raw_path": r.raw_path, "task": t, "planted": v, "expected": EXPECT[v],
                         "observed_output_match": match, "observed_error": err, "observed_executed": ran,
                         "consistent": ok, "wrong_but_output_matches": v == "wrong" and match})
    return pd.DataFrame(rows)
