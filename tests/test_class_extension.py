"""Class-dataset extension: loose alignment, regex output mode, class labs in the asset,
the LLM skip for class labs, the demo sample routes and scripts/eval_class.py."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.preprocessing.align_tasks import assign_tasks_loose
from src.preprocessing.code_checks import compare_output

ROOT = Path(__file__).resolve().parents[1]
CLASS_REFS = ROOT / "data" / "class" / "references"


def _cell(src, kind="code", out=""):
    return {"cell_type": kind, "source": src, "output_text": out, "task_id": None}


def test_regex_mode_rejects_fail_to_reject():
    spec = {"value": "(?<!fail to )reject", "match": "regex"}
    assert compare_output("Reject the null hypothesis", spec)
    assert not compare_output("We fail to reject the null hypothesis", spec)


def test_loose_alignment_rules_in_order():
    sol = [_cell("x = sum(range(10))\nprint(x)"), _cell("import numpy as np\nnp.mean([1,2])")]
    sol[0]["task_id"], sol[1]["task_id"] = "T1", "T2"
    cells = [_cell("lab two part", "markdown"), _cell("print(45)", out="45"),
             _cell("# Q2\nnp.mean([3, 4])"), _cell("y = 1")]
    cells, stats = assign_tasks_loose(cells, sol, {"T1", "T2"}, markers=[["part", "T1"]],
                                      expected_outputs={"T1": {"value": "45", "match": "contains"}})
    assert [c["task_id"] for c in cells] == ["T1", "T1", "T2", "T2"]
    assert stats["marker"] == 1 and stats["q_header"] == 1
    assert stats["unaligned"] == 0


def test_nearest_fallback_assigns_orphan_code():
    sol = [_cell("import pandas as pd\ndf = pd.read_csv('a.csv')")]
    sol[0]["task_id"] = "T1"
    cells, stats = assign_tasks_loose([_cell("zzz = 0")], sol, {"T1"}, min_similarity=0.99)
    assert cells[0]["task_id"] == "T1" and stats["nearest"] == 1


needs_class = pytest.mark.skipif(not (CLASS_REFS / "MU_L06" / "rubric.json").is_file(),
                                 reason="class references not built")


@needs_class
def test_class_lab_is_labelled_and_never_calls_llm(monkeypatch):
    from src import asset as asset_mod
    from src.models import llm_evaluator
    monkeypatch.setenv("NB_EVAL_LLM", "1")
    called = []
    monkeypatch.setattr(asset_mod, "maybe_grade", lambda *a, **k: called.append(1) or {}, raising=False)
    monkeypatch.setattr(llm_evaluator, "grade", lambda *a, **k: called.append(1) or {}, raising=False)
    a = asset_mod.load_asset()
    assert "MU_L06" in a.refs and "LAB_SYN1" in a.refs
    nb = (CLASS_REFS / "MU_L06" / "reference_solution.ipynb").read_bytes()
    rep = asset_mod.evaluate_notebook(nb, "MU_L06", a)
    assert "not faculty-issued" in rep["reference_status"]
    assert "unvalidated" in rep["prediction_status"]
    assert rep["alignment"]["method"] == "loose"
    assert set(rep["variant_marks"])
    assert "llm" not in rep and not called


@needs_class
def test_sample_routes():
    from src.api.app import app
    with TestClient(app) as client:
        names = [s["name"] for s in client.get("/samples").json()]
        assert "failure_hardcoded_output_MU_L01.ipynb" in names
        assert client.get("/samples/tfidf_lab.ipynb").status_code == 200
        assert client.get("/samples/../app.py").status_code == 404
        page = client.get("/").text
        assert "MU_L01" in page and "Call it from another app" in page


def _load_eval():
    spec = importlib.util.spec_from_file_location("eval_class", ROOT / "scripts" / "eval_class.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def test_eval_class_no_marks(tmp_path):
    ev = _load_eval()
    sheet, pred = tmp_path / "s.csv", tmp_path / "p.csv"
    _write(sheet, ["notebook_id", "status", "code_0_8", "explanation_0_8", "documentation_0_4"],
           [{"notebook_id": "a", "status": "unmarked"}])
    _write(pred, ["notebook_id", "tfidf_code"], [{"notebook_id": "a", "tfidf_code": 3}])
    assert ev.evaluate(sheet, pred)["status"] == "no human marks yet"


def test_eval_class_metrics(tmp_path):
    ev = _load_eval()
    sheet, pred = tmp_path / "s.csv", tmp_path / "p.csv"
    hs = [(2, 1, 1), (5, 3, 2), (8, 6, 4)]
    _write(sheet, ["notebook_id", "status", "team_member_author", "annotator", "code_0_8",
                   "explanation_0_8", "documentation_0_4"],
           [{"notebook_id": f"n{i}", "status": "marked", "team_member_author": "no", "annotator": "x",
             "code_0_8": c, "explanation_0_8": e, "documentation_0_4": d} for i, (c, e, d) in enumerate(hs)])
    cols = ["notebook_id"] + [f"{v}_{d}" for v in ("tfidf", "minilm")
                              for d in ("code", "explanation", "documentation", "overall")]
    rows = []
    for i, (c, e, d) in enumerate(hs):
        r = {"notebook_id": f"n{i}"}
        for v in ("tfidf", "minilm"):
            r |= {f"{v}_code": c + 1, f"{v}_explanation": e, f"{v}_documentation": d,
                  f"{v}_overall": c + e + d + 1}
        rows.append(r)
    _write(pred, cols, rows)
    res = ev.evaluate(sheet, pred)
    m = res["variants"]["tfidf"]["code"]
    assert res["n_marked"] == 3
    assert m["mae"] == 1.0 and m["rmse"] == 1.0 and m["within_1"] == 1.0
    assert m["pearson"] == 1.0 and m["spearman"] == 1.0
    e = res["variants"]["minilm"]["explanation"]
    assert e["mae"] == 0.0 and e["qwk"] == 1.0
