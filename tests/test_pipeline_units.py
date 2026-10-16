"""Unit tests. Every fixture here is synthetic and built in-code."""
import math
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.split import check_no_group_leakage, grouped_split, merge_near_duplicates  # noqa: E402
from src.evaluation.iaa import aggregate_labels, icc_2_1, krippendorff_alpha  # noqa: E402
from src.evaluation.metrics import regression_metrics  # noqa: E402
from src.preprocessing import code_checks as cc  # noqa: E402
from src.preprocessing.align_tasks import assign_tasks, flag_template_cells  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.preprocessing.anonymize import Scrubber  # noqa: E402
from src.preprocessing.parse_notebook import parse_cells  # noqa: E402
from src.preprocessing.text_clean import concept_coverage, markdown_to_text, tokenize  # noqa: E402


# ------------------------------------------------------------ anonymisation
def test_scrubber_removes_roll_email_name_and_path():
    s = Scrubber(names=["Fake Student Alpha"])
    t = s.scrub("Fake Student Alpha (J099) fake.a@example.com File /Users/fakeuser/lab.py")
    assert "J099" not in t and "example.com" not in t and "Alpha" not in t and "fakeuser" not in t
    assert "[ROLL]" in t and "[EMAIL]" in t and "[NAME]" in t and "/Users/[USER]" in t


def test_placeholder_salt_stops_before_scoring(monkeypatch):
    monkeypatch.delenv("NB_EVAL_SALT", raising=False)
    pipe = Pipeline(ROOT / "configs" / "class_data.yaml")
    with pytest.raises(SystemExit, match="NB_EVAL_SALT"):
        pipe.anonymise()
    assert not hasattr(pipe, "df")
    assert not hasattr(pipe, "models")


def test_scrub_notebook_drops_metadata_and_scrubs_outputs():
    nb = nbformat.v4.new_notebook()
    nb.metadata["authors"] = [{"name": "Fake Person"}]
    c = nbformat.v4.new_code_cell("print('J077')")
    c.outputs = [nbformat.v4.new_output("stream", name="stdout", text="J077\n")]
    c.metadata["collapsed"] = True
    nb.cells = [c]
    out = Scrubber().scrub_notebook(nb)
    assert "authors" not in out.metadata and out.cells[0].metadata == {}
    assert "J077" not in nbformat.writes(out)


# ------------------------------------------------------------ parsing
def _nb_with_error():
    nb = nbformat.v4.new_notebook()
    c = nbformat.v4.new_code_cell("1/0", execution_count=1)
    c.outputs = [nbformat.v4.new_output("error", ename="ZeroDivisionError", evalue="division by zero",
                                        traceback=["\x1b[31mZeroDivisionError\x1b[0m"])]
    img = nbformat.v4.new_code_cell("plot()", execution_count=2)
    img.outputs = [nbformat.v4.new_output("display_data", data={"image/png": "AAAA", "text/plain": "<Figure>"})]
    nb.cells = [nbformat.v4.new_markdown_cell("## Task 1: x"), c, img]
    return nb


def test_parse_cells_errors_images_ansi():
    cells = parse_cells(_nb_with_error(), "nb")
    assert cells[1]["has_error"] and cells[1]["error_name"] == "ZeroDivisionError"
    assert "\x1b" not in cells[1]["traceback"]
    assert cells[2]["has_image"] and "AAAA" not in cells[2]["output_text"]


# ------------------------------------------------------------ code checks
def test_magics_do_not_break_ast():
    ok, _ = cc.syntax_check("%matplotlib inline\n!pip install x\nx = 1")
    assert ok


def test_syntax_error_detected():
    ok, msg = cc.syntax_check("for i in range(3)\n    print(i)")
    assert not ok and "SyntaxError" in msg


def test_comment_hash_inside_string_not_counted():
    src = 's = "a # not a comment"\n# real comment\nx = 1  # inline'
    assert cc.extract_comments(src) == ["real comment", "inline"]


def test_comments_fallback_on_broken_code():
    assert cc.extract_comments("# note\nfor x in y\n  pass") == ["note"]


def test_ast_summary_docstrings_and_calls():
    s = cc.ast_summary('def f(x):\n    """doc"""\n    return np.log(x)\nf(1)')
    assert s["n_docstrings"] == 1 and {"log", "f"} <= s["calls"]


def test_output_comparison_modes():
    assert cc.compare_output("P(cat | the) = 0.3000", {"value": "0.3", "match": "numeric", "tol": 1e-3})
    assert not cc.compare_output("0.6667", {"value": "0.3", "match": "numeric", "tol": 1e-3})
    assert cc.compare_output("(3, 12)", {"value": "shape: (3, 12)", "match": "numeric"})
    assert not cc.compare_output("", {"value": "12", "match": "numeric"})
    assert cc.compare_output("Obj at 0x7f00AB", {"value": "obj at 0x1234", "match": "exact"})
    assert cc.classify_error("ModuleNotFoundError") == "env" and cc.classify_error("NameError") == "code"


# ------------------------------------------------------------ text
def test_markdown_cleaning_keeps_technical_terms():
    t = markdown_to_text("## Heading\n**TF-IDF** uses `word2vec`? See [docs](http://x.y) ![img](a.png) $P(w|c)$ <b>bi-lstm</b>")
    toks = tokenize(t)
    assert {"tf-idf", "word2vec", "bi-lstm", "docs"} <= set(toks)
    assert "http" not in t and "img" not in t


def test_concept_coverage_variants_and_phrases():
    concepts = [{"term": "inverse document frequency", "variants": ["idf"]},
                {"term": "logarithm", "variants": ["log"]}]
    score, hit, miss = concept_coverage("The IDF term is large for rare words.", concepts)
    assert score == 0.5 and hit == ["inverse document frequency"] and miss == ["logarithm"]
    # 'log' must not match inside 'login'
    assert concept_coverage("login page", concepts)[0] == 0.0


# ------------------------------------------------------------ alignment / template
def test_task_alignment_and_template_flagging():
    cells = [{"cell_type": "markdown", "source": "# Title"},
             {"cell_type": "markdown", "source": "## Task 1: Do X"},
             {"cell_type": "code", "source": "x = 1"},
             {"cell_type": "markdown", "source": "### Q2 explain"},
             {"cell_type": "markdown", "source": "My answer"}]
    cells = assign_tasks(cells)
    assert [c["task_id"] for c in cells] == [None, "T1", "T1", "T2", "T2"]
    tmpl = [{"cell_type": "markdown", "source": "## Task 1: Do X"}, {"cell_type": "markdown", "source": "### Q2 explain"}]
    cells = flag_template_cells(cells, tmpl)
    assert [c["is_template"] for c in cells] == [False, True, False, True, False]


# ------------------------------------------------------------ split
def test_grouped_split_no_leakage_and_reports_realised():
    ids = [f"n{i}" for i in range(40)]
    groups = [f"s{i // 2}" for i in range(40)]            # 20 students x 2 labs
    strata = ["L1" if i % 2 else "L2" for i in range(40)]
    out = grouped_split(ids, groups, strata, (0.6, 0.2, 0.2), seed=1)
    check_no_group_leakage(out["assignment"], dict(zip(ids, groups)))
    assert set(out["realised"]) == {"train", "val", "test"}
    assert abs(out["realised"]["test"] - 0.2) < 0.15


def test_leakage_check_raises():
    with pytest.raises(AssertionError):
        check_no_group_leakage({"a": "train", "b": "test"}, {"a": "g", "b": "g"})


def test_near_duplicates_merged_into_one_group():
    texts = ["def f(x): return x * 2  # double it", "def f(x): return x * 2  # double it", "totally different text"]
    g = merge_near_duplicates(["a", "b", "c"], texts, ["s1", "s2", "s3"], 0.95)
    assert g[0] == g[1] != g[2]


# ------------------------------------------------------------ metrics / IAA
def test_regression_metrics_known_values():
    m = regression_metrics([0, 4, 8], [1, 4, 6], 8)
    assert math.isclose(m["MAE"], 1.0) and math.isclose(m["RMSE"], math.sqrt(5 / 3))
    assert m["within_1"] == pytest.approx(2 / 3) and m["exact"] == pytest.approx(1 / 3)


def test_correlation_undefined_for_tiny_or_constant():
    assert regression_metrics([1, 2], [1, 2], 8)["pearson_r"] is None
    assert regression_metrics([1, 1, 1], [1, 2, 3], 8)["pearson_r"] is None


def test_iaa_perfect_agreement():
    m = np.array([[1, 1, 1], [3, 3, 3], [5, 5, 5]], float)
    assert krippendorff_alpha(m, "interval") == pytest.approx(1.0)
    assert krippendorff_alpha(m, "ordinal") == pytest.approx(1.0)
    assert icc_2_1(m) == pytest.approx(1.0)


def test_krippendorff_known_example():
    # Krippendorff (2011) interval example, 4 coders x 12 units with missing values: alpha ~= 0.849
    d = np.array([
        [1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
        [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan]]).T
    assert krippendorff_alpha(d, "interval") == pytest.approx(0.849, abs=0.002)
    assert krippendorff_alpha(d, "ordinal") == pytest.approx(0.815, abs=0.002)


def test_aggregation_flags_and_adjudication():
    df = pd.DataFrame({"notebook_id": ["a"] * 3 + ["b"] * 3, "annotator": ["A1", "A2", "A3"] * 2,
                       "code": [4, 5, 4, 2, 6, 4]})
    out = aggregate_labels(df, ["code"], tolerance=1)
    assert out.set_index("notebook_id").loc["b", "code_needs_discussion"]
    adj = pd.DataFrame({"notebook_id": ["b"], "code": [5]})
    out2 = aggregate_labels(df, ["code"], tolerance=1, adjudicated=adj).set_index("notebook_id")
    assert out2.loc["b", "code"] == 5 and out2.loc["b", "code_source"] == "adjudicated"
