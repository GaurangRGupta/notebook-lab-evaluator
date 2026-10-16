"""Runs the full pipeline on the SYNTHETIC fixtures and checks behaviour, not performance."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import Pipeline  # noqa: E402


@pytest.fixture(scope="module")
def run():
    return Pipeline(ROOT / "configs" / "synthetic_dev.yaml").run()


def _by_profile(run):
    f = run.df.merge(run.manifest[["notebook_id", "raw_path"]], on="notebook_id")
    f["key"] = f.raw_path.str.extract(r"(S0\d_LAB_SYN\d)")[0]
    return f.set_index("key")


def test_outputs_are_labelled_synthetic(run):
    for name in ["metrics.json", "split.json", "iaa.json", "dataset_stats.json"]:
        d = json.loads((run.out / name).read_text())
        assert d["data_status"] == "SYNTHETIC_DEVELOPMENT_DATA"
    assert "NOT how the evaluator performs" in json.loads((run.out / "metrics.json").read_text())["WARNING"]


def test_no_pii_survives_anonymisation(run):
    assert run.pii_residual == 0
    for p in run.interim.glob("*.ipynb"):
        t = p.read_text()
        assert "Fake Student" not in t and "example.com" not in t and "J099" not in t


def test_every_notebook_parsed_and_aligned(run):
    assert len(run.manifest) == 10
    student = run.cells[(~run.cells.is_template) & run.cells.task_id.notna()]
    assert set(student.task_id) == {"T1", "T2", "T3"}


def test_split_is_notebook_level_and_group_safe(run):
    m = run.manifest
    assert m.groupby("group")["split"].nunique().max() == 1
    assert run.cells.groupby("notebook_id")["split"].nunique().max() == 1


def test_code_track_detects_planted_problems(run):
    f = _by_profile(run)
    assert f.loc["S03_LAB_SYN1", "frac_syntax_ok"] < 1          # planted syntax error
    assert f.loc["S03_LAB_SYN1", "error_cell_rate"] > 0         # planted NameError
    assert f.loc["S04_LAB_SYN1", "output_match_rate"] < 1       # planted output mismatch
    assert f.loc["S04_LAB_SYN1", "construct_coverage"] < 1      # CountVectorizer instead of TfidfVectorizer
    assert f.loc["S05_LAB_SYN1", "frac_executed"] < 1           # never-executed cell
    assert f.loc["S05_LAB_SYN1", "out_of_order"] == 1
    assert f.loc["S01_LAB_SYN1", "output_match_rate"] == 1


def test_text_track_orders_explanations_sensibly(run):
    f = _by_profile(run)
    for lab in ("LAB_SYN1", "LAB_SYN2"):
        assert f.loc[f"S01_{lab}", "keyword_coverage"] > f.loc[f"S02_{lab}", "keyword_coverage"]
        assert f.loc[f"S01_{lab}", "tfidf_cos_mean"] > f.loc[f"S02_{lab}", "tfidf_cos_mean"]
        assert f.loc[f"S05_{lab}", "frac_tasks_explained"] == 0


def test_documentation_features(run):
    f = _by_profile(run)
    assert f.loc["S02_LAB_SYN2", "docstring_rate"] == 1 and f.loc["S05_LAB_SYN2", "comment_density"] == 0


def test_tfidf_fit_on_train_only(run):
    # the vectoriser vocabulary must not contain words that only occur in val/test explanations
    from src.models.features import TfidfSimilarity
    tr = run.manifest[run.manifest.split == "train"].notebook_id
    t = TfidfSimilarity().fit([run.expl[n] for n in tr], run.refs)
    assert "popular" not in t.vec.vocabulary_  # appears only in a held-out notebook (S02_LAB_SYN1)


def test_predictions_within_rubric_range(run):
    p = pd.read_csv(run.out / "predictions.csv")
    for d, mx in run.cfg["rubric_max"].items():
        q = p[p.dimension == d].predicted
        assert q.between(0, mx).all()
