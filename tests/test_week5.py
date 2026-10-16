"""Week 5 tests: real (ONNX) MiniLM encoder, metric definitions, invalid input handling,
synthetic v2 design consistency, split integrity, and score ranges."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import regression_metrics, round_half_up  # noqa: E402

ONNX = ROOT / "models" / "all-MiniLM-L6-v2-onnx"


@pytest.fixture(scope="module")
def onnx_sem():
    if not (ONNX / "onnx" / "model.onnx").exists():
        pytest.skip("run scripts/fetch_models.py first")
    from src.models.semantic_similarity import OnnxSentenceEncoder, SemanticSimilarity
    return SemanticSimilarity(OnnxSentenceEncoder(str(ONNX)))


# ------------------------------------------------------------------ real encoder
def test_onnx_reproduces_published_reference(onnx_sem):
    from src.models.semantic_similarity import verify_reference
    assert verify_reference(onnx_sem.encoder)["passed"]


def test_onnx_shape_norm_determinism(onnx_sem):
    a = onnx_sem.encoder.encode(["Smoothing avoids zero probabilities.", ""])
    b = onnx_sem.encoder.encode(["Smoothing avoids zero probabilities.", ""])
    assert a.shape == (2, 384) and np.allclose(np.linalg.norm(a[0]), 1, atol=1e-5) and not a[1].any()
    assert np.array_equal(a, b)


def test_onnx_identical_unrelated_and_empty(onnx_sem):
    ref = "Viterbi decoding uses dynamic programming to find the most likely sequence."
    para = "The algorithm finds the best state sequence by storing the optimal previous state at every step."
    assert onnx_sem.similarity(ref, ref) == pytest.approx(1.0, abs=1e-5)
    assert onnx_sem.similarity(ref, para) > onnx_sem.similarity(ref, "I cooked pasta for dinner.")
    assert onnx_sem.similarity(ref, "") == 0.0


def test_onnx_long_text_chunked(onnx_sem):
    long = " ".join(["Add-one smoothing adds one to every bigram count."] * 60)
    assert onnx_sem.encoder.n_tokens(long) > 256 and len(onnx_sem.encoder._chunks(long)) > 1
    assert np.isfinite(onnx_sem.encoder.encode([long])).all()


# ------------------------------------------------------------------ metric definitions
def test_round_half_up_and_agreement_definition():
    assert list(round_half_up([2.5, 3.5, 2.49])) == [3.0, 4.0, 2.0]
    m = regression_metrics([3, 3], [2.5, 4.5], 8)       # rounded preds 3, 5
    assert m["exact"] == 0.5 and m["within_1"] == 0.5


def test_metrics_match_hand_calculation():
    y, p = np.array([8, 2, 5, 0]), np.array([7.0, 3.0, 5.0, 2.0])
    m = regression_metrics(y, p, 8)
    assert m["MAE"] == pytest.approx(1.0) and m["RMSE"] == pytest.approx(np.sqrt(6 / 4))
    from scipy.stats import pearsonr
    assert m["pearson_r"] == pytest.approx(pearsonr(y, p)[0])
    assert m["within_1"] == pytest.approx(0.75)


# ------------------------------------------------------------------ data / split integrity
def test_v2_planted_design_matches_execution():
    from src.evaluation.design_check import check
    df = check(ROOT / "data/synthetic_v2")
    assert len(df) == 180 and df.consistent.all()


def test_v2_labels_follow_rules():
    d = pd.read_csv(ROOT / "data/synthetic_v2/design.csv")
    pts = {"correct_a": 1, "correct_b": 1, "unexecuted": .5, "syntax": 0, "runtime": 0, "wrong": 0}
    ep = {"correct": 4, "paraphrase": 4, "partial": 2, "vague": 1, "wrong": 0, "offtopic": 0, "empty": 0}
    for _, r in d.iterrows():
        assert r.label_code == int(8 * sum(pts[r[f"code_{t}"]] for t in ("T1", "T2", "T3")) / 3 + 0.5)
        assert r.label_explanation == ep[r.expl_T2] + ep[r.expl_T3]
        assert 0 <= r.label_documentation <= 4


RUN = ROOT / "results/week5/v2_run"


@pytest.mark.skipif(not (RUN / "split.json").exists(), reason="run scripts/run_week5.py first")
def test_v2_split_is_notebook_level_grouped_and_fixed():
    sp = json.loads((RUN / "split.json").read_text())
    man = pd.read_csv(ROOT / "data/synthetic_v2/manifest.csv")
    ids = pd.read_csv(ROOT / "data/synthetic_v2/interim/id_map.csv").merge(man, on="raw_path")
    ids["split"] = ids.notebook_id.map(sp["assignment"])
    assert ids.groupby("group_key").split.nunique().max() == 1      # no student crosses splits
    assert ids.split.value_counts().to_dict() == {"train": 36, "val": 12, "test": 12}
    cells = pd.read_csv(RUN / "cells.csv")
    assert cells.groupby("notebook_id").split.nunique().max() == 1  # cells follow their notebook


@pytest.mark.skipif(not (RUN / "predictions.csv").exists(), reason="run scripts/run_week5.py first")
def test_predictions_finite_and_in_range():
    p = pd.read_csv(RUN / "predictions.csv")
    mx = {"code": 8, "explanation": 8, "documentation": 4}
    assert np.isfinite(p.predicted).all()
    for d, m in mx.items():
        assert p[p.dimension == d].predicted.between(0, m).all()


def test_invalid_notebook_skipped_not_crash(tmp_path):
    import shutil
    from src.pipeline import Pipeline
    data = tmp_path / "data"
    shutil.copytree(ROOT / "data/synthetic", data)
    (data / "raw" / "broken.ipynb").write_text("{not json")
    man = pd.read_csv(data / "manifest.csv")
    pd.concat([man, pd.DataFrame([{**man.iloc[0].to_dict(), "raw_path": "raw/broken.ipynb"}])]).to_csv(data / "manifest.csv", index=False)
    run = Pipeline(ROOT / "configs/synthetic_dev.yaml", overrides={"data_dir": str(data), "interim_dir": str(tmp_path / "interim"),
                                                                    "results_dir": str(tmp_path / "out")}).run()
    assert len(run.skipped) == 1 and len(run.manifest) == 10
