"""Week 4 tests: frozen baseline, NLI component, estimator selection, variants, robustness,
error analysis. Stub scorers are LEXICAL test stand-ins; real-model tests skip when unavailable."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.features import FEATURE_VARIANTS, VARIANT_LABELS, variant_features  # noqa: E402
from src.models.nli_consistency import NLI_FEATURES, NLIConsistency, StubNLI, label_order  # noqa: E402
from src.evaluation.robustness import EXPECTED, make_variants, run_robustness  # noqa: E402

REF = "Perplexity is the inverse probability of the test text. A lower perplexity means a better language model."


# ------------------------------------------------------------------ frozen baseline
def test_week2_baseline_frozen(tmp_path):
    from src.pipeline import Pipeline
    run = Pipeline(ROOT / "configs" / "synthetic_dev.yaml", overrides={"results_dir": str(tmp_path)}).run()
    frozen = pd.read_csv(ROOT / "results/frozen_week2_baseline/predictions.csv")
    fresh = pd.read_csv(run.out / "predictions.csv")
    keys = ["variant", "notebook_id", "split", "dimension"]
    assert list(frozen.columns) == list(fresh.columns)
    merged = frozen.merge(fresh, on=keys, suffixes=("_frozen", "_fresh"))
    assert len(merged) == len(frozen) == len(fresh)
    assert (merged.human_frozen - merged.human_fresh).abs().max() == 0
    # Same fit. A fresh BLAS run can move the last bit of a float.
    assert (merged.predicted_frozen - merged.predicted_fresh).abs().max() < 1e-9


# ------------------------------------------------------------------ NLI component (stub)
def test_label_order_reads_model_config():
    assert label_order({0: "contradiction", 1: "entailment", 2: "neutral"}) == [0, 1, 2]
    assert label_order({"0": "ENTAILMENT", "1": "NEUTRAL", "2": "CONTRADICTION"}) == [2, 0, 1]
    with pytest.raises(ValueError):
        label_order({0: "positive", 1: "negative"})


def test_nli_probabilities_shape_and_sum():
    p = StubNLI().predict([(REF, "A lower perplexity is better."), (REF, "x")])
    assert p.shape == (2, 3) and np.allclose(p.sum(axis=1), 1.0)


def test_nli_empty_explanation_scores_zero():
    s = NLIConsistency(StubNLI()).task_scores("", REF)
    assert s["support"] == s["coverage"] == s["contradiction"] == 0.0 and s["worst_sentence"] is None


def test_nli_flags_contradiction_and_features():
    nli = NLIConsistency(StubNLI(), flag_threshold=0.5)
    good = nli.task_scores("A lower perplexity means a better model.", REF)
    bad = nli.task_scores("A higher perplexity means a better model.", REF)
    assert bad["contradiction"] > good["contradiction"]

    class R:
        explanations = {"T3": {"text": REF}}
    f, d = nli.notebook_features({"T3": "A higher perplexity means a better model."}, R())
    assert set(f) == set(NLI_FEATURES) and d["flags"] and "T3" in d["flags"][0]


# ------------------------------------------------------------------ variants
def test_week4_variant_is_week3_plus_nli_only():
    w3, w4 = variant_features("week3_combined"), variant_features("week4_improved")
    assert w4["code"] == w3["code"] and w4["documentation"] == w3["documentation"]
    assert w4["explanation"] == w3["explanation"] + NLI_FEATURES
    assert VARIANT_LABELS["week2_tfidf"] == "Week 2 Baseline" and VARIANT_LABELS["week4_improved"] == "Week 4 Improved Model"
    assert set(VARIANT_LABELS) == set(FEATURE_VARIANTS)


# ------------------------------------------------------------------ pipeline (stubs)
@pytest.fixture(scope="module")
def run4(tmp_path_factory):
    from src.pipeline import Pipeline
    out = tmp_path_factory.mktemp("w4stub")
    return Pipeline(ROOT / "configs" / "synthetic_dev_week4.yaml",
                    overrides={"semantic": {"encoder": "stub"}, "nli": {"encoder": "stub"}, "results_dir": str(out)}).run()


def test_week4_pipeline_outputs(run4):
    met = json.loads((run4.out / "metrics.json").read_text())
    assert set(met["variants"]) == {"week2_tfidf", "week3_semantic", "week3_combined", "week4_improved",
                                    "abl_nli_only", "abl_week4_plus_code"}
    assert met["primary_variant"] == "week4_improved" and "STUB" in met["WARNING"]
    for v in met["variants"].values():
        assert {"val", "test"} <= set(v["holdout_overall"]) and v["grouped_cv_overall"]["model"]["n"] == 8
    rep = json.loads(next((run4.out / "reports").glob("*.json")).read_text())
    assert rep["nli"]["is_stub"] and "possible_conceptual_errors" in rep["nli"]
    for f in ("robustness.csv", "robustness_summary.csv", "error_analysis.csv", "probe_results.csv"):
        assert (run4.out / f).exists()


def test_week2_variant_identical_inside_week4_run(run4):
    frozen = pd.read_csv(ROOT / "results/frozen_week2_baseline/predictions.csv")
    p = pd.read_csv(run4.out / "predictions.csv")
    p = p[p.variant == "week2_tfidf"].reset_index(drop=True)
    assert np.allclose(p.predicted.to_numpy(), frozen.predicted.to_numpy())


def test_estimator_selection_uses_train_val_only(run4):
    assert set(run4.dev.split) <= {"train", "val"}
    sel = json.loads((run4.out / "metrics.json").read_text())["estimator_selection"]
    for d, r in sel.items():
        assert r["chosen"] in ("ridge", "random_forest") and set(r["cv_MAE"]) == {"ridge", "random_forest"}
        expected = "random_forest" if r["relative_gain_of_forest"] >= 0.10 else "ridge"
        assert r["chosen"] == expected


def test_selection_rule_extremes(run4):
    from src.models.improved import select_estimators
    feats = variant_features("week4_improved")
    never = select_estimators(run4.dev, feats, ["explanation"], run4.cfg["rubric_max"], min_relative_gain=10.0)
    always = select_estimators(run4.dev, feats, ["explanation"], run4.cfg["rubric_max"], min_relative_gain=-10.0)
    assert never["explanation"]["chosen"] == "ridge" and always["explanation"]["chosen"] == "random_forest"


def test_forest_model_deterministic_and_clipped(run4):
    from src.models.improved import ForestDimensionModel
    f = variant_features("week4_improved")["explanation"]
    a = ForestDimensionModel("explanation", f, 8).fit(run4.dev, run4.dev.explanation.to_numpy(float)).predict(run4.dev)
    b = ForestDimensionModel("explanation", f, 8).fit(run4.dev, run4.dev.explanation.to_numpy(float)).predict(run4.dev)
    assert np.array_equal(a, b) and a.min() >= 0 and a.max() <= 8


def test_error_analysis_categories(run4):
    ea = pd.read_csv(run4.out / "error_analysis.csv")
    assert set(ea.category) <= {"both succeed", "both fail", "baseline fails, improved succeeds",
                                "baseline succeeds, improved fails"}
    assert set(ea["eval"]) >= {"cv_train_val", "val", "test"}


# ------------------------------------------------------------------ robustness
def test_robustness_variants_and_tfidf_behaviour():
    df = run_robustness(ROOT / "data/synthetic/week4/robustness_cases.json")
    assert set(df.variant) == set(EXPECTED) and len(df) == 5 * len(EXPECTED)
    assert (df[df.variant == "empty"].tfidf_cos == 0).all()
    assert (df[df.variant == "reference_copy"].tfidf_cos == 1.0).all()


def test_scrambled_variant_keeps_vocabulary():
    v = make_variants({"correct": "a b c d e f", "incorrect_similar": "x", "reference": "r"}, "irr")
    assert sorted(v["words_scrambled"].split()) == sorted("a b c d e f".split())


# ------------------------------------------------------------------ real NLI model (skips if unavailable)
@pytest.fixture(scope="module")
def real_nli():
    pytest.importorskip("sentence_transformers")
    from src.models.nli_consistency import CrossEncoderNLI
    try:
        return NLIConsistency(CrossEncoderNLI(local_path=os.environ.get("NB_EVAL_NLI_PATH")))
    except Exception as e:
        pytest.skip(f"NLI model not available here: {str(e)[:80]}")


def test_real_nli_negation_raises_contradiction(real_nli):
    good = real_nli.task_scores("A lower perplexity means a better language model.", REF)
    bad = real_nli.task_scores("A higher perplexity means a better language model.", REF)
    assert bad["contradiction"] > good["contradiction"]
    assert good["support"] > bad["support"]
