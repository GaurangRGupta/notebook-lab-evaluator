"""Week 3 tests for the semantic similarity component.

Two groups:
  * STUB tests run everywhere. They use HashingStubEncoder (lexical, NOT semantic) to test the
    plumbing: shapes, normalisation, empty/identical text, chunking, features, pipeline wiring.
  * REAL-MODEL tests load all-MiniLM-L6-v2 and are SKIPPED automatically if the model cannot be
    loaded (no internet and no cache). Set NB_EVAL_MODEL_PATH to a saved copy to run them offline.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.references import AssignmentReference  # noqa: E402
from src.evaluation.semantic_eval import calibrate_concept_threshold, load_concept_pairs  # noqa: E402
from src.models.features import DEFAULT_FEATURES, FEATURE_VARIANTS, variant_features  # noqa: E402
from src.models.semantic_similarity import (HashingStubEncoder, SemanticSimilarity, cosine,  # noqa: E402
                                            split_sentences)

REF = "Viterbi decoding uses dynamic programming to find the most likely sequence."
PARA = "The algorithm finds the best state sequence by storing the optimal previous state at every step."
UNREL = "I cooked pasta with tomato sauce for dinner yesterday."


@pytest.fixture
def stub():
    return SemanticSimilarity(HashingStubEncoder(), concept_threshold=0.5)


# ------------------------------------------------------------------ embeddings (stub)
def test_embedding_shape_norm_and_determinism(stub):
    e1 = stub.encoder.encode([REF, PARA])
    e2 = HashingStubEncoder().encode([REF, PARA])
    assert e1.shape == (2, stub.encoder.dim)
    assert np.allclose(np.linalg.norm(e1, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(e1, e2)


def test_empty_text_gives_zero_vector_and_zero_similarity(stub):
    e = stub.encoder.encode(["", "   ", REF])
    assert not e[0].any() and not e[1].any()
    assert stub.similarity("", REF) == 0.0 and stub.similarity(REF, "  ") == 0.0


def test_identical_text_similarity_is_one(stub):
    assert stub.similarity(REF, REF) == pytest.approx(1.0, abs=1e-5)


def test_unrelated_text_lower_than_identical(stub):
    assert stub.similarity(REF, UNREL) < stub.similarity(REF, REF)


def test_cosine_handles_zero_vectors_and_clips():
    assert cosine(np.zeros(3), np.ones(3)) == 0.0
    v = np.array([1.0, 0.0])
    assert cosine(v, v * 1.0000001) <= 1.0
    assert cosine(v, -v) == pytest.approx(-1.0)


def test_long_text_is_chunked_not_truncated():
    enc = HashingStubEncoder()
    enc.max_tokens = 12
    long = " ".join(f"Sentence number {i} talks about smoothing." for i in range(10))
    assert len(enc._chunks(long)) > 1
    v = enc.encode([long])[0]
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


def test_sentence_split():
    assert split_sentences("One. Two!\nThree; four") == ["One.", "Two!", "Three;", "four"]


# ------------------------------------------------------------------ features (stub)
def _ref():
    r = AssignmentReference("X")
    r.explanations = {"T2": {"text": REF, "concepts": [
        {"term": "dynamic programming", "variants": []},
        {"term": "backtracking", "variants": ["backpointer"]}]},
        "T3": {"text": "Perplexity: lower is better.", "concepts": []}}
    return r


def test_notebook_features_and_missing_explanation(stub):
    f, d = stub.notebook_features({"T2": PARA, "T3": ""}, _ref())
    assert set(f) == {"sem_cos_mean", "sem_cos_max", "sem_concept_coverage"}
    assert d["per_task_similarity"]["T3"] == 0.0            # unanswered task -> 0
    assert f["sem_cos_max"] >= f["sem_cos_mean"]


def test_semantic_concept_coverage_extends_lexical(stub):
    concepts = _ref().explanations["T2"]["concepts"]
    text = "It uses dynamic programming. It follows saved pointers back to recover the path."
    stub.concept_threshold = 1.01                            # semantic matching disabled
    c = stub.concept_coverage(text, concepts)
    assert c["lexical"] == ["dynamic programming"] and c["semantic_added"] == []
    stub.concept_threshold = -1.0                            # everything passes semantically
    c = stub.concept_coverage(text, concepts)
    assert c["score"] == 1.0 and c["semantic_added"] == ["backtracking"]
    assert stub.concept_coverage("", concepts)["score"] == 0.0


def test_features_are_stateless_no_split_dependence(stub):
    ref = _ref()
    a, _ = stub.notebook_features({"T2": PARA, "T3": "x"}, ref)
    other = SemanticSimilarity(HashingStubEncoder())         # fresh instance, different history
    other.notebook_features({"T2": UNREL, "T3": "y"}, ref)
    b, _ = other.notebook_features({"T2": PARA, "T3": "x"}, ref)
    assert a == b


def test_calibration_uses_disjoint_concepts(stub):
    pairs = load_concept_pairs(ROOT / "data/synthetic/week3/concept_pairs.csv")
    assert not set(pairs[pairs.split == "tune"].concept_term) & set(pairs[pairs.split == "eval"].concept_term)
    cal = calibrate_concept_threshold(stub, pairs)
    assert 0.2 <= cal["threshold"] <= 0.8
    assert set(cal["eval"]) == {"lexical_only", "semantic_only", "lexical_or_semantic"}
    bad = pairs.copy()
    bad.loc[bad.index[0], "split"] = "eval"
    with pytest.raises(ValueError):
        calibrate_concept_threshold(stub, bad)


# ------------------------------------------------------------------ variants
def test_variants_only_change_explanation_features():
    assert variant_features("week2_tfidf") == DEFAULT_FEATURES
    for v in FEATURE_VARIANTS:
        f = variant_features(v)
        assert f["code"] == DEFAULT_FEATURES["code"] and f["documentation"] == DEFAULT_FEATURES["documentation"]
    assert not any(x.startswith("tfidf") for x in variant_features("week3_semantic")["explanation"])


# ------------------------------------------------------------------ pipeline integration (stub)
@pytest.fixture(scope="module")
def stub_run(tmp_path_factory):
    from src.pipeline import Pipeline
    out = tmp_path_factory.mktemp("w3stub")
    return Pipeline(ROOT / "configs" / "synthetic_dev_week3.yaml",
                    overrides={"semantic": {"encoder": "stub"}, "results_dir": str(out)}).run()


def test_pipeline_integrates_semantic_features(stub_run):
    for c in ("sem_cos_mean", "sem_cos_max", "sem_concept_coverage", "tfidf_cos_mean"):
        assert c in stub_run.df.columns and stub_run.df[c].notna().all()
    met = json.loads((stub_run.out / "metrics.json").read_text())
    assert set(met["variants"]) == {"week2_tfidf", "week3_semantic", "week3_combined"}
    assert met["primary_variant"] == "week3_combined"
    assert "STUB SEMANTIC" in met["WARNING"] and met["semantic_model"]["is_stub"]
    rep = json.loads(next((stub_run.out / "reports").glob("*.json")).read_text())
    assert rep["semantic"]["is_stub"] and rep["scoring_variant"] == "week3_combined"


def test_week2_variant_unchanged_by_week3(stub_run):
    """The week2_tfidf variant inside the Week 3 run must equal the Week 2 run."""
    from src.pipeline import Pipeline
    w2 = Pipeline(ROOT / "configs" / "synthetic_dev.yaml",
                  overrides={"results_dir": str(stub_run.out.parent / "w2copy")}).run()
    a = stub_run.variant_results["week2_tfidf"]["holdout"]
    for d in ("code", "explanation", "documentation"):
        for s in ("val", "test"):
            assert a[d][s]["ridge"] == w2.results[d][s]["ridge"]


def test_code_and_documentation_scores_do_not_depend_on_embeddings(stub_run):
    v = stub_run.variant_results
    for d in ("code", "documentation"):
        assert v["week2_tfidf"]["holdout"][d]["val"] == v["week3_combined"]["holdout"][d]["val"]


def test_grouped_cv_excludes_test(stub_run):
    cv = stub_run.variant_results["week3_combined"]["grouped_cv_train_val"]
    n_dev = int((stub_run.manifest.split != "test").sum())
    assert cv["explanation"]["ridge"]["n"] == n_dev == 8


# ------------------------------------------------------------------ real model (skipped if unavailable)
@pytest.fixture(scope="module")
def real():
    pytest.importorskip("sentence_transformers")
    from src.models.semantic_similarity import SentenceTransformerEncoder
    try:
        enc = SentenceTransformerEncoder(local_path=os.environ.get("NB_EVAL_MODEL_PATH"))
    except Exception as e:  # no network / no cache
        pytest.skip(f"MiniLM not available here: {str(e)[:80]}")
    return SemanticSimilarity(enc)


def test_real_model_dimension_and_normalisation(real):
    e = real.encoder.encode([REF, PARA])
    assert e.shape == (2, 384)
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4)


def test_real_model_is_deterministic(real):
    assert np.allclose(real.encoder.encode([PARA]), real.encoder.encode([PARA]), atol=1e-6)


def test_real_model_identical_and_unrelated(real):
    assert real.similarity(REF, REF) == pytest.approx(1.0, abs=1e-4)
    assert real.similarity(REF, PARA) > real.similarity(REF, UNREL)
