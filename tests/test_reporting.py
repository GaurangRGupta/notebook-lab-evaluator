"""Shared report rules. These tests do not fit a model."""
from src.models.features import variant_features
from src.reporting import (
    agreement_confidence,
    choose_missing,
    diagnostic_confidence,
    feedback_and_flag,
    marks,
)

RUBRIC = {"code": 8, "explanation": 8, "documentation": 4}


def test_marks_share_one_scale_and_sum_to_overall():
    scored = marks(
        {"code": 6.0, "explanation": 4.5, "documentation": 2.25},
        RUBRIC,
    )
    report = {
        "code_correctness": scored["code"],
        "explanation_score": scored["explanation"],
        "documentation_quality": scored["documentation"],
        "overall_score": scored["overall"],
        "max_score": scored["max_score"],
    }
    assert report["code_correctness"] == 6.0
    assert report["explanation_score"] == 4.5
    assert report["documentation_quality"] == 2.25
    assert report["overall_score"] == 12.75
    assert (
        report["code_correctness"]
        + report["explanation_score"]
        + report["documentation_quality"]
        == report["overall_score"]
    )
    assert report["max_score"] == 20
    assert report["code_correctness"] == 6.0


def test_feedback_names_the_semantic_list_when_the_encoder_ran():
    lexical = {"T3": ["direction", "cosine"]}
    semantic = {"T3": ["direction"]}
    chosen = choose_missing(lexical, semantic, encoder_loaded=True)
    text, flag = feedback_and_flag(chosen, 1.0)
    assert chosen == {"T3": ["direction"]}
    assert "direction" in text
    assert "cosine" not in text
    assert flag is True


def test_feedback_names_the_lexical_list_when_the_encoder_is_off():
    lexical = {"T1": ["perplexity"]}
    semantic = {"T3": ["direction"]}
    chosen = choose_missing(lexical, semantic, encoder_loaded=False)
    text, flag = feedback_and_flag(chosen, 1.0)
    assert "perplexity" in text
    assert "direction" not in text
    assert flag is True


def test_feedback_flag_needs_a_full_output_match():
    missing = choose_missing({"T1": ["perplexity"]}, None, encoder_loaded=False)
    text, flag = feedback_and_flag(missing, 0.5)
    assert flag is False
    assert "perplexity" in text
    assert "The saved output matches." not in text


def test_confidence_is_null_where_feature_lists_match():
    left = variant_features("week2_tfidf")
    right = variant_features("week3_combined")
    assert left["code"] == right["code"]
    assert left["documentation"] == right["documentation"]
    assert left["explanation"] != right["explanation"]
    conf = agreement_confidence(
        {"code": 6.0, "explanation": 4.0, "documentation": 2.0},
        {"code": 7.5, "explanation": 5.0, "documentation": 3.5},
        RUBRIC,
        left,
        right,
    )
    assert conf["code"] is None
    assert conf["documentation"] is None
    assert conf["explanation"] == 0.875
    assert 0 <= conf["explanation"] <= 1
    assert conf["overall"] == conf["explanation"]
    assert "feature lists differ" in conf["method"]
    assert "not a calibrated probability" in conf["method"]


def test_diagnostic_confidence_when_the_encoder_is_off():
    conf = diagnostic_confidence(
        {"keyword_coverage": 0.5, "frac_tasks_explained": 1.0},
        {"errors": ["syntax"], "unexecuted_cells": [], "output_mismatch": []},
    )
    assert conf["code"] == 0.6
    assert conf["explanation"] == 0.5
    assert conf["documentation"] == 1.0
    assert conf["overall"] == 0.7
    assert "Diagnostics only" in conf["method"]
    assert "not a calibrated probability" in conf["method"]
