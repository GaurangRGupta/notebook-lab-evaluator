"""Mark scale, feedback text, and confidence for one notebook.

Pipeline.reports and the Week 6 asset both call this module.
The sentences and the numeric rules live here only.
"""
from __future__ import annotations

import math

DIMS = ("code", "explanation", "documentation")

AGREEMENT_METHOD = (
    "Agreement between week2_tfidf and week3_combined is computed only where their feature lists differ. "
    "This is not a calibrated probability."
)

DIAGNOSTIC_METHOD = (
    "Diagnostics only: syntax errors, unexecuted cells, output mismatch, and empty explanation. "
    "The Week 3 encoder was not loaded, so agreement between models is not available. "
    "This is not a calibrated probability."
)

_MISSING_VARIANTS_METHOD = (
    "Agreement between week2_tfidf and week3_combined is computed only where their feature lists differ. "
    "Those two variants were not both fitted. This is not a calibrated probability."
)


def _finite(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def marks(predictions: dict, rubric_max: dict) -> dict:
    """Code, explanation, and documentation as marks. Overall is round(sum of the raw marks, 2)."""
    raw = {dim: float(predictions[dim]) for dim in DIMS}
    scored = {dim: round(raw[dim], 2) for dim in DIMS}
    scored["overall"] = round(sum(raw.values()), 2)
    scored["max_score"] = int(sum(float(value) for value in rubric_max.values()))
    return scored


def filled_ideas(missing: dict | None) -> dict:
    if not missing:
        return {}
    return {
        str(task): [str(item) for item in ideas]
        for task, ideas in missing.items()
        if ideas
    }


def choose_missing(lexical, semantic, encoder_loaded: bool) -> dict:
    """Use the semantic list when the encoder ran and that list was computed."""
    if encoder_loaded and semantic is not None:
        return filled_ideas(semantic)
    return filled_ideas(lexical)


def feedback_and_flag(missing: dict, output_match_rate) -> tuple[str, bool]:
    """Flag is true when every saved output matches and the chosen idea list is not empty."""
    rate = _finite(output_match_rate)
    full_match = rate == 1.0
    chosen = filled_ideas(missing)
    flag = bool(full_match and chosen)
    sentences = []
    if flag:
        sentences.append(
            "The saved output matches. The write-up does not cover the reference ideas named below."
        )
    for task in sorted(chosen):
        ideas = ", ".join(chosen[task])
        sentences.append(f"The write-up for {task} does not cover: {ideas}.")
    if not sentences:
        if full_match:
            return (
                "The saved output matches, and the write-up covers the reference ideas this check looks for.",
                False,
            )
        return "The write-up covers the reference ideas this check looks for.", False
    return " ".join(sentences), flag


def agreement_value(left: float, right: float, mark_max: float) -> float:
    gap = abs(float(left) - float(right)) / float(mark_max)
    return round(1.0 - min(1.0, gap), 3)


def agreement_confidence(left_preds, right_preds, rubric_max, left_features, right_features) -> dict:
    """Null on a dimension whose two feature lists are equal. Overall follows explanation."""
    out = {}
    for dim in DIMS:
        same = list(left_features[dim]) == list(right_features[dim])
        if same:
            out[dim] = None
        else:
            out[dim] = agreement_value(left_preds[dim], right_preds[dim], rubric_max[dim])
    out["overall"] = out["explanation"]
    out["method"] = AGREEMENT_METHOD
    return out


def diagnostic_confidence(features: dict, diag: dict) -> dict:
    """Used only when the Week 3 encoder was not loaded."""
    code_penalty = 0.0
    if diag.get("errors"):
        code_penalty += 0.4
    if diag.get("unexecuted_cells"):
        code_penalty += 0.3
    if diag.get("output_mismatch"):
        code_penalty += 0.3
    coverage = _finite(features.get("keyword_coverage"))
    if coverage is None:
        coverage = 0.0
    explained = _finite(features.get("frac_tasks_explained"))
    if explained is None:
        explained = 0.0
    explanation = 0.0 if explained == 0 else round(coverage, 3)
    documentation = 0.5 if explained == 0 else 1.0
    code = round(max(0.0, 1.0 - min(1.0, code_penalty)), 3)
    overall = round((code + explanation + documentation) / 3.0, 3)
    return {
        "code": code,
        "explanation": explanation,
        "documentation": documentation,
        "overall": overall,
        "method": DIAGNOSTIC_METHOD,
    }


def confidence_from_variants(
    predictions_by_variant: dict,
    rubric_max: dict,
    left_features: dict,
    right_features: dict,
    features: dict,
    diag: dict,
    encoder_loaded: bool,
) -> dict:
    left = (predictions_by_variant or {}).get("week2_tfidf")
    right = (predictions_by_variant or {}).get("week3_combined")
    if encoder_loaded and left is not None and right is not None:
        return agreement_confidence(left, right, rubric_max, left_features, right_features)
    if not encoder_loaded:
        return diagnostic_confidence(features, diag)
    return {
        "code": None,
        "explanation": None,
        "documentation": None,
        "overall": None,
        "method": _MISSING_VARIANTS_METHOD,
    }
