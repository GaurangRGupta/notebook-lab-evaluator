"""Handoff from one notebook score to a student-support reader.

The reader does not score the notebook. It reads the JSON that
evaluate_notebook already returned.
"""
from __future__ import annotations


def support_signal(report: dict) -> dict:
    """Build the record a student-support tool would store.

    needs_human_review is true when the weak-explanation flag is set,
    when any error is listed, or when the saved output does not fully match.
    A missing output match is treated as a review, because the reader
    cannot see that the output was checked.
    """
    if not isinstance(report, dict):
        raise TypeError("support_signal expects the evaluator JSON object.")
    feedback = report.get("feedback")
    reason = feedback if isinstance(feedback, str) else ""
    return {
        "source": "notebook-lab-evaluator",
        "assignment_id": report.get("assignment_id"),
        "overall": report.get("overall"),
        "max_score": report.get("max_score"),
        "needs_human_review": _needs_human_review(report),
        "reason": reason,
    }


def _needs_human_review(report: dict) -> bool:
    if report.get("flag_output_ok_explanation_weak") is True:
        return True
    errors = report.get("errors")
    if isinstance(errors, list):
        if len(errors) > 0:
            return True
    elif errors:
        return True
    rate = report.get("output_match_rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        return True
    return float(rate) < 1.0
