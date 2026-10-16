"""Notebook-level feature extraction for the Week 2 baseline.

Two kinds of features:
  * static features: computed per notebook with no fitting (code checks, lengths, coverage)
  * TF-IDF similarity: needs a vectoriser, so it is FIT ON TRAINING NOTEBOOKS ONLY
    (see TfidfSimilarity) to avoid leaking validation/test vocabulary statistics.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..preprocessing import code_checks as cc
from ..preprocessing.text_clean import (concept_coverage, is_placeholder, markdown_to_text, tokenize)

CODE_FEATURES = ["frac_syntax_ok", "frac_executed", "error_cell_rate", "out_of_order",
                 "output_match_rate", "construct_coverage"]
EXPLANATION_FEATURES = ["tfidf_cos_mean", "tfidf_cos_max", "expl_tokens_log",
                        "frac_tasks_explained", "keyword_coverage"]
DOCUMENTATION_FEATURES = ["comment_density", "docstring_rate", "md_tokens_log", "md_to_code_ratio"]
DEFAULT_FEATURES = {"code": CODE_FEATURES, "explanation": EXPLANATION_FEATURES,
                    "documentation": DOCUMENTATION_FEATURES}

# ---------------------------------------------------------------- Week 3 feature-set variants
# Only the EXPLANATION feature set changes between variants. Code correctness stays on the
# deterministic code track and documentation stays on its Week 2 features in every variant.
# (A "semantic minus TF-IDF" difference feature is deliberately NOT added: for a linear model
#  it is an exact linear combination of the two similarities and adds only collinearity.)
FEATURE_VARIANTS = {
    # A: Week 2 baseline, reproduced exactly
    "week2_tfidf": {"explanation": EXPLANATION_FEATURES},
    # B: semantic similarity + semantic concept coverage REPLACE their lexical counterparts
    "week3_semantic": {"explanation": ["sem_cos_mean", "sem_cos_max", "expl_tokens_log",
                                       "frac_tasks_explained", "sem_concept_coverage"]},
    # C (Version 1): lexical and semantic evidence side by side
    "week3_combined": {"explanation": EXPLANATION_FEATURES + ["sem_cos_mean", "sem_cos_max",
                                                              "sem_concept_coverage"]},
    # ------------------------------------------------------------ Week 4
    # Improved model = Week 3 combined + NLI consistency (entailment/contradiction vs reference)
    "week4_improved": {"explanation": EXPLANATION_FEATURES + ["sem_cos_mean", "sem_cos_max", "sem_concept_coverage",
                                                              "nli_support", "nli_coverage", "nli_contradiction"]},
    # Ablations (explanation model only)
    "abl_nli_only": {"explanation": ["expl_tokens_log", "frac_tasks_explained",
                                     "nli_support", "nli_coverage", "nli_contradiction"]},
    "abl_week4_plus_code": {"explanation": EXPLANATION_FEATURES + ["sem_cos_mean", "sem_cos_max",
                                                                   "sem_concept_coverage", "nli_support", "nli_coverage",
                                                                   "nli_contradiction", "output_match_rate",
                                                                   "construct_coverage"]},
}

VARIANT_LABELS = {
    "week2_tfidf": "Week 2 Baseline",
    "week3_semantic": "Week 3 Semantic-only (ablation)",
    "week3_combined": "Week 3 Semantic Model",
    "week4_improved": "Week 4 Improved Model",
    "abl_nli_only": "Ablation: NLI only",
    "abl_week4_plus_code": "Ablation: Week 4 + code evidence",
}


def variant_features(variant: str, overrides: dict | None = None) -> dict:
    """Full {dimension: [features]} mapping for a named variant."""
    spec = dict(FEATURE_VARIANTS[variant]) if variant in FEATURE_VARIANTS else {}
    spec.update(overrides or {})
    return {d: list(spec.get(d, DEFAULT_FEATURES[d])) for d in DEFAULT_FEATURES}


def _strip_template_lines(src: str, template_md: list[str]) -> str:
    """If a student typed their answer INTO a prompt cell, drop the prompt's lines."""
    tlines = {l.strip().lower() for t in template_md for l in t.splitlines() if l.strip()}
    return "\n".join(l for l in src.splitlines() if l.strip().lower() not in tlines)


def student_cells(cells: list[dict]) -> list[dict]:
    """Cells written by the student inside a task (template and preamble excluded)."""
    return [c for c in cells if not c.get("is_template") and c.get("task_id")]


def explanation_texts(cells: list[dict], ref) -> dict[str, str]:
    """task_id -> cleaned student markdown for tasks that have a reference explanation."""
    tmpl_md = [t["source"] for t in ref.template_cells if t["cell_type"] == "markdown"]
    out = {t: "" for t in ref.explanations}
    for c in student_cells(cells):
        if c["cell_type"] == "markdown" and c["task_id"] in out and not is_placeholder(c["source"]):
            out[c["task_id"]] += " " + markdown_to_text(_strip_template_lines(c["source"], tmpl_md))
    return {k: v.strip() for k, v in out.items()}


def static_features(cells: list[dict], ref) -> tuple[dict, dict]:
    """Returns (features, diagnostics). Diagnostics feed the errors/feedback of the report."""
    sc = student_cells(cells)
    code = [c for c in sc if c["cell_type"] == "code" and c["source"].strip()]
    md = [c for c in sc if c["cell_type"] == "markdown" and not is_placeholder(c["source"])]
    f, diag = {}, {"errors": [], "missing_concepts": {}, "output_mismatch": [], "unexecuted_cells": []}

    # ---------------- code track
    n = len(code)
    syntax_ok = []
    n_err = 0
    for c in code:
        ok, msg = cc.syntax_check(c["source"])
        syntax_ok.append(ok)
        if not ok:
            diag["errors"].append(f"cell {c['cell_index']} ({c['task_id']}): {msg}")
        elif c["has_error"]:
            kind = cc.classify_error(c["error_name"])
            diag["errors"].append(f"cell {c['cell_index']} ({c['task_id']}): {c['error_name']} "
                                  f"[{kind} error] {c['error_value']}")
        n_err += int(c["has_error"] or not ok)
        if c["execution_count"] is None:
            diag["unexecuted_cells"].append(c["cell_index"])
    f["frac_syntax_ok"] = float(np.mean(syntax_ok)) if n else 0.0
    f["frac_executed"] = float(np.mean([c["execution_count"] is not None for c in code])) if n else 0.0
    f["error_cell_rate"] = n_err / n if n else 1.0
    counts = [c["execution_count"] for c in cells if c["cell_type"] == "code" and c["execution_count"]]
    f["out_of_order"] = float(any(b <= a for a, b in zip(counts, counts[1:])))

    matches = []
    for task, spec in ref.expected_outputs.items():
        out = "\n".join(c["output_text"] for c in code if c["task_id"] == task)
        ok = cc.compare_output(out, spec)
        matches.append(ok)
        if not ok:
            diag["output_mismatch"].append(task)
    f["output_match_rate"] = float(np.mean(matches)) if matches else np.nan

    cov = []
    for task, req in ref.required.items():
        if not req:
            continue
        used = set()
        for c in code:
            if c["task_id"] == task:
                used |= cc.ast_summary(c["source"])["calls"]
        cov.append(len(req & used) / len(req))
    f["construct_coverage"] = float(np.mean(cov)) if cov else np.nan

    # ---------------- text track (static part)
    expl = explanation_texts(cells, ref)
    toks = [len(tokenize(t)) for t in expl.values()]
    f["expl_tokens_log"] = math.log1p(sum(toks))
    f["frac_tasks_explained"] = float(np.mean([t > 0 for t in toks])) if toks else 0.0
    kc = []
    for task, info in ref.explanations.items():
        score, _, missing = concept_coverage(expl.get(task, ""), info.get("concepts", []))
        kc.append(score)
        if missing:
            diag["missing_concepts"][task] = missing
    f["keyword_coverage"] = float(np.mean(kc)) if kc else np.nan

    # ---------------- documentation
    comment_lines = sum(len(cc.extract_comments(c["source"])) for c in code)
    loc = sum(cc.code_line_count(c["source"]) for c in code)
    f["comment_density"] = comment_lines / loc if loc else 0.0
    defs = docs = 0
    for c in code:
        s = cc.ast_summary(c["source"])
        defs += s["n_defs"]
        docs += s["n_docstrings"]
    f["docstring_rate"] = docs / defs if defs else 0.0
    f["n_functions"] = defs
    md_tokens = sum(len(tokenize(markdown_to_text(c["source"]))) for c in md)
    f["md_tokens_log"] = math.log1p(md_tokens)
    f["md_to_code_ratio"] = len(md) / n if n else 0.0
    return f, diag


class TfidfSimilarity:
    """TF-IDF cosine between student explanation and reference explanation, per task.

    fit() sees only TRAINING student explanations + reference texts."""

    def __init__(self, **tfidf_kwargs):
        kw = dict(tokenizer=tokenize, lowercase=False, token_pattern=None,
                  ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        kw.update(tfidf_kwargs)
        self.vec = TfidfVectorizer(**kw)

    def fit(self, train_expl: list[dict], refs: dict):
        docs = [t for d in train_expl for t in d.values() if t]
        docs += [info["text"] for r in refs.values() for info in r.explanations.values()]
        self.vec.fit(docs)
        return self

    def transform_one(self, expl: dict, ref) -> dict:
        sims = []
        for task, info in ref.explanations.items():
            s = expl.get(task, "")
            if not s:
                sims.append(0.0)
                continue
            m = self.vec.transform([s, info["text"]])
            sims.append(float(cosine_similarity(m[0], m[1])[0, 0]))
        return {"tfidf_cos_mean": float(np.mean(sims)) if sims else np.nan,
                "tfidf_cos_max": float(np.max(sims)) if sims else np.nan}
