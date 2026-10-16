"""Week 3 evaluation helpers for the semantic component.

1. calibrate_concept_threshold: choose the cosine threshold for semantic concept matching on
   the TUNE half of an authored concept/sentence pair set, then report precision/recall/F1 for
   lexical vs semantic vs lexical-or-semantic matching on the held-out EVAL half. Tune and eval
   halves use DIFFERENT concepts. Notebooks (and in particular the test split) are never used.
2. run_probes: error-analysis probe pairs (paraphrases, wrong-but-similar-vocabulary, vague,
   unrelated, wrong relationships), scored by TF-IDF cosine and by semantic cosine.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..preprocessing.text_clean import concept_coverage, tokenize
from .metrics import binary_metrics


def load_concept_pairs(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["variants"] = df["variants"].fillna("").map(lambda s: [v for v in s.split("|") if v])
    return df


def _best_scores(sem, df: pd.DataFrame) -> np.ndarray:
    out = []
    for _, r in df.iterrows():
        phrases = [r.concept_term] + [v for v in r.variants if len(v) >= sem.min_variant_chars]
        P, S = sem.embed(phrases), sem.embed([r.sentence])
        out.append(float((P @ S.T).max()))
    return np.array(out)


def _lexical(df: pd.DataFrame) -> np.ndarray:
    return np.array([concept_coverage(r.sentence, [{"term": r.concept_term, "variants": r.variants}])[0] == 1.0
                     for _, r in df.iterrows()], dtype=int)


def calibrate_concept_threshold(sem, pairs: pd.DataFrame, grid=None) -> dict:
    grid = list(grid) if grid is not None else [round(x, 3) for x in np.arange(0.20, 0.801, 0.025)]
    tune, ev = pairs[pairs.split == "tune"], pairs[pairs.split == "eval"]
    if set(tune.concept_term) & set(ev.concept_term):
        raise ValueError("tune and eval halves must use different concepts")
    s_tune = _best_scores(sem, tune)
    lex_tune = _lexical(tune)
    table = []
    for t in grid:
        pred = np.maximum(lex_tune, (s_tune >= t).astype(int))  # coverage rule = lexical OR semantic
        m = binary_metrics(tune.label, pred)
        table.append({"threshold": t, **m})
    best_f1 = max(r["f1"] for r in table)
    # among thresholds tied on F1, take the median one (less sensitive than an edge value)
    tied = [r["threshold"] for r in table if r["f1"] == best_f1]
    thr = float(tied[len(tied) // 2])
    s_ev, lex_ev = _best_scores(sem, ev), _lexical(ev)
    sem_ev = (s_ev >= thr).astype(int)
    return {
        "threshold": thr, "chosen_on": "tune split of concept pairs only",
        "n_tune": int(len(tune)), "n_eval": int(len(ev)),
        "tune_curve": table,
        "eval": {"lexical_only": binary_metrics(ev.label, lex_ev),
                 "semantic_only": binary_metrics(ev.label, sem_ev),
                 "lexical_or_semantic": binary_metrics(ev.label, np.maximum(lex_ev, sem_ev))},
        "eval_rows": [{"concept": r.concept_term, "sentence": r.sentence, "label": int(r.label),
                       "lexical": int(l), "semantic_score": round(float(s), 3), "semantic": int(p)}
                      for (_, r), l, s, p in zip(ev.iterrows(), lex_ev, s_ev, sem_ev)],
    }


def run_probes(sem, probes_path) -> pd.DataFrame:
    probes = json.loads(open(probes_path).read())["probes"]
    texts = [p["reference"] for p in probes] + [p["student"] for p in probes]
    # stand-alone TF-IDF with the pipeline's settings, fitted on the probe texts themselves
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                          ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(texts)
    rows = []
    for p in probes:
        m = vec.transform([p["student"], p["reference"]])
        rows.append({"id": p["id"], "category": p["category"], "student_is_correct": p["correct"],
                     "tfidf_cos": round(float(cosine_similarity(m[0], m[1])[0, 0]), 3),
                     "semantic_cos": round(sem.similarity(p["student"], p["reference"]), 3),
                     "reference": p["reference"], "student": p["student"], "note": p.get("note", "")})
    return pd.DataFrame(rows)


def probe_separation(df: pd.DataFrame) -> dict:
    """Within each reference group: share of (correct, incorrect) probe pairs where the correct
    explanation gets the higher similarity. 0.5 = no separation. Tiny, authored probe set."""
    out = {}
    for col in ("tfidf_cos", "semantic_cos"):
        wins = total = 0
        for _, g in df.groupby("reference"):
            ok, bad = g[g.student_is_correct][col], g[~g.student_is_correct][col]
            for a in ok:
                for b in bad:
                    total += 1
                    wins += 1.0 if a > b else 0.5 if a == b else 0.0
        out[col] = {"pairs": total, "correct_ranked_higher": (wins / total) if total else None}
    return out
