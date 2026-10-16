"""Week 4 robustness suite: controlled variations of a correct explanation.

For each case (reference explanation + a correct student paraphrase + a wrong-but-similar answer),
the variants are:
  correct_paraphrase      the correct explanation as written
  sentence_order_reversed same sentences, reverse order (meaning preserved)
  words_scrambled         same words in random order (seed 42): meaning destroyed, vocabulary kept
  irrelevant_appended     correct explanation + an unrelated sentence
  short                   first 5 words only
  empty                   ""
  incorrect_similar       wrong explanation that reuses the reference vocabulary
  reference_copy          verbatim copy of the reference
Each variant is scored by every available scorer. These are synthetic behaviour checks, not
real-world validation.
"""
from __future__ import annotations

import json
import random

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..models.semantic_similarity import split_sentences
from ..preprocessing.text_clean import tokenize

EXPECTED = {  # what a scorer that tracks MEANING should do, relative to correct_paraphrase
    "correct_paraphrase": "high (anchor)",
    "sentence_order_reversed": "about the same as anchor",
    "words_scrambled": "much lower than anchor",
    "irrelevant_appended": "about the same or slightly lower",
    "short": "lower",
    "empty": "zero",
    "incorrect_similar": "lower than anchor; contradiction should rise",
    "reference_copy": "maximal",
}


def make_variants(case: dict, irrelevant: str, seed: int = 42) -> dict:
    good = case["correct"]
    words = good.split()
    rng = random.Random(seed)
    scrambled = words[:]
    rng.shuffle(scrambled)
    sents = split_sentences(good)
    return {
        "correct_paraphrase": good,
        "sentence_order_reversed": " ".join(reversed(sents)) if len(sents) > 1 else
        ", ".join(reversed([p.strip() for p in good.rstrip(".").split(",")])) + ".",
        "words_scrambled": " ".join(scrambled),
        "irrelevant_appended": good + " " + irrelevant,
        "short": " ".join(words[:5]),
        "empty": "",
        "incorrect_similar": case["incorrect_similar"],
        "reference_copy": case["reference"],
    }


def run_robustness(path, sem=None, nli=None) -> pd.DataFrame:
    spec = json.loads(open(path).read())
    rows = []
    for case in spec["cases"]:
        for name, text in make_variants(case, spec["irrelevant_sentence"]).items():
            rows.append({"case": case["id"], "variant": name, "student": text, "reference": case["reference"]})
    df = pd.DataFrame(rows)
    corpus = list(dict.fromkeys(df.reference.tolist() + [t for t in df.student if t]))
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2),
                          sublinear_tf=True, stop_words="english").fit(corpus)

    def tf(a, b):
        if not a.strip():
            return 0.0
        m = vec.transform([a, b])
        return round(float(cosine_similarity(m[0], m[1])[0, 0]), 3)

    df["tfidf_cos"] = [tf(s, r) for s, r in zip(df.student, df.reference)]
    if sem is not None:
        df["semantic_cos"] = [round(sem.similarity(s, r), 3) for s, r in zip(df.student, df.reference)]
    if nli is not None:
        sc = [nli.task_scores(s, r) for s, r in zip(df.student, df.reference)]
        df["nli_support"] = [round(x["support"], 3) for x in sc]
        df["nli_contradiction"] = [round(x["contradiction"], 3) for x in sc]
    df["expected_if_meaning_tracked"] = df.variant.map(EXPECTED)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean score per variant and scorer across cases."""
    cols = [c for c in ("tfidf_cos", "semantic_cos", "nli_support", "nli_contradiction") if c in df.columns]
    order = list(EXPECTED)
    out = df.groupby("variant")[cols].mean().round(3).reindex(order)
    out["expected_if_meaning_tracked"] = [EXPECTED[v] for v in out.index]
    return out
