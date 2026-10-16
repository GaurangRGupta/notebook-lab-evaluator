"""Markdown cleaning and concept matching.

Deliberately light: technical terms (tf-idf, word2vec, p(w|c), bi-lstm) must survive.
No stemming or lemmatisation in the baseline.
"""
from __future__ import annotations

import difflib
import re

HTML_TAG_RE = re.compile(r"<[^>]+>")
IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
EMPH_RE = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\n?")
MATH_RE = re.compile(r"\$\$?(.+?)\$\$?", re.DOTALL)
# keeps tf-idf, word2vec, bi-lstm, p(w|c)-style tokens split sensibly, 3.14, n-gram
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.'][a-z0-9]+)*")
PLACEHOLDERS = {"", "-", "todo", "tbd", "your answer here", "answer here", "write your answer here",
                "na", "n/a", "..."}


def markdown_to_text(md: str) -> str:
    t = md or ""
    t = IMG_RE.sub(" ", t)
    t = LINK_RE.sub(r"\1", t)
    t = HTML_TAG_RE.sub(" ", t)
    t = FENCE_RE.sub("", t)
    t = MATH_RE.sub(r" \1 ", t)
    t = HEADER_RE.sub("", t)
    t = EMPH_RE.sub(r"\2", t)
    t = t.replace("`", "")
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.MULTILINE)  # bullets
    return re.sub(r"\s+", " ", t).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def is_placeholder(text: str) -> bool:
    return markdown_to_text(text).lower().strip(" .") in PLACEHOLDERS


def similarity(a: str, b: str) -> float:
    na = re.sub(r"\s+", " ", (a or "").lower()).strip()
    nb = re.sub(r"\s+", " ", (b or "").lower()).strip()
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def concept_coverage(text: str, concepts: list[dict]) -> tuple[float, list[str], list[str]]:
    """concepts: [{"term": str, "variants": [str, ...]}]. Phrase match on token sequences."""
    if not concepts:
        return 0.0, [], []
    toks = tokenize(text)
    joined = " " + " ".join(toks) + " "
    hit, miss = [], []
    for c in concepts:
        variants = [c["term"]] + list(c.get("variants", []))
        found = any(" " + " ".join(tokenize(v)) + " " in joined for v in variants if tokenize(v))
        (hit if found else miss).append(c["term"])
    return len(hit) / len(concepts), hit, miss
