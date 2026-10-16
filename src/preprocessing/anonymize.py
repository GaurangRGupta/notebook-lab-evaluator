"""Anonymisation of .ipynb submissions.

Runs BEFORE any other processing. Nothing here calls an external service.

What is removed / replaced:
  * roll numbers (configurable regex, default J + 3 digits)  -> [ROLL]
  * e-mail addresses                                        -> [EMAIL]
  * names from a local class list (case-insensitive)        -> [NAME]
  * user names inside file paths in tracebacks/outputs      -> /Users/[USER]
  * notebook- and cell-level metadata (authors, widgets, Colab ids, timing)

The original file name is replaced by a salted hash so submissions cannot be
traced back without the salt, which must be stored OUTSIDE the repository.
"""
from __future__ import annotations

import copy
import hashlib
import re
from typing import Iterable

import nbformat

DEFAULT_ROLL_PATTERN = r"\b[A-Z]\d{3}\b"
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PATH_USER_RE = re.compile(r"(/Users/|/home/|C:\\\\Users\\\\|C:\\Users\\)([^/\\\s'\"]+)")


def notebook_hash(original_name: str, salt: str, length: int = 12) -> str:
    return hashlib.sha256((salt + "::" + original_name).encode()).hexdigest()[:length]


class Scrubber:
    def __init__(self, roll_pattern: str = DEFAULT_ROLL_PATTERN, names: Iterable[str] = ()):
        self.roll_re = re.compile(roll_pattern)
        # longest names first so "Fake Student" is replaced before "Fake"
        names = sorted({n.strip() for n in names if n.strip()}, key=len, reverse=True)
        self.name_res = [re.compile(r"\b" + re.escape(n) + r"\b", re.IGNORECASE) for n in names]

    def scrub(self, text: str) -> str:
        if not text:
            return text
        text = EMAIL_RE.sub("[EMAIL]", text)
        text = PATH_USER_RE.sub(lambda m: m.group(1) + "[USER]", text)
        text = self.roll_re.sub("[ROLL]", text)
        for rx in self.name_res:
            text = rx.sub("[NAME]", text)
        return text

    def _scrub_value(self, v):
        if isinstance(v, str):
            return self.scrub(v)
        if isinstance(v, list):
            return [self._scrub_value(x) for x in v]
        return v

    def scrub_notebook(self, nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
        nb = copy.deepcopy(nb)
        lang = nb.metadata.get("kernelspec", {}).get("language") or \
            nb.metadata.get("language_info", {}).get("name", "python")
        synthetic = nb.metadata.get("synthetic_development_data", False)
        nb.metadata = nbformat.NotebookNode({"language": lang})
        if synthetic:  # keep the SYNTHETIC marker so it can never be confused with real data
            nb.metadata["synthetic_development_data"] = True
        for cell in nb.cells:
            cell.metadata = nbformat.NotebookNode({})
            cell.pop("attachments", None)
            cell.source = self.scrub(cell.source)
            for out in cell.get("outputs", []):
                if "text" in out:
                    out["text"] = self._scrub_value(out["text"])
                if "evalue" in out:
                    out["evalue"] = self.scrub(out["evalue"])
                if "traceback" in out:
                    out["traceback"] = [self.scrub(t) for t in out["traceback"]]
                if "data" in out:
                    for mime in list(out["data"].keys()):
                        if mime.startswith("text/"):
                            out["data"][mime] = self._scrub_value(out["data"][mime])
                out.pop("metadata", None)
                if out.get("output_type") in ("execute_result", "display_data"):
                    out["metadata"] = {}
        return nb


def load_class_list(path) -> list[str]:
    """One name per line. Kept locally, never committed for real data."""
    try:
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip() and not l.startswith("#")]
    except FileNotFoundError:
        return []
