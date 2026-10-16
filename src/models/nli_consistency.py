"""Week 4 improvement: NLI (natural language inference) consistency of student explanations.

Why: the Week 3 bi-encoder embeds the student text and the reference SEPARATELY and compares
the two vectors. That measures topical closeness, so "a higher perplexity means a better model"
stays close to "a lower perplexity means a better model". An NLI cross-encoder reads the
reference (premise) and a student sentence (hypothesis) TOGETHER and outputs probabilities for
  entailment    - the reference supports the student's statement
  neutral       - the reference neither supports nor contradicts it
  contradiction - the reference says the opposite
which is the question a grader actually asks.

Default model: cross-encoder/nli-deberta-v3-xsmall (pretrained on SNLI + MultiNLI by the
sentence-transformers project), run locally on CPU, FROZEN (no fine-tuning, inference only).
The label order is read from the model config at load time, never hard-coded.

Features per notebook (tasks with a reference explanation; unanswered task -> 0):
  nli_support       mean over tasks of mean_s P(entail | premise=reference, hypothesis=student sentence s)
                    "how much of what the student says is backed by the reference"   (precision-like)
  nli_coverage      mean over tasks of mean_r P(entail | premise=student text, hypothesis=reference sentence r)
                    "how much of the reference content the student's text implies"    (recall-like)
  nli_contradiction max over tasks and student sentences of P(contradiction | reference, s)
                    "the strongest statement that conflicts with the reference"

`StubNLI` is a TEST-ONLY lexical stand-in so tests run without downloads. Never report its outputs.
"""
from __future__ import annotations

import re

import numpy as np

from ..preprocessing.text_clean import tokenize
from .semantic_similarity import set_seeds, split_sentences

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
NLI_FEATURES = ["nli_support", "nli_coverage", "nli_contradiction"]
LABELS = ("contradiction", "entailment", "neutral")


def label_order(id2label: dict) -> list[int]:
    """Indices of (contradiction, entailment, neutral) in the model's own output order."""
    norm = {int(k): str(v).lower() for k, v in id2label.items()}
    order = []
    for lab in LABELS:
        hits = [i for i, l in norm.items() if l.startswith(lab[:6])]
        if len(hits) != 1:
            raise ValueError(f"cannot map NLI label {lab!r} from {id2label}")
        order.append(hits[0])
    return order


class BaseNLI:
    name = "base"
    is_stub = False
    resolved_revision = None

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """(premise, hypothesis) pairs -> array (n, 3) of probabilities in LABELS order."""
        raise NotImplementedError


class CrossEncoderNLI(BaseNLI):
    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, revision: str | None = None,
                 local_path: str | None = None, device: str = "cpu", batch_size: int = 16, seed: int = 42):
        set_seeds(seed)
        from pathlib import Path
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError("pip install sentence-transformers (see requirements.txt)") from e
        if local_path:
            path = Path(local_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"nli.local_path {path} does not exist")
            self.resolved_revision = "local:" + str(path)
        else:
            try:
                from huggingface_hub import snapshot_download
                path = snapshot_download(model_name, revision=revision)
            except Exception as e:
                raise RuntimeError(
                    f"Could not load {model_name}. Run once with internet access (it is cached afterwards), "
                    f"or download it and set nli.local_path in the config. Original error: {e}") from e
            self.resolved_revision = Path(path).name
        self.model = CrossEncoder(str(path), device=device)
        inner = getattr(self.model, "model", None)
        cfg = getattr(self.model, "config", None) or getattr(inner, "config", None)
        self.order = label_order(cfg.id2label)
        if inner is not None:
            inner.eval()
            for p in inner.parameters():
                p.requires_grad_(False)
        self.name, self.batch_size = model_name, batch_size

    def predict(self, pairs):
        import torch
        if not pairs:
            return np.zeros((0, 3))
        with torch.inference_mode():
            logits = np.asarray(self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False,
                                                   activation_fn=torch.nn.Identity(),  # raw logits
                                                   convert_to_numpy=True), dtype=np.float64)
        logits = logits[:, self.order]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


class StubNLI(BaseNLI):
    """TEST-ONLY. Entailment ~ token overlap; contradiction fires when the hypothesis swaps a
    polarity word relative to the premise. Lexical heuristics, NOT a model."""

    name = "STUB lexical NLI (NOT A MODEL, tests only)"
    is_stub = True
    resolved_revision = "stub"
    FLIPS = {("higher", "lower"), ("more", "less"), ("increase", "decrease"), ("rare", "common"),
             ("better", "worse"), ("most", "least")}

    def predict(self, pairs):
        out = []
        for prem, hyp in pairs:
            p, h = set(tokenize(prem)), set(tokenize(hyp))
            overlap = len(p & h) / max(1, len(h))
            flip = any((a in p and b in h and b not in p) or (b in p and a in h and a not in p) for a, b in self.FLIPS)
            c = 0.8 if flip else 0.05
            e = (1 - c) * overlap
            out.append([c, e, max(0.0, 1 - c - e)])
        return np.array(out).reshape(-1, 3)


def build_nli(cfg: dict) -> BaseNLI:
    kind = cfg.get("encoder", "cross-encoder")
    if kind == "stub":
        return StubNLI()
    if kind == "cross-encoder":
        return CrossEncoderNLI(cfg.get("model_name", DEFAULT_NLI_MODEL), cfg.get("revision"), cfg.get("local_path"),
                               cfg.get("device", "cpu"), cfg.get("batch_size", 16), cfg.get("seed", 42))
    raise ValueError(f"unknown nli.encoder {kind}")


class NLIConsistency:
    """Stateless apart from a prediction cache: nothing is learned from our data."""

    def __init__(self, nli: BaseNLI, flag_threshold: float = 0.5, max_premise_chars: int = 2000):
        self.nli = nli
        self.flag_threshold = flag_threshold
        self.max_premise_chars = max_premise_chars
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def probs(self, pairs):
        pairs = [(p[: self.max_premise_chars], h) for p, h in pairs]
        missing = [pr for pr in dict.fromkeys(pairs) if pr not in self._cache]
        if missing:
            for pr, v in zip(missing, self.nli.predict(missing)):
                self._cache[pr] = v
        return np.stack([self._cache[pr] for pr in pairs]) if pairs else np.zeros((0, 3))

    def task_scores(self, student: str, reference: str) -> dict:
        s_sents = [s for s in split_sentences(student) if tokenize(s)]
        r_sents = [s for s in split_sentences(reference) if tokenize(s)]
        if not s_sents or not reference.strip():
            return {"support": 0.0, "coverage": 0.0, "contradiction": 0.0, "worst_sentence": None}
        fwd = self.probs([(reference, s) for s in s_sents])          # reference -> student sentences
        bwd = self.probs([(student, r) for r in r_sents])            # student text -> reference sentences
        worst = int(np.argmax(fwd[:, 0]))
        return {"support": float(fwd[:, 1].mean()), "coverage": float(bwd[:, 1].mean()) if len(bwd) else 0.0,
                "contradiction": float(fwd[:, 0].max()),
                "worst_sentence": {"text": s_sents[worst], "p_contradiction": round(float(fwd[worst, 0]), 3)}}

    def notebook_features(self, expl: dict, ref) -> tuple[dict, dict]:
        per = {t: self.task_scores(expl.get(t, ""), info["text"]) for t, info in ref.explanations.items()}
        if not per:
            return {f: np.nan for f in NLI_FEATURES}, {"per_task": {}, "flags": []}
        feats = {"nli_support": float(np.mean([v["support"] for v in per.values()])),
                 "nli_coverage": float(np.mean([v["coverage"] for v in per.values()])),
                 "nli_contradiction": float(np.max([v["contradiction"] for v in per.values()]))}
        flags = [f"{t}: \"{v['worst_sentence']['text']}\" may contradict the reference "
                 f"(p_contradiction={v['worst_sentence']['p_contradiction']})"
                 for t, v in per.items() if v["worst_sentence"] and v["contradiction"] >= self.flag_threshold]
        return feats, {"per_task": {t: {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}
                                    for t, v in per.items()}, "flags": flags}

    def info(self) -> dict:
        return {"model": self.nli.name, "revision": self.nli.resolved_revision, "is_stub": self.nli.is_stub,
                "frozen": True, "labels": list(LABELS), "flag_threshold": self.flag_threshold}
