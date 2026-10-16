"""Week 3 core NLP component: semantic similarity of student explanations.

Student explanation ─► sentence encoder ─► 384-d unit vector ┐
Reference explanation ─► sentence encoder ─► 384-d unit vector ┴► cosine = dot product ─► features

Default model: sentence-transformers/all-MiniLM-L6-v2 (pretrained, FROZEN, run locally).
  * no fine-tuning, no gradients (torch.inference_mode), model.eval()
  * each embedding depends only on its own text, so there is nothing to "fit" across
    train/val/test and no leakage path through the encoder
  * runs on a laptop CPU; no student text leaves the machine (Week 1 privacy rule)

The module is reusable on its own (later: POST /evaluate-notebook, Group 2's grader):

    enc = SentenceTransformerEncoder()                 # or local_path="models/all-MiniLM-L6-v2"
    sem = SemanticSimilarity(enc, concept_threshold=0.5)
    sem.similarity("student text", "reference text")   # float in [-1, 1]

`HashingStubEncoder` exists ONLY so unit tests and CI can run without downloading the
model. It is a character n-gram hashing projection, i.e. LEXICAL, not semantic. Results
produced with it must never be reported as semantic-embedding results.
"""
from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

import numpy as np

from ..preprocessing.text_clean import concept_coverage, tokenize

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT_RE.split(text or "") if s and s.strip()]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Encoders return unit vectors, so this is the dot product;
    norms are still checked so zero vectors (empty text) give 0 instead of NaN."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


# ------------------------------------------------------------------ encoders
class BaseEncoder:
    name = "base"
    dim = 0
    max_tokens = 256
    is_stub = False
    resolved_revision = None

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def n_tokens(self, text: str) -> int:
        return int(len(tokenize(text)) * 1.3) + 2  # rough word-piece estimate

    def encode(self, texts: list[str]) -> np.ndarray:
        """Unit-normalised embeddings, shape (len(texts), dim). Empty text -> zero vector.
        Texts longer than the model window are split into sentence windows whose
        embeddings are averaged and re-normalised (instead of silently truncating)."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        plan = []  # (row, [chunks])
        for i, t in enumerate(texts):
            t = (t or "").strip()
            if not t:
                continue
            plan.append((i, self._chunks(t)))
        flat = [c for _, cs in plan for c in cs]
        if not flat:
            return out
        embs = self._encode_batch(flat)
        k = 0
        for i, cs in plan:
            v = embs[k:k + len(cs)].mean(axis=0)
            k += len(cs)
            n = np.linalg.norm(v)
            out[i] = v / n if n else v
        return out

    def _chunks(self, text: str) -> list[str]:
        if self.n_tokens(text) <= self.max_tokens:
            return [text]
        chunks, cur = [], ""
        for s in split_sentences(text):
            cand = (cur + " " + s).strip()
            if cur and self.n_tokens(cand) > self.max_tokens:
                chunks.append(cur)
                cur = s
            else:
                cur = cand
        if cur:
            chunks.append(cur)
        return chunks  # a single over-long sentence is left to the model's own truncation


class SentenceTransformerEncoder(BaseEncoder):
    """Frozen pretrained Sentence Transformer, loaded from the Hugging Face cache or a local folder."""

    def __init__(self, model_name: str = DEFAULT_MODEL, revision: str | None = None,
                 local_path: str | None = None, device: str = "cpu", batch_size: int = 32, seed: int = 42):
        set_seeds(seed)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("pip install sentence-transformers (see requirements.txt)") from e
        if local_path:
            path = Path(local_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"semantic.local_path {path} does not exist")
            self.resolved_revision = "local:" + str(path)
        else:
            try:
                from huggingface_hub import snapshot_download
                path = snapshot_download(model_name, revision=revision)
            except Exception as e:  # network blocked, no cache, etc.
                raise RuntimeError(
                    f"Could not load {model_name}. Run once with internet access (it is cached afterwards), "
                    f"or download it and set semantic.local_path in the config. Original error: {e}") from e
            self.resolved_revision = Path(path).name  # HF snapshot folder name = commit hash
        self.model = SentenceTransformer(str(path), device=device)
        self.model.eval()
        for p in self.model.parameters():  # frozen feature extractor
            p.requires_grad_(False)
        self.name = model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.max_tokens = int(self.model.max_seq_length)
        self.batch_size = batch_size

    def n_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=True)["input_ids"])

    def _encode_batch(self, texts):
        import torch
        with torch.inference_mode():
            return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


class OnnxSentenceEncoder(BaseEncoder):
    """The same all-MiniLM-L6-v2 weights exported to ONNX (Xenova export used by transformers.js),
    run with onnxruntime + tokenizers. Added in Week 5 because huggingface.co is blocked in the
    development environment. Pooling reproduces the sentence-transformers pipeline exactly:
    mean over non-padding tokens, then L2 normalisation; 256-token window (model max_seq_length).
    Provenance is checked behaviourally by `verify_reference` against the published
    sentence-transformers README values (0.6660, 0.1046, 0.1411)."""

    def __init__(self, model_dir: str, max_tokens: int = 256, batch_size: int = 32, threads: int = 1):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        d = Path(model_dir).expanduser()
        if not (d / "onnx" / "model.onnx").exists():
            raise FileNotFoundError(f"{d}/onnx/model.onnx not found (run scripts/fetch_models.py)")
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_tokens)
        self.tok.enable_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1  # deterministic CPU execution
        self.sess = ort.InferenceSession(str(d / "onnx" / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.name = "sentence-transformers/all-MiniLM-L6-v2 (ONNX export)"
        self.dim, self.max_tokens, self.batch_size = 384, max_tokens, batch_size
        self.resolved_revision = "onnx:" + str(d)

    def n_tokens(self, text: str) -> int:
        self.tok.no_truncation()
        n = len(self.tok.encode(text).ids)
        self.tok.enable_truncation(self.max_tokens)
        return n

    def _encode_batch(self, texts):
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok.encode_batch(texts[i:i + self.batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            am = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": am}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]
            m = am[..., None].astype(np.float32)
            v = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32)


README_SENTENCES = ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]
README_SIMILARITIES = {(0, 1): 0.6660, (0, 2): 0.1046, (1, 2): 0.1411}  # sentence-transformers README


def verify_reference(encoder: BaseEncoder, tol: float = 5e-4) -> dict:
    """Behavioural provenance check: reproduce the published all-MiniLM-L6-v2 similarity matrix."""
    e = encoder.encode(README_SENTENCES)
    got = {f"{a}-{b}": round(float(e[a] @ e[b]), 4) for a, b in README_SIMILARITIES}
    ok = all(abs(float(e[a] @ e[b]) - v) <= tol for (a, b), v in README_SIMILARITIES.items())
    return {"expected": {f"{a}-{b}": v for (a, b), v in README_SIMILARITIES.items()}, "got": got,
            "tolerance": tol, "passed": bool(ok)}


class HashingStubEncoder(BaseEncoder):
    """TEST-ONLY stand-in. Character 3-gram hashing -> L2-normalised vector.
    Deterministic, dependency-free, and LEXICAL (not semantic)."""

    name = "STUB hashing char-3-grams (NOT SEMANTIC, tests only)"
    is_stub = True

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.resolved_revision = "stub"

    def _encode_batch(self, texts):
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            s = f"  {t.lower()}  "
            for j in range(len(s) - 2):
                h = int(hashlib.md5(s[j:j + 3].encode()).hexdigest(), 16)
                m[i, h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
            n = np.linalg.norm(m[i])
            if n:
                m[i] /= n
        return m


def build_encoder(cfg: dict) -> BaseEncoder:
    kind = cfg.get("encoder", "sentence-transformers")
    if kind == "stub":
        return HashingStubEncoder(cfg.get("stub_dim", 256))
    if kind == "onnx":
        return OnnxSentenceEncoder(cfg["onnx_dir"], cfg.get("max_tokens", 256), cfg.get("batch_size", 32))
    if kind == "sentence-transformers":
        return SentenceTransformerEncoder(cfg.get("model_name", DEFAULT_MODEL), cfg.get("revision"),
                                          cfg.get("local_path"), cfg.get("device", "cpu"),
                                          cfg.get("batch_size", 32), cfg.get("seed", 42))
    raise ValueError(f"unknown semantic.encoder {kind}")


# ------------------------------------------------------------------ similarity + features
SEMANTIC_FEATURES = ["sem_cos_mean", "sem_cos_max", "sem_concept_coverage"]


class SemanticSimilarity:
    """Stateless apart from an embedding cache: nothing is learned from the data."""

    def __init__(self, encoder: BaseEncoder, concept_threshold: float = 0.5, min_variant_chars: int = 3):
        self.encoder = encoder
        self.concept_threshold = float(concept_threshold)
        self.min_variant_chars = min_variant_chars
        self._cache: dict[str, np.ndarray] = {}

    # -- embeddings
    def embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for t, v in zip(missing, self.encoder.encode(missing)):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, self.encoder.dim))

    def similarity(self, a: str, b: str) -> float:
        if not (a or "").strip() or not (b or "").strip():
            return 0.0  # policy: a missing explanation has zero similarity
        va, vb = self.embed([a, b])
        return cosine(va, vb)

    # -- per task / per notebook
    def task_similarities(self, expl: dict, ref) -> dict:
        """task_id -> cosine(student explanation, reference explanation); 0 when unanswered."""
        return {t: self.similarity(expl.get(t, ""), info["text"]) for t, info in ref.explanations.items()}

    def concept_coverage(self, text: str, concepts: list[dict]) -> dict:
        """A concept counts as covered if it is matched LEXICALLY (Week 2 rule) or if the best
        cosine between any of its phrasings and any student sentence >= concept_threshold.
        Returns score, lexical hits, semantic-only hits, misses and the best score per concept."""
        if not concepts:
            return {"score": 0.0, "lexical": [], "semantic_added": [], "missing": [], "best": {}}
        _, lex_hit, _ = concept_coverage(text, concepts)
        sents = split_sentences(text)
        best = {}
        if sents:
            S = self.embed(sents)
            for c in concepts:
                phrases = [c["term"]] + [v for v in c.get("variants", []) if len(v) >= self.min_variant_chars]
                P = self.embed(phrases)
                best[c["term"]] = float((P @ S.T).max())
        sem_added = [c["term"] for c in concepts
                     if c["term"] not in lex_hit and c["term"] in best and best[c["term"]] >= self.concept_threshold]
        covered = set(lex_hit) | set(sem_added)
        return {"score": len(covered) / len(concepts), "lexical": lex_hit, "semantic_added": sem_added,
                "missing": [c["term"] for c in concepts if c["term"] not in covered], "best": best}

    def notebook_features(self, expl: dict, ref) -> tuple[dict, dict]:
        sims = self.task_similarities(expl, ref)
        cov = {t: self.concept_coverage(expl.get(t, ""), info.get("concepts", []))
               for t, info in ref.explanations.items()}
        vals = list(sims.values())
        feats = {
            "sem_cos_mean": float(np.mean(vals)) if vals else np.nan,
            "sem_cos_max": float(np.max(vals)) if vals else np.nan,
            "sem_concept_coverage": float(np.mean([c["score"] for c in cov.values()])) if cov else np.nan,
        }
        diag = {"per_task_similarity": sims,
                "semantic_concepts_added": {t: c["semantic_added"] for t, c in cov.items() if c["semantic_added"]},
                "missing_concepts_after_semantic": {t: c["missing"] for t, c in cov.items() if c["missing"]}}
        return feats, diag

    def info(self) -> dict:
        e = self.encoder
        return {"model": e.name, "revision": e.resolved_revision, "dim": e.dim, "max_tokens": e.max_tokens,
                "is_stub": e.is_stub, "concept_threshold": self.concept_threshold, "frozen": True}
