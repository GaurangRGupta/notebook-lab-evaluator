"""Week 6 asset: score one notebook with the frozen Week 5 models.

The notebook is read and not executed. Label files are not opened.
Marks are code 0-8, explanation 0-8, documentation 0-4, overall 0-20.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import nbformat
import pandas as pd

from .data.references import load_reference
from .models.features import explanation_texts, static_features, variant_features
from .models.semantic_similarity import SemanticSimilarity, build_encoder
from .preprocessing.align_tasks import assign_tasks, flag_template_cells
from .preprocessing.anonymize import Scrubber
from .preprocessing.parse_notebook import parse_cells
from .reporting import (
    AGREEMENT_METHOD,
    _finite,
    choose_missing,
    confidence_from_variants,
    feedback_and_flag,
    marks,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_DIR = ROOT / "results" / "week6" / "asset"
DIMS = ("code", "explanation", "documentation")
CONFIDENCE_METHOD = AGREEMENT_METHOD
FIT_HINT = "python scripts/fit_week6_asset.py"


class Asset:
    def __init__(self, meta: dict, tfidf, variant_models: dict, sem, refs: dict, asset_dir: Path):
        self.meta = meta
        self.tfidf = tfidf
        self.variant_models = variant_models
        self.sem = sem
        self.refs = refs
        self.asset_dir = asset_dir
        self.primary = meta["primary_variant"]
        self.rubric_max = {k: float(v) for k, v in meta["rubric_max"].items()}
        self.data_status = meta["data_status"]
        self.encoder_name = meta.get("encoder_name")
        self.confidence_method = AGREEMENT_METHOD
        self.template_threshold = float(meta["template_similarity_threshold"])
        self.roll_pattern = meta.get("roll_pattern", r"\b[A-Z]\d{3}\b")


def load_asset(asset_dir: Path | None = None) -> Asset:
    folder = Path(asset_dir) if asset_dir else DEFAULT_ASSET_DIR
    meta_path = folder / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"No fitted asset at {folder}. Run: {FIT_HINT}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tfidf = joblib.load(folder / "tfidf.joblib")
    models = joblib.load(folder / "variant_models.joblib")
    sem = None
    if meta.get("semantic_enabled"):
        onnx_dir = Path(meta["onnx_dir"])
        if not onnx_dir.is_absolute():
            onnx_dir = ROOT / onnx_dir
        encoder = build_encoder({"encoder": "onnx", "onnx_dir": str(onnx_dir), "batch_size": 32})
        sem = SemanticSimilarity(encoder, concept_threshold=float(meta["concept_threshold"]))
    data = ROOT / meta["data_dir"]
    ref_root = data / meta["references_dir"]
    refs = {}
    if ref_root.is_dir():
        for path in sorted(p for p in ref_root.iterdir() if p.is_dir()):
            refs[path.name] = load_reference(ref_root, path.name)
    return Asset(meta, tfidf, models, sem, refs, folder)


def _notebook_from_upload(notebook) -> dict:
    if isinstance(notebook, (bytes, bytearray)):
        text = notebook.decode("utf-8")
    elif isinstance(notebook, Path):
        text = notebook.read_text(encoding="utf-8")
    else:
        text = str(notebook)
    return nbformat.reads(text, as_version=4)


def evaluate_notebook(notebook, assignment_id: str, asset: Asset | None = None) -> dict:
    asset = asset or load_asset()
    if assignment_id not in asset.refs:
        known = ", ".join(sorted(asset.refs)) or "(none)"
        raise ValueError(f"Unknown assignment_id {assignment_id!r}. Known ids: {known}.")
    ref = asset.refs[assignment_id]
    nb = Scrubber(asset.roll_pattern).scrub_notebook(_notebook_from_upload(notebook))
    cells = flag_template_cells(
        assign_tasks(parse_cells(nb, "request")),
        ref.template_cells,
        asset.template_threshold,
    )
    features, diag = static_features(cells, ref)
    expl = explanation_texts(cells, ref)
    features.update(asset.tfidf.transform_one(expl, ref))
    semantic_missing = None
    if asset.sem is not None:
        sem_feats, sem_diag = asset.sem.notebook_features(expl, ref)
        features.update(sem_feats)
        semantic_missing = sem_diag.get("missing_concepts_after_semantic")
    row = pd.DataFrame([features])
    raw = {}
    for variant, models in asset.variant_models.items():
        raw[variant] = {dim: float(models[dim].predict(row)[0]) for dim in DIMS}
    primary = raw[asset.primary]
    scored = marks(primary, asset.rubric_max)
    encoder_loaded = asset.sem is not None
    conf = confidence_from_variants(
        raw,
        asset.rubric_max,
        variant_features("week2_tfidf"),
        variant_features("week3_combined"),
        features,
        diag,
        encoder_loaded,
    )
    missing = choose_missing(diag["missing_concepts"], semantic_missing, encoder_loaded)
    rate_num = _finite(features.get("output_match_rate"))
    text, flag = feedback_and_flag(missing, rate_num)
    errors = (
        list(diag["errors"])
        + [f"output mismatch in {task}" for task in diag["output_mismatch"]]
        + [f"cell {index} was never executed" for index in diag["unexecuted_cells"]]
    )
    return {
        "assignment_id": assignment_id,
        "scoring_variant": asset.primary,
        "code": scored["code"],
        "explanation": scored["explanation"],
        "documentation": scored["documentation"],
        "overall": scored["overall"],
        "max_score": scored["max_score"],
        "output_match_rate": None if rate_num is None else round(rate_num, 4),
        "errors": errors,
        "missing_concepts": missing,
        "feedback": text,
        "confidence": conf,
        "flag_output_ok_explanation_weak": flag,
        "data_status": asset.data_status,
    }
