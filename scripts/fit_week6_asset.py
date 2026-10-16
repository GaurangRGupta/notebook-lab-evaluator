"""Fit the Week 6 asset into results/week6 and compare it with Week 5 predictions.

Does not write results/week5. Does not enable NLI. Stops if the new predictions
disagree with the Week 5 file by more than 0.05 marks.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reporting import AGREEMENT_METHOD as CONFIDENCE_METHOD  # noqa: E402
from src.pipeline import DIMS, Pipeline  # noqa: E402

MAX_ABS_DIFF = 0.05
WEEK5_PREDICTIONS = ROOT / "results" / "week5" / "predictions.csv"
ONNX_FILE = ROOT / "models" / "all-MiniLM-L6-v2-onnx" / "onnx" / "model.onnx"
PROVENANCE = ROOT / "models" / "all-MiniLM-L6-v2-onnx" / "provenance_check.json"


def ensure_onnx() -> bool:
    if ONNX_FILE.is_file() and PROVENANCE.is_file():
        report = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        if report.get("passed"):
            print("[week6] ONNX provenance check already passed")
            return True
    print("[week6] fetching the verified ONNX encoder")
    completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_models.py")], cwd=ROOT)
    if completed.returncode != 0 or not PROVENANCE.is_file():
        print("[week6] encoder fetch failed")
        return False
    report = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    if not report.get("passed"):
        print("[week6] encoder provenance check failed")
        return False
    return True


def save_asset(pipe: Pipeline, semantic_enabled: bool) -> Path:
    folder = pipe.out / "asset"
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe.tfidf, folder / "tfidf.joblib")
    joblib.dump(pipe.variant_models, folder / "variant_models.joblib")
    meta = {
        "primary_variant": pipe.primary,
        "rubric_max": pipe.cfg["rubric_max"],
        "data_status": pipe.status,
        "data_dir": pipe.cfg["data_dir"],
        "references_dir": pipe.cfg["references_dir"],
        "template_similarity_threshold": pipe.cfg["preprocessing"]["template_similarity_threshold"],
        "roll_pattern": pipe.cfg["anonymisation"].get("roll_pattern", r"\b[A-Z]\d{3}\b"),
        "semantic_enabled": semantic_enabled,
        "onnx_dir": "models/all-MiniLM-L6-v2-onnx" if semantic_enabled else None,
        "encoder_name": None if pipe.sem_info is None else pipe.sem_info.get("model"),
        "concept_threshold": None if pipe.sem is None else pipe.sem.concept_threshold,
        "confidence_method": CONFIDENCE_METHOD,
        "variants": list(pipe.variant_models),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def compare_with_week5(pipe: Pipeline) -> dict:
    if not WEEK5_PREDICTIONS.is_file():
        raise SystemExit(f"Week 5 predictions are missing: {WEEK5_PREDICTIONS}")
    saved = pd.read_csv(WEEK5_PREDICTIONS)
    rows = []
    for _, item in pipe.manifest.iterrows():
        scored = pipe.score_notebook(pipe.data / item.raw_path, item.assignment_id)
        for variant, preds in scored["predictions"].items():
            if variant not in set(saved["variant"]):
                continue
            for dim in DIMS:
                match = saved[
                    (saved.notebook_id == item.notebook_id)
                    & (saved.variant == variant)
                    & (saved.dimension == dim)
                ]
                if match.empty:
                    continue
                got = float(preds[dim])
                old = float(match.predicted.iloc[0])
                rows.append({
                    "notebook_id": item.notebook_id,
                    "variant": variant,
                    "dimension": dim,
                    "week6": got,
                    "week5": old,
                    "abs_diff": abs(got - old),
                })
    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("No overlapping Week 5 predictions to compare.")
    summary = {
        "n": int(len(table)),
        "max_abs_diff": float(table.abs_diff.max()),
        "mean_abs_diff": float(table.abs_diff.mean()),
        "limit": MAX_ABS_DIFF,
    }
    print(
        f"[week6] agreement with Week 5 predictions: n={summary['n']} "
        f"max_abs_diff={summary['max_abs_diff']:.6f} mean_abs_diff={summary['mean_abs_diff']:.6f}"
    )
    worst = table.sort_values("abs_diff", ascending=False).head(5)
    print(worst.to_string(index=False))
    return summary


def write_tfidf_fallback_config() -> Path:
    """Used only when the ONNX check fails. Week 3 variants are omitted."""
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "week6.yaml").read_text(encoding="utf-8"))
    cfg["model"]["variants"] = ["week2_tfidf"]
    cfg["model"]["primary_variant"] = "week2_tfidf"
    cfg["model"]["compare"] = []
    cfg["semantic"]["enabled"] = False
    path = ROOT / "configs" / "week6_tfidf_fallback.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def main():
    semantic_ok = ensure_onnx()
    if semantic_ok:
        config = ROOT / "configs" / "week6.yaml"
    else:
        print("[week6] continuing with the TF-IDF asset only")
        config = write_tfidf_fallback_config()
    pipe = Pipeline(config)
    pipe.run()
    if semantic_ok and getattr(pipe, "sem_info", None) and pipe.sem_info.get("is_stub"):
        raise SystemExit("Stub encoder was loaded. Week 6 will not save those scores.")
    folder = save_asset(pipe, semantic_ok and pipe.sem is not None)
    if semantic_ok:
        summary = compare_with_week5(pipe)
        (folder / "week5_agreement.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if summary["max_abs_diff"] > MAX_ABS_DIFF:
            raise SystemExit(
                f"Week 6 predictions differ from Week 5 by {summary['max_abs_diff']:.4f} marks. "
                "Nothing was retuned. results/week5 was not modified."
            )
    print(f"[week6] asset saved to {folder}")


if __name__ == "__main__":
    main()
