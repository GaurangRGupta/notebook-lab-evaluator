"""One command to produce the real Week 3 development results on a laptop with internet access.

    pip install -r requirements.txt
    python -m ipykernel install --user
    python scripts/run_week3.py                       # add --save-model models/all-MiniLM-L6-v2 for offline reuse

Steps:
  1. download (first time only, ~90 MB) and load all-MiniLM-L6-v2; print dimension and resolved revision
  2. run the full test suite, including the real-model tests (they are skipped when the model is absent)
  3. run the Week 3 pipeline on the SYNTHETIC development data -> results/synthetic_dev_week3/
  4. execute notebooks/week3_report.ipynb with the real model
Afterwards, pin the printed revision in configs/*.yaml (semantic.revision) for reproducibility.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src.models.semantic_similarity import SentenceTransformerEncoder
    print("[1/4] loading sentence-transformers/all-MiniLM-L6-v2 (downloads once, then cached)")
    enc = SentenceTransformerEncoder()
    n = sum(p.numel() for p in enc.model.parameters())
    print(f"      dim={enc.dim} max_tokens={enc.max_tokens} params={n:,} revision={enc.resolved_revision}")
    if "--save-model" in sys.argv:
        dest = ROOT / sys.argv[sys.argv.index("--save-model") + 1]
        enc.model.save(str(dest))
        print(f"      saved a local copy to {dest} (set semantic.local_path to use it offline)")
    print("[2/4] tests")
    rc = subprocess.call([sys.executable, "-m", "pytest", "-q", "-rs", "tests"], cwd=ROOT)
    if rc:
        sys.exit("tests failed; stopping")
    print("[3/4] Week 3 pipeline (synthetic development data)")
    from src.pipeline import Pipeline
    Pipeline(ROOT / "configs" / "synthetic_dev_week3.yaml").run()
    print("[4/4] executing notebooks/week3_report.ipynb")
    subprocess.check_call([sys.executable, "scripts/build_week3_report.py", "--execute"], cwd=ROOT)
    print("\nDone. Synthetic development run: results are not evidence of real student performance.")


if __name__ == "__main__":
    main()
