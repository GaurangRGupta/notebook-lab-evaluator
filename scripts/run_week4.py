"""One command for the Week 4 baseline-vs-improved comparison (needs internet once for two models).

    pip install -r requirements.txt
    python -m ipykernel install --user
    python scripts/run_week4.py          # add --save-models models/ to keep offline copies

Steps: load MiniLM + NLI cross-encoder (cached after first download) -> full test suite (real-model tests
included) -> Week 4 pipeline on SYNTHETIC development data -> executed notebooks/week4_report.ipynb.
Then pin the printed revisions in configs/*.yaml (semantic.revision, nli.revision).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src.models.nli_consistency import CrossEncoderNLI
    from src.models.semantic_similarity import SentenceTransformerEncoder
    print("[1/4] loading models")
    enc = SentenceTransformerEncoder()
    nli = CrossEncoderNLI()
    print(f"      {enc.name}: dim={enc.dim} revision={enc.resolved_revision}")
    print(f"      {nli.name}: label order={nli.order} revision={nli.resolved_revision}")
    if "--save-models" in sys.argv:
        dest = ROOT / sys.argv[sys.argv.index("--save-models") + 1]
        enc.model.save(str(dest / "all-MiniLM-L6-v2"))
        nli.model.save(str(dest / "nli-deberta-v3-xsmall"))
        print(f"      saved local copies under {dest} (set semantic.local_path / nli.local_path)")
    print("[2/4] tests")
    if subprocess.call([sys.executable, "-m", "pytest", "-q", "-rs", "tests"], cwd=ROOT):
        sys.exit("tests failed; stopping")
    print("[3/4] Week 4 pipeline (synthetic development data)")
    from src.pipeline import Pipeline
    Pipeline(ROOT / "configs" / "synthetic_dev_week4.yaml").run()
    print("[4/4] executing notebooks/week4_report.ipynb")
    subprocess.check_call([sys.executable, "scripts/build_week4_report.py", "--execute"], cwd=ROOT)
    print("\nDone. Synthetic development results: not evidence of real student performance.")


if __name__ == "__main__":
    main()
