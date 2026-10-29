"""Copy three anonymised class notebooks into demo/samples/class/ (gitignored) and save the
JSON fallbacks for every demo sample into demo/fallback/.

    python scripts/prepare_demo_samples.py      # after scripts/run_class.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.api.app import SAMPLES, SAMPLES_DIR  # noqa: E402
from src.asset import evaluate_notebook, load_asset  # noqa: E402
from src.integration import support_signal  # noqa: E402

# Authors of these three are not project team members (see data/class/manifest.csv).
CLASS_PICKS = {"class/MU_L06_sample.ipynb": "MU_L06_S3cd6d6",
               "class/MU_L07_08_sample.ipynb": "MU_L07_08_S3cd6d6",
               "class/MU_L01_sample.ipynb": "MU_L01_S95158b"}


def main():
    interim = ROOT / "data" / "class" / "interim"
    (SAMPLES_DIR / "class").mkdir(parents=True, exist_ok=True)
    for dst, nid in CLASS_PICKS.items():
        src = interim / f"{nid}.ipynb"
        if src.is_file():
            shutil.copy(src, SAMPLES_DIR / dst)
        else:
            print(f"skip {dst}: {src.name} not found (run scripts/run_class.py first)")
    asset = load_asset()
    fb = ROOT / "demo" / "fallback"
    fb.mkdir(parents=True, exist_ok=True)
    for name, (lab, _) in SAMPLES.items():
        p = SAMPLES_DIR / name
        if not p.is_file():
            continue
        rep = evaluate_notebook(p.read_bytes(), lab, asset)
        stem = Path(name).stem
        (fb / f"{stem}.report.json").write_text(json.dumps(rep, indent=2))
        (fb / f"{stem}.support.json").write_text(json.dumps(support_signal(rep), indent=2))
        print(f"{name}: overall {rep['overall']} / {rep['max_score']}")


if __name__ == "__main__":
    main()
