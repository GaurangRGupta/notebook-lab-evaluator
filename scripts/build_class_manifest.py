"""Inventory the Managing Uncertainty (MU) class dataset and decide keep/drop per file.

Reads the raw dataset folder (student names in file and folder names), and writes:
  data/class/manifest.csv            tracked. Anonymous ids only, one row per file, with the
                                     keep/drop decision and the reason.
  data/class/raw/<LAB>/<id>.ipynb    gitignored. Byte copies of the kept notebooks, renamed.
  data/class/interim/id_map.csv      gitignored. Original path -> anonymous id.
  data/class/class_list.local.txt    gitignored. Student names for the anonymiser.

    NB_EVAL_SALT=... python scripts/build_class_manifest.py --src <MU folder> --roster roster.csv

roster.csv (kept outside git) has columns folder,roll,name,name_tokens  (name_tokens are
';'-separated words to scrub, e.g. "Gaurang;Gupta").
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASS = ROOT / "data" / "class"
GROUP3 = {"J013", "J015", "J016", "J017", "J018"}   # project team (Group 3): conflict of interest

# Lab detection from code content (file names are unreliable). Order matters.
LAB_RULES = [
    ("MU_L10", r"n1\s*\+\s*n2|x1|sample1|xbar1|mean1|two.?sample|ttest_ind|200\s*,\s*250|44\b.*28\b"),
    ("MU_L09", r"ttest_1samp|11\.8|t_stat|t\.ppf|critical"),
    ("MU_L07_08", r"binom|poisson|comb\("),
    ("MU_L06", r"kerala|FLOODS"),
    ("MU_L05", r"train\.csv|Survived|titanic"),
    ("MU_L02", r"population_by_county|np\.eye|linspace|arange\(1\s*,\s*26\)"),
    ("MU_L04", r"randint\(1\s*,\s*7\)|integers\(1\s*,\s*7\)|dice|die"),
    ("MU_L03", r"statistics|median|mode|\[1\s*,\s*3\s*,\s*4\s*,\s*5\s*,\s*7\s*,\s*9"),
    ("MU_L01", r"SUV|sedan|hatchback|rent"),
]
ROLL_RE = re.compile(r"(?<![A-Za-z0-9])[Jj]\s*0?(\d{2,3})(?!\d)")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_of(nb: dict) -> str:
    return "\n".join("".join(c.get("source", "")) for c in nb.get("cells", []) if c.get("cell_type") == "code")


def stats(nb: dict) -> dict:
    cs = nb.get("cells", [])
    code = [c for c in cs if c.get("cell_type") == "code" and "".join(c.get("source", "")).strip()]
    return {"n_code_cells": len(code),
            "n_with_output": sum(1 for c in code if c.get("outputs")),
            "n_error_outputs": sum(1 for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"),
            "n_markdown": sum(1 for c in cs if c.get("cell_type") == "markdown" and "".join(c.get("source", "")).strip())}


def detect_lab(nb: dict) -> str | None:
    text = code_of(nb) + "\n" + "\n".join("".join(c.get("source", "")) for c in nb.get("cells", []))
    for lab, rx in LAB_RULES:
        if re.search(rx, text, re.IGNORECASE):
            return lab
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--lab-overrides", type=Path, help="optional JSON {relative_path: lab}")
    a = ap.parse_args()
    salt = os.environ.get("NB_EVAL_SALT")
    if not salt:
        raise SystemExit("Set NB_EVAL_SALT (outside the repo) first.")
    roster = {r["folder"]: r for r in csv.DictReader(a.roster.open(encoding="utf-8"))}
    overrides = json.loads(a.lab_overrides.read_text()) if a.lab_overrides else {}

    def sid(roll: str) -> str:
        return "S" + hashlib.sha256((salt + roll).encode()).hexdigest()[:6]

    files = sorted(p for p in a.src.rglob("*") if p.is_file())
    rows, by_sha, by_code = [], defaultdict(list), defaultdict(list)
    for p in files:
        rel = p.relative_to(a.src).as_posix()
        folder = rel.split("/")[0] if "/" in rel else ""
        raw = p.read_bytes()
        r = {"rel": rel, "folder": folder, "sha": sha(raw), "size_kb": round(len(raw) / 1024, 1),
             "ext": p.suffix.lower()}
        if r["ext"] == ".ipynb":
            nb = json.loads(raw.decode("utf-8"))
            r.update(stats(nb))
            r["lab"] = overrides.get(rel) or detect_lab(nb)
            r["code_sha"] = sha(code_of(nb).encode())
            m = ROLL_RE.search(p.name)
            r["file_roll"] = f"J{int(m.group(1)):03d}" if m else ""
            r["roll"] = roster[folder]["roll"] if folder in roster else r["file_roll"]
            by_sha[r["sha"]].append(r)
            by_code[(r["lab"], r["code_sha"])].append(r)
        rows.append(r)

    # --- decisions -------------------------------------------------------------------------
    for r in rows:
        r["decision"], r["reason"], r["duplicate_of"], r["flags"] = "keep", "", "", []
        if r["ext"] != ".ipynb":
            r["decision"] = "drop"
            r["reason"] = {".csv": "data file, not a submission (used as reference data where relevant)",
                           ".xlsx": "data file, not a submission (used as reference data for MU_L02)",
                           ".docx": "Word document, not a notebook"}.get(r["ext"], "not a notebook")
            continue
        if not r["folder"]:
            r["decision"], r["reason"] = "drop", "root-level copy, byte-identical to a file in a student folder"
            continue
        if r["n_code_cells"] == 0:
            r["decision"], r["reason"] = "drop", "blank handout (no code); used as the MU_L02 template"
            continue
        if not r["lab"]:
            r["decision"], r["reason"] = "drop", "lab could not be identified"
    # same student, same lab, same code -> keep one (prefer the name without '(1)' / 'Copy')
    groups = defaultdict(list)
    for r in rows:
        if r["decision"] == "keep":
            groups[(r["roll"], r["lab"])].append(r)
    for (_, _), g in groups.items():
        if len(g) > 1:
            g.sort(key=lambda x: (bool(re.search(r"\(\d\)|copy", x["rel"], re.I)), x["rel"]))
            for x in g[1:]:
                x["decision"] = "drop"
                x["reason"] = ("duplicate of the same student's file (identical code)"
                               if x["code_sha"] == g[0]["code_sha"] else
                               "second submission for the same lab; first kept")
                x["dup_target"] = g[0]
    # cross-student identical files or identical code
    for (lab, _), g in by_code.items():
        live = [x for x in g if x["decision"] == "keep"]
        rolls = {x["roll"] for x in live}
        if len(live) > 1 and len(rolls) > 1:
            same_bytes = len({x["sha"] for x in live}) == 1
            # keep the copy whose file name carries the folder owner's roll
            owner = [x for x in live if x["file_roll"] and x["file_roll"] == x["roll"]]
            if same_bytes and len(owner) == 1:
                for x in live:
                    if x is not owner[0]:
                        x["decision"] = "drop"
                        x["reason"] = "byte-identical to another student's file (attribution conflict)"
                        x["dup_target"] = owner[0]
                owner[0]["flags"].append("identical file found in another student's folder")
            else:
                for x in live:
                    x["flags"].append("identical code to another student's notebook"
                                      if not same_bytes else "byte-identical to another student's file")
    # --- ids, copies, outputs ------------------------------------------------------------------
    (CLASS / "interim").mkdir(parents=True, exist_ok=True)
    counter, idmap, out = defaultdict(int), [], []
    # kept notebooks get the plain ids first, so a dropped copy never takes a keeper's id
    for r in sorted(rows, key=lambda x: (x["decision"] != "keep", x["rel"])):
        if r["ext"] == ".ipynb" and r.get("roll") and r["folder"]:
            r["student_id"] = sid(r["roll"])
            counter[(r["lab"], r["student_id"])] += 1
            k = counter[(r["lab"], r["student_id"])]
            r["notebook_id"] = f"{r['lab']}_{r['student_id']}" + (f"_{k}" if k > 1 else "")
        else:
            r["student_id"] = ""
            counter[("file", "")] += 1
            r["notebook_id"] = f"FILE_{counter[('file', '')]:02d}"
        if r["decision"] != "keep" and r["ext"] == ".ipynb" and not r["notebook_id"].startswith("FILE"):
            r["notebook_id"] = "DROP_" + r["notebook_id"]
        idmap.append({"original_path": r["rel"], "notebook_id": r["notebook_id"], "roll": r.get("roll", "")})
    for r in rows:
        if r.get("dup_target"):
            r["duplicate_of"] = r["dup_target"]["notebook_id"]
        if r["decision"] == "keep":
            dst = CLASS / "raw" / r["lab"] / f"{r['notebook_id']}.ipynb"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(a.src / r["rel"], dst)
            r["raw_path"] = dst.relative_to(CLASS).as_posix()
            if r["n_error_outputs"]:
                r["flags"].append(f"{r['n_error_outputs']} error output(s) saved")
            if r["n_markdown"] == 0:
                r["flags"].append("no markdown cells: explanation cannot be assessed")
            if r.get("roll") in GROUP3:
                r["flags"].append("author is a project team member: they must not mark it")
        out.append({
            "notebook_id": r["notebook_id"], "raw_path": r.get("raw_path", ""),
            "assignment_id": r.get("lab") or "", "group_key": r["student_id"],
            "source": "MU class dataset 2026 (shared by Group 3 members)",
            "license": "course use only; not for redistribution",
            "file_type": r["ext"].lstrip("."), "size_kb": r["size_kb"],
            "n_code_cells": r.get("n_code_cells", ""), "n_with_output": r.get("n_with_output", ""),
            "n_error_outputs": r.get("n_error_outputs", ""), "n_markdown": r.get("n_markdown", ""),
            "decision": r["decision"], "reason": r["reason"] or "complete submission for an identified lab",
            "duplicate_of": r["duplicate_of"], "flags": "; ".join(r["flags"]),
            "team_member_author": "yes" if r.get("roll") in GROUP3 else ("no" if r.get("roll") else ""),
        })
    with (CLASS / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    with (CLASS / "interim" / "id_map.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["original_path", "notebook_id", "roll"])
        w.writeheader(); w.writerows(idmap)
    names = sorted({t for r in roster.values() for t in r["name_tokens"].split(";") if t.strip()})
    (CLASS / "class_list.local.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    kept = [o for o in out if o["decision"] == "keep"]
    print(f"{len(out)} files: {len(kept)} kept, {len(out) - len(kept)} dropped")
    per = defaultdict(int)
    for o in kept:
        per[o["assignment_id"]] += 1
    print(dict(sorted(per.items())))


if __name__ == "__main__":
    main()
