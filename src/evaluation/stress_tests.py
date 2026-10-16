"""Week 5 notebook-level stress tests (controlled robustness experiments).

Each test builds ONE notebook for LAB_SYN1 by perturbing a high-quality base notebook, executes
it with a real kernel where execution matters, and scores it with the FROZEN models fitted on the
v2 training split. Pass criteria are fixed in this file before the run (`CRITERIA`). These results
are reported separately and are never mixed into the held-out test metrics.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fx = _load("fx", ROOT / "scripts" / "make_synthetic_fixtures.py")
v2 = _load("v2", ROOT / "scripts" / "make_synthetic_v2.py")
LAB = "LAB_SYN1"
POOL = v2.EXPL


def _code(variant, task, comments=2, func="doc"):
    return v2.apply_traits(LAB, v2.CODE[LAB][task][variant], comments, func)


def base_fill(comments=2, func="doc"):
    return {"T1": {"code": _code("correct_a", "T1", comments, func)},
            "T2": {"code": _code("correct_a", "T2", comments, func), "md": POOL[(LAB, "T2")]["correct"][0]},
            "T3": {"code": _code("correct_a", "T3", comments, func), "md": POOL[(LAB, "T3")]["correct"][0]}}


def build(fill, func="doc", unexecuted=(), reorder=False, drop_task_code=None, extra_md=None):
    lab = fx.LABS[LAB]
    cells = fx.build_cells(lab, fill, "**Name:** [synthetic]\n\n**Roll No:** [synthetic]")
    if func != "none":
        i = next(j for j, c in enumerate(cells) if c.cell_type == "code" and c.source == fill["T1"]["code"])
        cells.insert(i, fx._code(v2.helper_prefix(LAB, func).rstrip()))
    for c in cells:
        if c.cell_type == "code" and any(c.source == fill[t]["code"] for t in unexecuted):
            c.metadata["_unexecuted"] = True
    if drop_task_code:
        cells = [c for c in cells if not (c.cell_type == "code" and c.source == fill[drop_task_code]["code"])]
    if extra_md:
        cells.append(fx._md(extra_md))
    if reorder:  # move the whole Task 3 block (header..end) before Task 1
        h1 = next(j for j, c in enumerate(cells) if c.cell_type == "markdown" and c.source.startswith("## Task 1"))
        h3 = next(j for j, c in enumerate(cells) if c.cell_type == "markdown" and c.source.startswith("## Task 3"))
        cells = cells[:h1] + cells[h3:] + cells[h1:h3]
    return fx.new_nb(cells)


def make_cases():
    P = POOL
    c = {}
    b = base_fill()
    c["BASE"] = ("High-quality base notebook (all tasks correct, comments, docstring, correct explanations)", build(b))
    f = base_fill(); f["T2"]["md"] = ""; f["T3"]["md"] = ""
    c["A_empty_explanation"] = ("Both explanation answers left empty", build(f))
    f = base_fill(); f["T2"]["md"] = "TF-IDF weights words."; f["T3"]["md"] = "Cosine compares vectors."
    c["B_very_short_explanation"] = ("Correct but very short (3-word) explanations", build(f))
    f = base_fill(); f["T2"]["md"] = P[(LAB, "T2")]["paraphrase"][0]; f["T3"]["md"] = P[(LAB, "T3")]["paraphrase"][0]
    c["C_paraphrased_explanation"] = ("Correct explanations written as paraphrases (little word overlap)", build(f))
    f = base_fill(); f["T2"]["md"] = P[(LAB, "T2")]["wrong"][0]; f["T3"]["md"] = P[(LAB, "T3")]["wrong"][0]
    c["D_E_correct_code_wrong_explanation"] = ("Correct code; wrong explanations that reuse many reference keywords", build(f))
    f = base_fill()
    for t in ("T1", "T2", "T3"):
        f[t]["code"] = _code("wrong", t)
    c["F_wrong_code_convincing_explanation"] = ("Wrong logic in every task; correct, convincing explanations", build(f))
    c["G_missing_output"] = ("Correct code but no cell was executed (no outputs)", build(base_fill(), unexecuted=("T1", "T2", "T3")))
    f = base_fill()
    for t in ("T1", "T2", "T3"):
        f[t]["code"] = _code("runtime", t)
    c["H_runtime_error"] = ("Runtime error (NameError) in every task", build(f))
    f = base_fill(); f["T3"]["code"] = _code("wrong", "T3"); f["T1"]["code"] = _code("wrong", "T1")
    c["I_output_mismatch"] = ("T1 and T3 run without error but print wrong values", build(f))
    f = base_fill()
    f["T2"]["md"] = P[(LAB, "T2")]["correct"][0] + " " + P[(LAB, "T2")]["offtopic"][0] + " My favourite football team won last night."
    c["J_unrelated_markdown"] = ("Correct explanation plus unrelated sentences, and an unrelated extra markdown cell",
                                 build(f, extra_md="## Notes\nI enjoyed the lab. The weather was great today."))
    c["K1_reordered_task_blocks"] = ("Task 3 block moved before Task 1 (headers kept)", build(base_fill(), reorder=True))
    c["K2_missing_code_cell"] = ("Task 2 code cell deleted", build(base_fill(), drop_task_code="T2"))
    f = base_fill()
    f["T2"]["md"] = ("**Tf–Idf** = $tf(t,d) \\times \\log \\frac{N}{df(t)}$: the `TFIDF` weight of a term is its term-frequency "
                     "times the log inverse-document-frequency, so words in most docs (e.g. *the*) are down-weighted and rare terms get more weight.")
    c["L_unusual_term_formatting"] = ("Correct explanation written with LaTeX, en-dash 'Tf–Idf', 'TFIDF', markdown emphasis", build(f))
    f = base_fill()
    f["T1"]["code"] = "import nonexistent_nlp_toolkit as nt\n" + f["T1"]["code"]
    f["T2"]["code"] = "extra = open('lab_data/corpus_extra.txt').read()\n" + f["T2"]["code"]
    c["M_missing_package_or_file"] = ("ModuleNotFoundError in T1 and FileNotFoundError in T2 (environment problems)", build(f))
    f = base_fill()
    f["T2"]["md"] = "### Explain: What does TF-IDF measure, and why does it down-weight words such as 'the'?"
    f["T3"]["md"] = "### Explain: What does cosine similarity measure, and why is it preferred over Euclidean distance for text?"
    c["N_template_text_as_answer"] = ("Student pasted the explanation PROMPTS as their answers", build(f))
    c["O_no_documentation"] = ("Correct code, no comments, no docstring, no explanations",
                               build({t: {"code": _code("correct_a", t, 0, "none"), **({"md": ""} if t != "T1" else {})}
                                      for t in ("T1", "T2", "T3")}, func="none"))
    return c


# Pre-specified expectations. Values compared with the BASE prediction of the same model.
CRITERIA = {
    "A_empty_explanation": ("explanation <= 2", lambda p, b: p["explanation"] <= 2),
    "B_very_short_explanation": ("explanation at least 2 below base", lambda p, b: p["explanation"] <= b["explanation"] - 2),
    "C_paraphrased_explanation": ("explanation within 2 of base (paraphrase is correct)", lambda p, b: abs(p["explanation"] - b["explanation"]) <= 2),
    "D_E_correct_code_wrong_explanation": ("explanation at least 4 below base AND code within 1 of base",
                                           lambda p, b: p["explanation"] <= b["explanation"] - 4 and abs(p["code"] - b["code"]) <= 1),
    "F_wrong_code_convincing_explanation": ("code <= 3 AND explanation within 2 of base",
                                            lambda p, b: p["code"] <= 3 and abs(p["explanation"] - b["explanation"]) <= 2),
    "G_missing_output": ("code at least 2 below base", lambda p, b: p["code"] <= b["code"] - 2),
    "H_runtime_error": ("code <= 2", lambda p, b: p["code"] <= 2),
    "I_output_mismatch": ("code at least 3 below base", lambda p, b: p["code"] <= b["code"] - 3),
    "J_unrelated_markdown": ("explanation within 2 of base", lambda p, b: abs(p["explanation"] - b["explanation"]) <= 2),
    "K1_reordered_task_blocks": ("all dimensions within 0.5 of base", lambda p, b: all(abs(p[d] - b[d]) <= 0.5 for d in ("code", "explanation", "documentation"))),
    "K2_missing_code_cell": ("code at least 2 below base", lambda p, b: p["code"] <= b["code"] - 2),
    "L_unusual_term_formatting": ("explanation within 2 of base", lambda p, b: abs(p["explanation"] - b["explanation"]) <= 2),
    "M_missing_package_or_file": ("code at least 2 below base", lambda p, b: p["code"] <= b["code"] - 2),
    "N_template_text_as_answer": ("explanation <= 2 (prompt text must not count as an answer)", lambda p, b: p["explanation"] <= 2),
    "O_no_documentation": ("documentation <= 1 AND code within 1 of base", lambda p, b: p["documentation"] <= 1 and abs(p["code"] - b["code"]) <= 1),
}
