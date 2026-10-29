"""Static (AST) and output-based checks for code cells. No code is executed here."""
from __future__ import annotations

import ast
import builtins
import io
import math
import re
import tokenize

MAGIC_LINE_RE = re.compile(r"^(\s*)([%!?].*)$")
ENV_ERRORS = {"ModuleNotFoundError", "ImportError", "FileNotFoundError", "PermissionError",
              "OSError", "MemoryError", "TimeoutError", "DeadKernelError", "CellTimeoutError"}
# calls too generic to count as "required constructs" from a reference solution
GENERIC_CALLS = set(dir(builtins)) | {"append", "items", "keys", "values", "get", "split", "join",
                                     "lower", "upper", "strip", "format", "update", "add", "extend"}


# ---------------------------------------------------------------- source handling
def strip_magics(src: str) -> str:
    """Comment out IPython magics / shell escapes so the cell can be parsed by ast."""
    lines = src.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        return "\n".join("# " + l for l in lines)  # whole-cell magic
    out = []
    for l in lines:
        m = MAGIC_LINE_RE.match(l)
        out.append(f"{m.group(1)}# {m.group(2)}" if m else l)
    return "\n".join(out)


def syntax_check(src: str) -> tuple[bool, str | None]:
    try:
        ast.parse(strip_magics(src))
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"


def extract_comments(src: str) -> list[str]:
    """Comments via the tokenizer, so '#' inside strings is not counted.
    Falls back to a line regex when the cell does not tokenize (e.g. syntax error)."""
    code = strip_magics(src)
    magic_lines = {i for i, l in enumerate(src.splitlines()) if MAGIC_LINE_RE.match(l)}
    comments = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT and (tok.start[0] - 1) not in magic_lines:
                comments.append(tok.string.lstrip("#").strip())
    except (tokenize.TokenError, IndentationError, SyntaxError):
        comments = [l.split("#", 1)[1].strip() for i, l in enumerate(src.splitlines())
                    if "#" in l and i not in magic_lines and l.strip().startswith("#")]
    return [c for c in comments if c]


def code_line_count(src: str) -> int:
    return sum(1 for l in src.splitlines() if l.strip() and not l.strip().startswith("#"))


def ast_summary(src: str) -> dict:
    """Constructs used in a cell. Empty summary if the cell does not parse."""
    summ = {"functions": [], "classes": [], "calls": set(), "imports": set(),
            "n_loops": 0, "n_docstrings": 0, "n_defs": 0}
    try:
        tree = ast.parse(strip_magics(src))
    except SyntaxError:
        return summ
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            (summ["classes"] if isinstance(node, ast.ClassDef) else summ["functions"]).append(node.name)
            summ["n_defs"] += 1
            if ast.get_docstring(node):
                summ["n_docstrings"] += 1
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name:
                summ["calls"].add(name)
        elif isinstance(node, ast.Import):
            summ["imports"].update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            summ["imports"].add(node.module.split(".")[0])
        elif isinstance(node, (ast.For, ast.While, ast.comprehension)):
            summ["n_loops"] += 1
    return summ


def required_constructs(reference_code: str) -> set[str]:
    """Non-generic calls + defined function names in a reference task solution."""
    s = ast_summary(reference_code)
    return {c for c in s["calls"] if c not in GENERIC_CALLS}


def classify_error(error_name: str | None) -> str | None:
    if not error_name:
        return None
    return "env" if error_name in ENV_ERRORS else "code"


# ---------------------------------------------------------------- output comparison
ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")
NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan")


def normalize_output(text: str) -> str:
    text = ADDR_RE.sub("0xADDR", text or "")
    return re.sub(r"\s+", " ", text).strip().lower()


def _numbers(text: str) -> list[float]:
    vals = []
    for t in NUM_RE.findall((text or "").lower()):
        try:
            vals.append(float(t))
        except ValueError:
            pass
    return vals


def compare_output(student_text: str, spec: dict) -> bool:
    """spec = {"value": str, "match": "exact"|"contains"|"numeric"|"regex", "tol": float}"""
    expected = spec["value"]
    mode = spec.get("match", "contains")
    s, e = normalize_output(student_text), normalize_output(expected)
    if mode == "regex":
        e = expected
    if not s:
        return False
    if mode == "exact":
        return s == e
    if mode == "contains":
        return e in s
    if mode == "regex":
        # value is a regular expression searched in the normalised (lower-case) output,
        # e.g. "(?<!fail to )reject the null" so that "fail to reject" does not count.
        return re.search(expected, s) is not None
    if mode == "numeric":
        tol = float(spec.get("tol", 1e-3))
        got = _numbers(student_text)
        want = _numbers(expected)
        if not want:
            return e in s
        return all(any((math.isinf(w) and math.isinf(g) and w == g) or abs(g - w) <= tol for g in got)
                   for w in want)
    raise ValueError(f"unknown match mode {mode}")
