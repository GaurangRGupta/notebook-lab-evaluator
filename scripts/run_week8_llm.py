"""Week 8: Approach C, rubric-based LLM evaluation, compared with week2_tfidf and week3_combined.

Same notebooks and labels as results/week5 (synthetic v2, rule-derived labels, split seed 42).
Primary view: the 12 held-out TEST notebooks (identical rows to results/week5/predictions.csv).
Supplementary view: all 60 v2 notebooks (the LLM is not trained, so train/val notebooks are also
unseen by it; the Ridge baselines are not comparable there and are not reported for that view).
Responses are cached in results/week8_llm/cache, so a rerun costs nothing.

    python scripts/run_week8_llm.py            # needs XAI_API_KEY unless every call is cached
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.references import load_reference  # noqa: E402
from src.evaluation.metrics import bootstrap_ci, regression_metrics  # noqa: E402
from src.models import llm_evaluator  # noqa: E402
from src.models.features import static_features  # noqa: E402
from src.preprocessing.align_tasks import assign_tasks, flag_template_cells  # noqa: E402
from src.preprocessing.parse_notebook import parse_notebook_file  # noqa: E402

DATA = ROOT / "data" / "synthetic_v2"
RUN = ROOT / "results" / "week5" / "v2_run"
W5 = ROOT / "results" / "week5"
OUT = ROOT / "results" / "week8_llm"
CACHE = OUT / "cache"
DIMS = ["code", "explanation", "documentation"]
MAX = {"code": 8, "explanation": 8, "documentation": 4, "overall": 20}
BUDGET_USD = 3.0
CONSISTENCY_RUNS = ["rerun_1", "rerun_2"]
CONSISTENCY_N = 6

_spec = importlib.util.spec_from_file_location("run_week5", ROOT / "scripts" / "run_week5.py")
_w5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_w5)
_paired_row = _w5._paired_row


def notebook_inputs(nid, assignment_id, refs):
    ref = refs[assignment_id]
    _, cells = parse_notebook_file(DATA / "interim" / f"{nid}.ipynb", nid)
    cells = flag_template_cells(assign_tasks(cells), ref.template_cells, 0.9)
    _, diag = static_features(cells, ref)
    errors = (list(diag["errors"]) + [f"output mismatch in {t}" for t in diag["output_mismatch"]]
              + [f"cell {i} was never executed" for i in diag["unexecuted_cells"]])
    return cells, ref, errors


def grade(nid, aid, refs, spent, run_tag="", complete=None):
    cells, ref, errors = notebook_inputs(nid, aid, refs)
    rep = llm_evaluator.maybe_grade(cells, ref, errors, complete=complete, cache_dir=CACHE, force=True, run_tag=run_tag)
    if rep is None or rep.get("status") != "ok":
        raise SystemExit(f"LLM call failed for {nid}: {rep}")
    if not rep.get("cached"):
        spent["usd"] += rep.get("cost_usd") or 0.0
        spent["calls"] += 1
        if spent["usd"] > BUDGET_USD:
            raise SystemExit(f"budget guard: spent {spent['usd']:.4f} USD > {BUDGET_USD}")
    return rep


def main(complete=None):
    OUT.mkdir(parents=True, exist_ok=True)
    idmap = pd.read_csv(DATA / "interim" / "id_map.csv")
    man = pd.read_csv(DATA / "manifest.csv").merge(idmap, on="raw_path")
    split = json.loads((RUN / "split.json").read_text())["assignment"]
    labels = pd.read_csv(RUN / "labels_final.csv").set_index("notebook_id")
    man["split"] = man.notebook_id.map(split)
    refs = {a: load_reference(DATA / "references", a) for a in man.assignment_id.unique()}
    spent = {"usd": 0.0, "calls": 0}

    rows, reports = [], {}
    for _, r in man.iterrows():
        rep = grade(r.notebook_id, r.assignment_id, refs, spent, complete=complete)
        reports[r.notebook_id] = rep
        for d in DIMS + ["overall"]:
            human = float(labels.loc[r.notebook_id, DIMS].sum()) if d == "overall" else float(labels.loc[r.notebook_id, d])
            rows.append({"variant": "week8_llm", "notebook_id": r.notebook_id, "split": r.split, "dimension": d,
                         "human": human, "predicted": float(rep[d]), "assignment_id": r.assignment_id,
                         "group": r.group_key, "profile": r.profile})
    llm = pd.DataFrame(rows)
    llm["error"] = llm.predicted - llm.human
    llm["abs_error"] = llm.error.abs()
    llm.to_csv(OUT / "llm_predictions.csv", index=False)

    w5 = pd.read_csv(W5 / "predictions.csv")
    w5 = w5[(w5.dataset == "synthetic_v2") & (w5.split == "test") & w5.variant.isin(["week2_tfidf", "week3_combined"])]
    test = pd.concat([w5[["variant", "notebook_id", "split", "dimension", "human", "predicted"]],
                      llm[llm.split == "test"][["variant", "notebook_id", "split", "dimension", "human", "predicted"]]])
    # sanity: labels used here equal the labels in week5
    chk = test.pivot_table(index=["notebook_id", "dimension"], columns="variant", values="human")
    assert np.allclose(chk.week8_llm, chk.week2_tfidf), "label mismatch with week5"

    mrows = []
    views = {"test": test}
    views["all_v2_llm_only"] = llm[["variant", "notebook_id", "split", "dimension", "human", "predicted"]]
    for view, df in views.items():
        for v, g in df.groupby("variant"):
            for d in DIMS + ["overall"]:
                s = g[g.dimension == d]
                m = regression_metrics(s.human, s.predicted, MAX[d])
                ci = bootstrap_ci(s.human, s.predicted, "MAE")
                rci = bootstrap_ci(s.human, s.predicted, "RMSE")
                mrows.append({"view": view, "variant": v, "dimension": d, **m,
                              "MAE_ci95_low": ci[0] if ci else None, "MAE_ci95_high": ci[1] if ci else None,
                              "RMSE_ci95_low": rci[0] if rci else None, "RMSE_ci95_high": rci[1] if rci else None})
    pd.DataFrame(mrows).to_csv(OUT / "metrics.csv", index=False)

    from scipy.stats import binomtest
    prow = []
    for base in ["week2_tfidf", "week3_combined"]:
        for d in DIMS + ["overall"]:
            a = test[(test.variant == base) & (test.dimension == d)].set_index("notebook_id").sort_index()
            b = test[(test.variant == "week8_llm") & (test.dimension == d)].set_index("notebook_id").loc[a.index]
            row = _paired_row("test", d, a.human.to_numpy(), a.predicted.to_numpy(), b.predicted.to_numpy(), 5000, 0, binomtest)
            prow.append({"baseline": base, "improved": "week8_llm", **row})
    pd.DataFrame(prow).to_csv(OUT / "paired_comparison.csv", index=False)

    # run-to-run consistency: the first CONSISTENCY_N test notebooks, sorted by id, graded again
    sub = sorted(man[man.split == "test"].notebook_id)[:CONSISTENCY_N]
    crows = []
    for nid in sub:
        aid = man.set_index("notebook_id").loc[nid, "assignment_id"]
        runs = {"run_0": reports[nid]}
        for tag in CONSISTENCY_RUNS:
            runs[tag] = grade(nid, aid, refs, spent, run_tag=tag, complete=complete)
        for tag, rep in runs.items():
            crows.append({"notebook_id": nid, "run": tag, **{d: rep[d] for d in DIMS + ["overall"]}})
    cons = pd.DataFrame(crows)
    cons.to_csv(OUT / "consistency_runs.csv", index=False)
    spread = cons.groupby("notebook_id")[DIMS + ["overall"]].agg(lambda x: float(x.max() - x.min()))
    cons_summary = {"notebooks": len(sub), "runs_per_notebook": 1 + len(CONSISTENCY_RUNS), "temperature": 0,
                    "max_range_per_dimension": {d: float(spread[d].max()) for d in spread},
                    "mean_range_per_dimension": {d: float(spread[d].mean()) for d in spread},
                    "notebooks_identical_on_all_dimensions": int((spread.max(axis=1) == 0).sum())}

    # error analysis: largest absolute overall errors on the test split, plus explanation errors
    t = llm[llm.split == "test"]
    ea = []
    for d in ["overall", "explanation", "code", "documentation"]:
        for _, r in t[t.dimension == d].sort_values("abs_error", ascending=False).head(4).iterrows():
            rep = reports[r.notebook_id]
            base = w5[(w5.notebook_id == r.notebook_id) & (w5.dimension == d)].set_index("variant").predicted
            ea.append({"dimension": d, "notebook_id": r.notebook_id, "assignment_id": r.assignment_id,
                       "profile": r.profile, "human": r.human, "llm": r.predicted, "llm_error": r.error,
                       "week2_tfidf": float(base.get("week2_tfidf", np.nan)), "week3_combined": float(base.get("week3_combined", np.nan)),
                       "llm_justification": rep.get("justification", {}).get(d, rep.get("feedback", "")) if d != "overall" else rep.get("feedback", "")})
    pd.DataFrame(ea).to_csv(OUT / "error_cases.csv", index=False)

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls_in_cache": 0}
    for f in CACHE.glob("*.json"):
        u = json.loads(f.read_text()).get("usage") or {}
        usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage["completion_tokens"] += u.get("completion_tokens", 0)
        usage["calls_in_cache"] += 1
    usage["model"] = llm_evaluator.MODEL
    usage["price_usd_per_1M"] = llm_evaluator.PRICES.get(llm_evaluator.MODEL)
    usage["total_cost_usd_all_cached_calls"] = llm_evaluator.cost_usd(usage)
    usage["new_calls_this_run"] = spent["calls"]
    usage["new_cost_usd_this_run"] = spent["usd"]
    (OUT / "usage.json").write_text(json.dumps(usage, indent=2))
    (OUT / "consistency_summary.json").write_text(json.dumps(cons_summary, indent=2))
    (OUT / "run_config.json").write_text(json.dumps({
        "model": llm_evaluator.MODEL, "temperature": 0, "base_url": llm_evaluator.BASE_URL,
        "labels": "results/week5/v2_run/labels_final.csv (rule-derived, synthetic)",
        "split": "results/week5/v2_run/split.json (seed 42)", "bootstrap": "bootstrap_ci n_boot=2000 seed=0; paired n_boot=5000 seed=0",
        "data_status": "SYNTHETIC_DEVELOPMENT_DATA_V2_RULE_LABELS"}, indent=2))
    print(json.dumps({"usage": usage, "consistency": cons_summary}, indent=2))


if __name__ == "__main__":
    main()
