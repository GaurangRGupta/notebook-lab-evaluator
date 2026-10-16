"""End-to-end Week 2 pipeline, driven entirely by a YAML config.

Swapping synthetic data for the real class dataset = pointing the config at a
different data folder with the same layout. No code changes are needed.

Usage:  python -m src.pipeline configs/synthetic_dev.yaml
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import yaml

from .data.references import load_reference
from .data.split import grouped_split, merge_near_duplicates
from .evaluation.iaa import aggregate_labels, agreement_report
from .evaluation.metrics import bootstrap_ci, regression_metrics
from .models.baseline import DimensionModel, MeanBaseline
from .models.features import (
    DEFAULT_FEATURES,
    TfidfSimilarity,
    explanation_texts,
    static_features,
    variant_features,
)
from .reporting import _finite, choose_missing, confidence_from_variants, feedback_and_flag, marks
from .preprocessing.align_tasks import assign_tasks, flag_template_cells
from .preprocessing.anonymize import EMAIL_RE, Scrubber, load_class_list, notebook_hash
from .preprocessing.parse_notebook import parse_notebook_file

DIMS = ["code", "explanation", "documentation"]


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _dump(obj, path: Path, status: str):
    path.write_text(json.dumps(_jsonable({"data_status": status, **obj}), indent=2))


class Pipeline:
    def __init__(self, config_path, overrides: dict | None = None):
        self.cfg_path = Path(config_path)
        self.cfg = yaml.safe_load(self.cfg_path.read_text())
        for k, v in (overrides or {}).items():  # shallow-merge dict sections (used by tests)
            self.cfg[k] = {**self.cfg[k], **v} if isinstance(v, dict) and isinstance(self.cfg.get(k), dict) else v
        root = self.cfg_path.resolve().parents[1]
        self.data = (root / self.cfg["data_dir"]).resolve()
        self.out = (root / self.cfg["results_dir"]).resolve()
        self.interim = (root / self.cfg["interim_dir"]).resolve()
        self.status = self.cfg["data_status"]
        self.log = []

    def say(self, msg):
        self.log.append(msg)
        print(msg)

    # ---------------------------------------------------------------- 1. anonymise
    def anonymise(self):
        c = dict(self.cfg["anonymisation"])
        m = re.fullmatch(r"\$\{(\w+)\}", str(c["salt"]).strip())
        if m:  # real data: salt comes from the environment, never from the repo
            import os
            if not os.environ.get(m.group(1)):
                raise SystemExit(f"Set environment variable {m.group(1)} (anonymisation salt) before running.")
            c["salt"] = os.environ[m.group(1)]
        scrub = Scrubber(c.get("roll_pattern", r"\b[A-Z]\d{3}\b"),
                         load_class_list(self.data / c["class_list"]) if c.get("class_list") else [])
        self.interim.mkdir(parents=True, exist_ok=True)
        man = pd.read_csv(self.data / self.cfg["manifest"])
        ids, residual = [], 0
        roll_re = re.compile(c.get("roll_pattern", r"\b[A-Z]\d{3}\b"))
        self.skipped = []
        keep = []
        for _, r in man.iterrows():
            try:  # Week 5 fix: one unreadable submission must not crash the whole run
                nb = nbformat.read(str(self.data / r["raw_path"]), as_version=4)
                nbformat.validate(nb)
            except Exception as e:
                self.skipped.append({"raw_path": r["raw_path"], "error": f"{type(e).__name__}: {str(e)[:200]}"})
                continue
            keep.append(r.name)
            clean = scrub.scrub_notebook(nb)
            nid = notebook_hash(r["raw_path"], c["salt"])
            nbformat.write(clean, self.interim / f"{nid}.ipynb")
            text = nbformat.writes(clean)
            residual += len(roll_re.findall(text)) + len(EMAIL_RE.findall(text))
            ids.append(nid)
        man = man.loc[keep].reset_index(drop=True)
        man["notebook_id"] = ids
        self.manifest = man
        if self.skipped:
            pd.DataFrame(self.skipped).to_csv(self.out / "skipped_notebooks.csv", index=False)
            self.say(f"[anonymise] SKIPPED {len(self.skipped)} unreadable notebook(s): see skipped_notebooks.csv")
        # id map links raw files to hashes; for real data it must be stored outside the repo
        man[["raw_path", "notebook_id"]].to_csv(self.interim / "id_map.csv", index=False)
        self.say(f"[anonymise] {len(man)} notebooks, residual PII pattern matches after scrub: {residual}")
        self.pii_residual = residual

    # ---------------------------------------------------------------- 2. parse + clean
    def parse(self):
        self.refs = {a: load_reference(self.data / self.cfg["references_dir"], a)
                     for a in self.manifest["assignment_id"].unique()}
        for a, ref in self.refs.items():
            for w in ref.warnings:
                self.say(f"[references] {a}: WARNING {w}")
        thr = self.cfg["preprocessing"]["template_similarity_threshold"]
        all_cells, self.nb_cells = [], {}
        for _, r in self.manifest.iterrows():
            meta, cells = parse_notebook_file(self.interim / f"{r.notebook_id}.ipynb", r.notebook_id)
            cells = flag_template_cells(assign_tasks(cells), self.refs[r.assignment_id].template_cells, thr)
            for c in cells:
                c["assignment_id"] = r.assignment_id
            self.nb_cells[r.notebook_id] = cells
            all_cells += cells
        self.cells = pd.DataFrame(all_cells)
        self.cells.to_csv(self.out / "cells.csv", index=False)
        self.say(f"[parse] {len(self.cells)} cell records "
                 f"({(self.cells.cell_type == 'code').sum()} code, {(self.cells.cell_type == 'markdown').sum()} markdown), "
                 f"{int(self.cells.is_template.sum())} template cells flagged")

    # ---------------------------------------------------------------- 3. static features
    def featurise(self):
        rows, self.diag, self.expl = [], {}, {}
        for _, r in self.manifest.iterrows():
            ref = self.refs[r.assignment_id]
            f, d = static_features(self.nb_cells[r.notebook_id], ref)
            self.diag[r.notebook_id] = d
            self.expl[r.notebook_id] = explanation_texts(self.nb_cells[r.notebook_id], ref)
            rows.append({"notebook_id": r.notebook_id, "assignment_id": r.assignment_id, **f})
        self.features = pd.DataFrame(rows)

    # ---------------------------------------------------------------- 4. labels
    def labels(self):
        ann = pd.read_csv(self.data / self.cfg["annotations"]["raw"])
        ann = ann.merge(self.manifest[["raw_path", "notebook_id"]], on="raw_path", how="inner")
        maxes = self.cfg["rubric_max"]
        self.iaa = {d: agreement_report(ann, d, maxes[d]) for d in DIMS}
        adj_path = self.cfg["annotations"].get("adjudicated")
        adj = pd.read_csv(self.data / adj_path) if adj_path and (self.data / adj_path).exists() else None
        if adj is not None and "raw_path" in adj:
            adj = adj.merge(self.manifest[["raw_path", "notebook_id"]], on="raw_path")
        self.y = aggregate_labels(ann, DIMS, tolerance=self.cfg["annotations"]["discussion_tolerance"],
                                  adjudicated=adj)
        n_flag = int(self.y[[f"{d}_needs_discussion" for d in DIMS]].any(axis=1).sum())
        self.y.to_csv(self.out / "labels_final.csv", index=False)
        _dump({"iaa": self.iaa, "items_flagged_for_discussion": n_flag}, self.out / "iaa.json", self.status)
        self.say(f"[labels] {len(self.y)} labelled notebooks, {n_flag} flagged for discussion")

    # ---------------------------------------------------------------- 5. split
    def split(self):
        sc = self.cfg["split"]
        man = self.manifest
        texts = [" ".join(c["source"] for c in self.nb_cells[n] if not c.get("is_template") and c.get("task_id"))
                 for n in man.notebook_id]
        groups = merge_near_duplicates(list(man.notebook_id), texts, list(man.group_key.astype(str)),
                                       sc["near_duplicate_threshold"])
        strata = (man.assignment_id + "|" + man.source).tolist()
        self.split_info = grouped_split(man.notebook_id, groups, strata, tuple(sc["target"]), sc["seed"])
        self.split_info["merged_groups"] = int(man.group_key.nunique() - len(set(groups)))
        self.manifest["split"] = man.notebook_id.map(self.split_info["assignment"])
        self.manifest["group"] = groups
        self.cells["split"] = self.cells.notebook_id.map(self.split_info["assignment"])
        self.cells.to_csv(self.out / "cells.csv", index=False)
        _dump(self.split_info, self.out / "split.json", self.status)
        self.say(f"[split] {self.split_info['method']}; realised {self.split_info['realised']} "
                 f"(target {self.split_info['target']})")

    # ---------------------------------------------------------------- 5b. semantic features (Week 3)
    def semantic(self):
        """Frozen pretrained encoder: per-notebook features computed independently of the split,
        so no statistics are learned from validation or test notebooks."""
        sc = self.cfg.get("semantic") or {}
        self.sem = None
        self.sem_info = None
        if not sc.get("enabled"):
            return
        from .models.semantic_similarity import SemanticSimilarity, build_encoder, set_seeds
        set_seeds(sc.get("seed", 42))
        sc = dict(sc)
        if sc.get("onnx_dir") and not Path(sc["onnx_dir"]).is_absolute():
            sc["onnx_dir"] = str(self.cfg_path.resolve().parents[1] / sc["onnx_dir"])
        enc = build_encoder(sc)
        thr = sc.get("concept_threshold", 0.5)
        sem = SemanticSimilarity(enc, 0.5)
        if thr == "calibrate":
            from .evaluation.semantic_eval import calibrate_concept_threshold, load_concept_pairs
            cal = calibrate_concept_threshold(sem, load_concept_pairs(self.data / sc["calibration_pairs"]))
            thr = cal["threshold"]
            _dump({"encoder": enc.name, **cal}, self.out / "concept_calibration.json", self.status)
            self.say(f"[semantic] concept threshold {thr} chosen on authored tune pairs (never on notebooks)")
        sem.concept_threshold = float(thr)
        self.sem = sem
        self.sem_info = sem.info()
        rows, self.sem_diag = [], {}
        for n, a in zip(self.manifest.notebook_id, self.manifest.assignment_id):
            f, d = sem.notebook_features(self.expl[n], self.refs[a])
            rows.append({"notebook_id": n, **f})
            self.sem_diag[n] = d
        self.sem_features = pd.DataFrame(rows)
        if sc.get("probes"):
            from .evaluation.semantic_eval import probe_separation, run_probes
            pr = run_probes(sem, self.data / sc["probes"])
            pr.to_csv(self.out / "probe_results.csv", index=False)
            _dump({"encoder": enc.name, "separation": probe_separation(pr)}, self.out / "probe_summary.json", self.status)
        tag = "STUB ENCODER (lexical stand-in, NOT semantic)" if enc.is_stub else f"{enc.name} @ {enc.resolved_revision}"
        self.say(f"[semantic] {tag}: dim={enc.dim}, frozen, {len(rows)} notebooks embedded")

    # ---------------------------------------------------------------- 5c. NLI consistency (Week 4)
    def nli(self):
        """Frozen pretrained NLI cross-encoder; per-notebook features independent of the split."""
        nc = self.cfg.get("nli") or {}
        self.nli_model, self.nli_info = None, None
        if not nc.get("enabled"):
            return
        from .models.nli_consistency import NLIConsistency, build_nli
        scorer = NLIConsistency(build_nli(nc), nc.get("flag_threshold", 0.5))
        self.nli_model, self.nli_info = scorer, scorer.info()
        rows, self.nli_diag = [], {}
        for n, a in zip(self.manifest.notebook_id, self.manifest.assignment_id):
            f, d = scorer.notebook_features(self.expl[n], self.refs[a])
            rows.append({"notebook_id": n, **f})
            self.nli_diag[n] = d
        self.nli_features = pd.DataFrame(rows)
        tag = "STUB NLI (lexical stand-in, NOT a model)" if scorer.nli.is_stub else \
            f"{scorer.nli.name} @ {scorer.nli.resolved_revision}"
        self.say(f"[nli] {tag}: frozen, {len(rows)} notebooks scored")

    # ---------------------------------------------------------------- 5d. behaviour checks (Week 4)
    def behaviour_checks(self):
        """Probe pairs and robustness variants scored by every available scorer (no notebooks used)."""
        rb = (self.cfg.get("robustness") or {}).get("suite")
        sem, nli = getattr(self, "sem", None), getattr(self, "nli_model", None)
        if rb:
            from .evaluation.robustness import run_robustness, summarise
            df = run_robustness(self.data / rb, sem, nli)
            df.to_csv(self.out / "robustness.csv", index=False)
            summarise(df).to_csv(self.out / "robustness_summary.csv")
            self.say(f"[robustness] {len(df)} variant/scorer rows written")
        pp = self.out / "probe_results.csv"
        if nli is not None and pp.exists():
            pr = pd.read_csv(pp)
            sc = [nli.task_scores(s, r) for s, r in zip(pr.student, pr.reference)]
            pr["nli_support"] = [round(x["support"], 3) for x in sc]
            pr["nli_contradiction"] = [round(x["contradiction"], 3) for x in sc]
            pr.to_csv(pp, index=False)

    # ---------------------------------------------------------------- 6-7. model + evaluation
    def _variants(self) -> dict:
        from .models.features import variant_features
        names = (self.cfg.get("model") or {}).get("variants") or ["week2_tfidf"]
        if self.cfg.get("features"):  # explicit override keeps the Week 2 behaviour
            return {names[0]: {d: self.cfg["features"].get(d, DEFAULT_FEATURES[d]) for d in DIMS}}
        return {v: variant_features(v) for v in names}

    def _estimator(self, variant, d):
        return (self.estimators.get(variant) or {}).get(d, "ridge")

    def _fit_eval(self, feats, parts, record_preds, variant):
        from .models.improved import make_dimension_model
        results, preds, models = {}, [], {}
        sums = {s: {"human": 0.0, "pred": 0.0, "mean": 0.0} for s in ("val", "test")}
        for d in DIMS:
            mx = self.cfg["rubric_max"][d]
            est = self._estimator(variant, d)
            m = make_dimension_model(est, d, feats[d], mx, self.cfg["model"]["alpha_grid"]).fit(
                parts["train"], parts["train"][d].to_numpy(float), parts["val"], parts["val"][d].to_numpy(float))
            models[d] = m
            mean = MeanBaseline(mx).fit(parts["train"][d].to_numpy(float))
            res = {"features": feats[d], "estimator": est, "alpha": m.alpha, "alpha_selection": m.selection,
                   "coefficients": m.coefficients()}
            for s in ("val", "test") if self.cfg["model"]["evaluate_test"] else ("val",):
                P = parts[s]
                if not len(P):
                    continue
                yt, yp = P[d].to_numpy(float), m.predict(P)
                mp = mean.predict(len(P))
                res[s] = {"model": regression_metrics(yt, yp, mx),
                          "mean_predictor": regression_metrics(yt, mp, mx),
                          "MAE_95CI": bootstrap_ci(yt, yp, "MAE")}
                if est == "ridge":
                    res[s]["ridge"] = res[s]["model"]  # backward-compatible key (Week 2/3 reports)
                sums[s]["human"] = sums[s]["human"] + yt
                sums[s]["pred"] = sums[s]["pred"] + yp
                sums[s]["mean"] = sums[s]["mean"] + mp
                if record_preds:
                    preds += [{"variant": variant, "notebook_id": n, "split": s, "dimension": d,
                               "human": t, "predicted": p} for n, t, p in zip(P.notebook_id, yt, yp)]
            res["human_variability_mean_pairwise_MAE"] = self.iaa[d]["mean_pairwise_mae"]
            results[d] = res
        mx_all = float(sum(self.cfg["rubric_max"].values()))
        results["overall"] = {s: {"model": regression_metrics(v["human"], v["pred"], mx_all),
                                  "mean_predictor": regression_metrics(v["human"], v["mean"], mx_all)}
                              for s, v in sums.items() if not np.isscalar(v["human"])}
        return results, preds, models

    def _grouped_cv(self, feats, dev, variant) -> dict:
        """Leave-one-group-out CV on TRAIN+VAL only (test untouched), fixed alpha. Gives more
        out-of-fold predictions than the tiny validation split for a development comparison."""
        from sklearn.model_selection import LeaveOneGroupOut
        from .models.improved import make_dimension_model
        alpha = self.cfg["model"].get("cv_alpha", 1.0)
        groups = dev["group"].to_numpy()
        if len(set(groups)) < 2:
            return {}
        out, tot, tot_mean = {}, np.zeros(len(dev)), np.zeros(len(dev))
        self.oof.setdefault(variant, {"notebook_id": list(dev.notebook_id)})
        for d in DIMS:
            mx = self.cfg["rubric_max"][d]
            est = self._estimator(variant, d)
            oof, oof_mean = np.zeros(len(dev)), np.zeros(len(dev))
            for tr, te in LeaveOneGroupOut().split(dev, groups=groups):
                A, B = dev.iloc[tr], dev.iloc[te]
                m = make_dimension_model(est, d, feats[d], mx, [alpha]).fit(A, A[d].to_numpy(float))
                oof[te] = m.predict(B)
                oof_mean[te] = MeanBaseline(mx).fit(A[d].to_numpy(float)).predict(len(B))
            y = dev[d].to_numpy(float)
            out[d] = {"model": regression_metrics(y, oof, mx), "mean_predictor": regression_metrics(y, oof_mean, mx),
                      "n_folds": int(len(set(groups))), "alpha": alpha, "estimator": est}
            if est == "ridge":
                out[d]["ridge"] = out[d]["model"]
            self.oof[variant][d] = oof
            tot, tot_mean = tot + oof, tot_mean + oof_mean
        y_all = dev[DIMS].sum(axis=1).to_numpy(float)
        mx_all = float(sum(self.cfg["rubric_max"].values()))
        out["overall"] = {"model": regression_metrics(y_all, tot, mx_all),
                          "mean_predictor": regression_metrics(y_all, tot_mean, mx_all)}
        return out

    def model(self):
        df = self.features.merge(self.manifest[["notebook_id", "split", "group_key", "group", "source"]],
                                 on="notebook_id")
        tr_ids = df.loc[df.split == "train", "notebook_id"]
        tfidf = TfidfSimilarity().fit([self.expl[n] for n in tr_ids], self.refs)  # TRAIN ONLY
        self.tfidf = tfidf
        sims = pd.DataFrame([{"notebook_id": n, **tfidf.transform_one(self.expl[n], self.refs[a])}
                             for n, a in zip(df.notebook_id, df.assignment_id)])
        df = df.merge(sims, on="notebook_id")
        if getattr(self, "sem", None) is not None:
            df = df.merge(self.sem_features, on="notebook_id")
        if getattr(self, "nli_model", None) is not None:
            df = df.merge(self.nli_features, on="notebook_id")
        df = df.merge(self.y[["notebook_id"] + DIMS], on="notebook_id", how="left")
        df.to_csv(self.out / "features.csv", index=False)
        self.df = df
        labelled = df.dropna(subset=DIMS)
        parts = {s: labelled[labelled.split == s] for s in ("train", "val", "test")}
        dev = labelled[labelled.split.isin(["train", "val"])].reset_index(drop=True)
        variants = self._variants()
        for v, feats in variants.items():
            missing = [f for d in DIMS for f in feats[d] if f not in df.columns]
            if missing:
                raise ValueError(f"variant {v} needs features {missing}; enable the semantic/nli blocks in the config")
        primary = (self.cfg.get("model") or {}).get("primary_variant") or list(variants)[0]
        # pre-registered estimator selection for Week 4 variants (train+val CV only; test never used)
        self.estimators, self.estimator_selection, self.oof = {}, None, {}
        sel = (self.cfg.get("model") or {}).get("week4_estimator_selection") or {}
        if sel.get("enabled") and sel.get("reference_variant") in variants:
            from .models.improved import select_estimators
            self.estimator_selection = select_estimators(
                dev, variants[sel["reference_variant"]], DIMS, self.cfg["rubric_max"],
                self.cfg["model"].get("cv_alpha", 1.0), sel.get("min_relative_gain", 0.10))
            chosen = {d: r["chosen"] for d, r in self.estimator_selection.items()}
            for v in sel.get("apply_to", []):
                self.estimators[v] = chosen
            self.say(f"[select] Week 4 estimators (pre-registered rule, train+val CV): {chosen}")
        all_res, all_preds = {}, []
        self.variant_models = {}
        for v, feats in variants.items():
            res, preds, models = self._fit_eval(feats, parts, True, v)
            self.variant_models[v] = models
            overall = res.pop("overall")
            cv = self._grouped_cv(feats, dev, v) if self.cfg["model"].get("grouped_cv") else None
            all_res[v] = {"holdout": res, "holdout_overall": overall,
                          "grouped_cv_train_val": cv, "grouped_cv_overall": cv.pop("overall") if cv else None}
            all_preds += preds
            if v == primary:
                self.models, self.results = models, res
        self.variant_results, self.primary = all_res, primary
        pd.DataFrame(all_preds).to_csv(self.out / "predictions.csv", index=False)
        self.dev = dev
        cmp = (self.cfg.get("model") or {}).get("compare")
        if cmp and len(cmp) >= 2 and all(c in variants for c in cmp) and self.oof:
            from .evaluation.error_analysis import compare_predictions
            ea = compare_predictions(self, cmp[0], cmp[1])
            ea.to_csv(self.out / "error_analysis.csv", index=False)
        from .models.features import VARIANT_LABELS
        warning = None
        if self.status.startswith("SYNTHETIC"):
            warning = ("Synthetic development run: results are not evidence of real student performance. "
                       "Computed on SYNTHETIC development fixtures with SIMULATED labels: demonstrates "
                       "that the pipeline runs end to end, NOT how the evaluator performs on real submissions.")
        stubs = [n for n, i in (("semantic", getattr(self, "sem_info", None)), ("nli", getattr(self, "nli_info", None)))
                 if i and i["is_stub"]]
        if stubs:
            warning = (warning or "") + (f" STUB {'/'.join(stubs).upper()}: features from lexical test stand-ins, "
                                         "so Week 3/4 variants here are a wiring check, NOT model results.")
        _dump({"n_labelled": len(labelled), "split_sizes": {s: len(p) for s, p in parts.items()},
               "primary_variant": primary, "variant_labels": {v: VARIANT_LABELS.get(v, v) for v in variants},
               "results": self.results, "variants": all_res, "estimator_selection": self.estimator_selection,
               "semantic_model": getattr(self, "sem_info", None), "nli_model": getattr(self, "nli_info", None),
               "WARNING": warning},
              self.out / "metrics.json", self.status)
        for v in variants:
            for d in DIMS:
                for s in ("val", "test"):
                    r = all_res[v]["holdout"][d].get(s)
                    if r:
                        self.say(f"[eval:{v}:{d}:{s}] n={r['model']['n']} model MAE={r['model']['MAE']:.2f} "
                                 f"| mean-predictor MAE={r['mean_predictor']['MAE']:.2f}")

    # ---------------------------------------------------------------- score one new notebook (Week 5)
    def score_notebook(self, nb_path, assignment_id: str) -> dict:
        """Run the FROZEN, already-fitted pipeline on a notebook that was not part of the dataset
        (used for robustness/stress tests). Raises on unreadable files."""
        from .models.features import explanation_texts as _expl, static_features as _static
        nb = nbformat.read(str(nb_path), as_version=4)
        c = dict(self.cfg["anonymisation"])
        nb = Scrubber(c.get("roll_pattern", r"\b[A-Z]\d{3}\b")).scrub_notebook(nb)
        nid = "stress"
        from .preprocessing.parse_notebook import parse_cells
        ref = self.refs[assignment_id]
        cells = flag_template_cells(assign_tasks(parse_cells(nb, nid)), ref.template_cells,
                                    self.cfg["preprocessing"]["template_similarity_threshold"])
        f, diag = _static(cells, ref)
        expl = _expl(cells, ref)
        f.update(self.tfidf.transform_one(expl, ref))
        if getattr(self, "sem", None) is not None:
            f.update(self.sem.notebook_features(expl, ref)[0])
        if getattr(self, "nli_model", None) is not None:
            f.update(self.nli_model.notebook_features(expl, ref)[0])
        row = pd.DataFrame([f])
        preds = {v: {d: float(m.predict(row)[0]) for d, m in ms.items()} for v, ms in self.variant_models.items()}
        for v in preds:
            preds[v]["overall"] = sum(preds[v][d] for d in DIMS)
        return {"features": f, "diagnostics": diag, "predictions": preds}

    # ---------------------------------------------------------------- report per notebook
    def reports(self):
        rdir = self.out / "reports"
        rdir.mkdir(exist_ok=True)
        mx = self.cfg["rubric_max"]
        variants = getattr(self, "variant_models", {}) or {}
        encoder_loaded = getattr(self, "sem", None) is not None
        left_features = variant_features("week2_tfidf")
        right_features = variant_features("week3_combined")
        for _, r in self.df.iterrows():
            row = r.to_frame().T
            pred = {d: float(self.models[d].predict(row)[0]) for d in DIMS}
            scored = marks(pred, mx)
            diag = self.diag[r.notebook_id]
            semantic_missing = None
            if encoder_loaded and r.notebook_id in getattr(self, "sem_diag", {}):
                semantic_missing = self.sem_diag[r.notebook_id].get("missing_concepts_after_semantic")
            missing = choose_missing(diag["missing_concepts"], semantic_missing, encoder_loaded)
            rate_num = _finite(r.output_match_rate)
            text, flag = feedback_and_flag(missing, rate_num)
            raw = {
                name: {d: float(models[d].predict(row)[0]) for d in DIMS}
                for name, models in variants.items()
            }
            conf = confidence_from_variants(
                raw, mx, left_features, right_features, r.to_dict(), diag, encoder_loaded,
            )
            rep = {
                "notebook_id": r.notebook_id, "assignment_id": r.assignment_id, "split": r.split,
                "scoring_variant": self.primary,
                "code_correctness": scored["code"],
                "output_match": None if rate_num is None else bool(rate_num == 1.0),
                "output_match_rate": r.output_match_rate if rate_num is None else rate_num,
                "explanation_score": scored["explanation"],
                "documentation_quality": scored["documentation"],
                "overall_score": scored["overall"], "max_score": scored["max_score"],
                "errors": diag["errors"] + [f"output mismatch in {t}" for t in diag["output_mismatch"]]
                          + [f"cell {i} was never executed" for i in diag["unexecuted_cells"]],
                "missing_concepts": missing,
                "feedback": text,
                "confidence": conf,
                "flag_output_ok_explanation_weak": flag,
            }
            if getattr(self, "sem", None) is not None:
                sd = self.sem_diag[r.notebook_id]
                rep["semantic"] = {"model": self.sem_info["model"], "is_stub": self.sem_info["is_stub"],
                                   "per_task_similarity": {k: round(v, 3) for k, v in sd["per_task_similarity"].items()},
                                   "concepts_matched_semantically": sd["semantic_concepts_added"],
                                   "missing_concepts_after_semantic": sd["missing_concepts_after_semantic"]}
            if getattr(self, "nli_model", None) is not None:
                nd = self.nli_diag[r.notebook_id]
                rep["nli"] = {"model": self.nli_info["model"], "is_stub": self.nli_info["is_stub"],
                              "per_task": nd["per_task"], "possible_conceptual_errors": nd["flags"]}
            _dump(rep, rdir / f"{r.notebook_id}.json", self.status)
        self.say(f"[reports] wrote {len(self.df)} per-notebook reports")

    def stats(self):
        c, m = self.cells, self.manifest
        st = {
            "n_notebooks": len(m), "n_cells": len(c),
            "cells_by_type": c.cell_type.value_counts().to_dict(),
            "template_cells": int(c.is_template.sum()),
            "student_code_cells": int(((c.cell_type == "code") & ~c.is_template & c.task_id.notna()).sum()),
            "student_markdown_cells": int(((c.cell_type == "markdown") & ~c.is_template & c.task_id.notna()).sum()),
            "notebooks_by_assignment": m.assignment_id.value_counts().to_dict(),
            "notebooks_by_source": m.source.value_counts().to_dict(),
            "notebooks_by_split": m.split.value_counts().to_dict(),
            "cells_with_error_output": int(c.has_error.sum()),
            "unexecuted_code_cells": int(((c.cell_type == "code") & c.execution_count.isna()).sum()),
            "label_distribution": {d: self.y[d].value_counts().sort_index().to_dict() for d in DIMS},
            "pii_residual_matches": self.pii_residual,
        }
        _dump(st, self.out / "dataset_stats.json", self.status)
        self.stats_ = st

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self.say(f"=== data_status: {self.status} ===")
        self.anonymise(); self.parse(); self.featurise(); self.labels()
        self.split(); self.semantic(); self.nli(); self.behaviour_checks(); self.model(); self.reports(); self.stats()
        (self.out / "run_log.txt").write_text("\n".join(self.log) + "\n")
        return self


if __name__ == "__main__":
    Pipeline(sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic_dev.yaml").run()
