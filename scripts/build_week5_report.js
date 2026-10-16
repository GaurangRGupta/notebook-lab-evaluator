// Week 5 evaluation report generator. Every number is read from results/week5/report_bundle.json
// (produced by scripts/run_week5.py + scripts/export_week5_bundle.py). Usage:
//   node scripts/build_week5_report.js [output.docx]
const path = require("path");
const { H1, H2, H3, P, B, N, T, Code, Note, PB, build } = require("./report_lib");
const ROOT = path.resolve(__dirname, "..");
const b = require(path.join(ROOT, "results/week5/report_bundle.json"));
const OUT = process.argv[2] || path.join(ROOT, "results/week5/Group3_Week5_Evaluation_Report.docx");

const c = [];
const add = (...xs) => xs.forEach((x) => (Array.isArray(x) ? c.push(...x) : c.push(x)));
const f2 = (v) => (v === null || v === undefined ? "undefined" : Number(v).toFixed(2));
const f3 = (v) => (v === null || v === undefined ? "undefined" : Number(v).toFixed(3));
const pc = (v) => (v === null || v === undefined ? "n/a" : `${(100 * v).toFixed(0)}%`);
const sgn = (v) => (v > 0 ? "+" : "") + Number(v).toFixed(3);
const LBL = { mean_predictor: "Mean predictor", week2_tfidf: "Week 2 Baseline", week3_semantic: "Week 3 Semantic-only (ablation)",
  week3_combined: "Week 3 Semantic Model (improved)" };
const MODELS = ["mean_predictor", "week2_tfidf", "week3_semantic", "week3_combined"];
const DIMS = ["code", "explanation", "documentation", "overall"];
const M = (ds, sp, d, m) => b.metrics.find((r) => r.dataset === ds && r.split === sp && r.dimension === d && r.model === m);
const PR = (view, d) => b.paired.find((r) => r.view === view && r.dimension === d);
const A = b.audit.v2, A1 = b.audit.v1, CFG = b.config;
const stress = b.stress.filter((r) => r.model !== "pipeline");
const nPass = (m) => stress.filter((r) => r.model === m && r.passed === true).length;
const nCrit = (m) => stress.filter((r) => r.model === m && r.passed !== null && r.passed !== undefined).length;
const kind = (v, k) => b.root_cause_kind.find((r) => r.variant === v && r.planted_kind === k);
const tex = (d, m) => M("synthetic_v2", "test", d, m);
const pe = PR("test", "explanation"), po = PR("test", "overall"), pcv = PR("cv_train_val", "explanation");

// ============================================================================ 1-2
add(Note("warn", ["**All quantitative evidence in this report is synthetic.** No real student notebooks or faculty marks were available. " +
  "Results show how the evaluator behaves on controlled, team-authored notebooks whose labels are rule-derived or simulated. " +
  "They are not evidence of agreement with faculty grading of real submissions."]));
add(H1("1. Project identification"));
add(P("Group 3: Jupyter Notebook Lab Evaluator · Team J013, J015, J016, J017, J018 · Week 5 (Evaluation & Error Analysis), target 9 October 2026."));
add(H1("2. Executive summary"));
add(B([
  `**Audit outcome.** Before Week 5 the "improved" models of Weeks 3 and 4 had never been executed with their real pretrained models (Hugging Face is blocked in our development environment), and the only test set had 2 notebooks. Week 5 fixed the first problem for the Week 3 model by running a verified ONNX export of all-MiniLM-L6-v2 (it reproduces the published reference similarities exactly), and addressed the second with a pre-registered synthetic benchmark (v2: ${A.notebooks} notebooks, ${A.groups} synthetic students). The Week 4 NLI model still cannot be obtained, so it is **not evaluated**.`,
  `**Main comparison (v2 held-out test, n = ${pe.n} notebooks).** Explanation MAE: Week 2 Baseline ${f3(pe.MAE_baseline)}, Week 3 Semantic Model ${f3(pe.MAE_improved)} (difference ${sgn(pe.delta_MAE_improved_minus_baseline)}, ${f2(pe.relative_change_pct)}%; paired bootstrap 95% CI ${f3(pe.delta_ci95_low)} to ${f3(pe.delta_ci95_high)}; sign test p = ${f3(pe.sign_test_p)}). Overall-score MAE (0-20): ${f3(po.MAE_baseline)} vs ${f3(po.MAE_improved)} (CI ${f3(po.delta_ci95_low)} to ${f3(po.delta_ci95_high)}). The confidence intervals include zero, so the data do **not** establish that the improved model is better.`,
  `**Code and documentation scores are identical** for both models by design (the semantic component only changes the explanation model). Code MAE on test: ${f3(tex("code", "week2_tfidf").MAE)} (mean predictor ${f3(tex("code", "mean_predictor").MAE)}).`,
  `**Main failure mode.** Out-of-fold, explanations that are **wrong but reuse the reference vocabulary** are over-scored by ${f2(kind("week2_tfidf", "wrong").mean_signed_error)} marks (Week 2) and ${f2(kind("week3_combined", "wrong").mean_signed_error)} (Week 3); correct **paraphrases** are under-scored by ${f2(-kind("week2_tfidf", "paraphrase").mean_signed_error)} and ${f2(-kind("week3_combined", "paraphrase").mean_signed_error)}. In the robustness suite the semantic cosine of wrong-but-similar explanations (${f3(b.text_robustness_summary.find((r) => r.variant === "incorrect_similar").semantic_cos)}) is **higher** than that of correct paraphrases (${f3(b.text_robustness_summary.find((r) => r.variant === "correct_paraphrase").semantic_cos)}): semantic similarity measures topic, not correctness.`,
  `**Evidence produced:** ${b.error_cases.length} error-analysis cases selected by a fixed rule, ${stress.length / 2} notebook stress tests plus an invalid-file test, text robustness and probe sets, ${b.tests.passed} passing automated tests (${b.tests.skipped} skipped, ${b.tests.failed} failed), and machine-readable results in results/week5/.`,
]));

// ============================================================================ 3-4
add(H1("3. Official Week 5 objectives"));
add(T(["Source (assignment PDF)", "Requirement"], [
  ["Section 14, Week 5 (target 3-9 October)", "Focus on quantitative evaluation, test set, error analysis, robustness, failure cases. Deliverable: evaluation report with metrics and at least 5-10 error cases."],
  ["Section 7, Mandatory experimental evaluation", "Every group must report quantitative results; for grading tasks: MAE, RMSE, correlation, agreement with human graders. \"Simply saying the model performed well is not acceptable.\""],
  ["Section 8, Dataset requirements", "Document source, size, format, labels, split, preprocessing, data-quality issues, privacy, licensing."],
  ["Section 12, Mandatory documentation", "Error analysis: show at least 5-10 representative errors and explain why the system failed; responsible AI (privacy, bias, hallucination, explainability, security, human oversight); limitations."],
  ["Section 13, Part G", "Demonstrate at least one case where the system fails and explain why."],
  ["Section 6, Requirement 2", "Baseline + improved approach; students must compare the two."],
  ["Section 17, Assessment rubric", "Quantitative evaluation & error analysis 15 marks; dataset & experimental methodology 10; responsible AI / privacy 2."],
  ["Group 3 brief (Section 5)", "Evaluate code and natural-language content; stretch goal: detect \"correct output but the explanation does not correctly describe the algorithm\"; reusable across NLP/ML lab assignments."],
], [30, 70]));
add(H1("4. Project overview"));
add(P("The evaluator takes a lab notebook and produces scores for code correctness (0-8), explanation quality (0-8) and documentation quality (0-4), an overall score (0-20), detected errors and missing concepts. Implemented pipeline:"));
add(Code([".ipynb -> anonymise -> nbformat parse -> task alignment -> template removal",
  "  CODE TRACK: AST syntax | runtime vs env errors | executed? | output vs expected",
  "              | required constructs",
  "  TEXT TRACK: markdown cleaning | TF-IDF cosine (train-fitted) | MiniLM embeddings",
  "              -> semantic cosine | lexical + semantic concept coverage",
  "-> notebook features -> Ridge per dimension -> structured JSON report",
  "NOT implemented: code-explanation consistency, feedback text, confidence,",
  "                 code summarisation"]));

// ============================================================================ 5 audit
add(H1("5. Audit of the previous implementation (Weeks 2-4)"));
add(P("Each component was checked in the code and, where possible, by running it. \"Tested\" means an automated test exercises it in the current suite."));
const AUD = [
  ["Notebook parsing", "nbformat parse", "same", "same", "Implemented, tested", "parse_notebook.py: parse_cells; test_parse_cells_errors_images_ansi"],
  ["Anonymisation", "regex + class list + salted hash", "same", "same", "Implemented, tested", `anonymize.py: Scrubber; residual PII matches on v2: ${A.pii_residual_matches}`],
  ["Cell segmentation", "## Task N alignment", "same", "same", "Implemented, tested", "align_tasks.py: assign_tasks; stress test K1"],
  ["Code analysis (AST)", "calls, defs, docstrings", "same", "same", "Implemented, tested", "code_checks.py: ast_summary, required_constructs"],
  ["Syntax / error detection", "syntax, code vs env errors", "same", "same", "Implemented, tested", "code_checks.py: syntax_check, classify_error; stress tests H, M"],
  ["Output comparison", "numeric/contains/exact", "same", "same", "Implemented, tested; known blind spot", `compare_output; ${A.wrong_logic_but_output_matches} planted wrong-method tasks still match the output`],
  ["Text cleaning", "markdown to text, tech-term tokens", "same", "same", "Implemented, tested", "text_clean.py: markdown_to_text, tokenize"],
  ["TF-IDF similarity", "fit on train", "kept", "kept", "Implemented, tested", "features.py: TfidfSimilarity; test_tfidf_fit_on_train_only"],
  ["Sentence embeddings", "none", "MiniLM bi-encoder (code)", "same", "Implemented in W3 but NEVER executed before Week 5; executed in Week 5 via verified ONNX", "semantic_similarity.py: OnnxSentenceEncoder, verify_reference; test_week5 ONNX tests"],
  ["Semantic similarity", "none", "per-task cosine, semantic concept coverage", "same", "As above", "SemanticSimilarity.notebook_features"],
  ["Handcrafted features", "15 features", "+3 semantic", "+3 NLI", "Implemented, tested; weakness found", `md_tokens_log equals expl_tokens_log on all v2 notebooks: ${b.audit.md_tokens_equals_expl_tokens_v2}`],
  ["Feature aggregation", "merge per notebook", "variants", "variants + ablations", "Implemented", "pipeline.py: model()"],
  ["Scoring model", "Ridge per dimension", "same", "Ridge or forest (pre-registered rule)", "Ridge implemented, tested; W4 forest rule tested only with stand-in features", "baseline.py, improved.py"],
  ["Train/val/test split", "grouped, notebook level", "same", "same", "Implemented, tested; v1 test n = 2 is too small", "split.py: grouped_split; test_v2_split_is_notebook_level_grouped_and_fixed"],
  ["Leakage prevention", "group split, train-only fit", "frozen encoder", "frozen NLI", "Implemented, tested", "check_no_group_leakage; tests"],
  ["Baseline model", "Week 2 Ridge", "kept", "frozen snapshot", "Implemented, reproduced exactly", `frozen predictions reproduced: ${b.week2_frozen_reproduced}`],
  ["Improved model", "n/a", "week3_combined", "week4_improved (NLI)", "W3 evaluated in Week 5; W4 implemented but NOT executable (model unavailable)", "configs/week5_v2.yaml; docs/week5_protocol.md"],
  ["Evaluation metrics", "MAE, RMSE, r, rho, ±1, exact", "+ grouped CV", "+ overall", "Implemented, tested; rounding bug fixed in Week 5", "metrics.py: regression_metrics, round_half_up"],
  ["Error analysis", "none", "probe pairs (TF-IDF only measured)", "error_analysis.py (stubs only)", "Executed for the first time in Week 5", "results/week5/error_cases.csv"],
  ["Robustness tests", "none", "none", "text suite (TF-IDF only measured)", "Executed in Week 5 with the real encoder + notebook stress tests", "stress_tests.py; robustness_results.csv"],
  ["Reproducibility", "config + scripts", "run_week3.py", "run_week4.py", "Week 5 rerun produced identical metrics and predictions", "scripts/run_week5.py"],
  ["Tests", "29", "45", "59 (4 skipped)", `${b.tests.passed} passed, ${b.tests.skipped} skipped`, "results/week5/test_summary.txt"],
  ["Documentation", "README, datasheet, guidelines", "model card", "model card", "Implemented", "docs/"],
  ["Code-explanation consistency (Week 1 task f)", "planned", "planned", "planned", "NOT implemented", "no module; stress tests E and F"],
  ["Feedback text, confidence, code summarisation", "planned", "planned", "planned", "NOT implemented", "report fields are null"],
];
add(T(["Requirement", "Week 2", "Week 3", "Week 4", "Actual status", "Evidence"], AUD, [17, 14, 14, 14, 20, 21]));
add(H2("Five most important weaknesses found"));
add(N([
  "**The improved models had never produced a single real number.** Week 3 and Week 4 reports contained only pending values because the pretrained models could not be downloaded. Fixed for Week 3 (verified ONNX export); Week 4 remains unevaluated.",
  `**No usable test set.** v1 has ${A1.notebooks} notebooks from ${A1.groups} synthetic students; the test split has ${A1.split_counts.test} notebooks, so correlations are undefined and any comparison is noise. Addressed with the pre-registered v2 benchmark (still synthetic).`,
  "**Agreement metrics used half-to-even rounding** (numpy rint: 2.5 rounds to 2, 3.5 to 4). Fixed (round half up); no previously reported value changed.",
  "**One unreadable notebook crashed the entire run.** Fixed: the file is skipped and logged; tested.",
  "**Design gaps:** no code-explanation consistency check exists, and the documentation feature `md_tokens_log` is identical to the explanation length feature, so documentation scores partly reflect explanation length. Not fixed in Week 5 (changing it would alter the frozen baseline); quantified below.",
]));

// ============================================================================ 6-7 dataset
add(H1("6. Dataset description"));
add(Note("synthetic", "Both datasets are team-authored synthetic notebooks executed with a real Python kernel. No student data were used."));
add(T(["Property", "v1 (Week 2 development set)", "v2 (Week 5 benchmark)"], [
  ["Source", "10 hand-written notebooks, 5 quality profiles x 2 labs", "Generated by scripts/make_synthetic_v2.py (seed 2026) from fixed pools of code variants and explanations"],
  ["Notebooks / groups (students)", `${A1.notebooks} / ${A1.groups}`, `${A.notebooks} / ${A.groups}`],
  ["Cells", `${A1.cells}`, `${A.cells} (${A.cells_by_type.code} code, ${A.cells_by_type.markdown} markdown)`],
  ["Assignments", "LAB_SYN1 (TF-IDF, cosine), LAB_SYN2 (bigram LM)", "same two labs, same references"],
  ["Labels", A1.label_source, A.label_source],
  ["Split (train/val/test notebooks)", `${A1.split_counts.train}/${A1.split_counts.val}/${A1.split_counts.test}`, `${A.split_counts.train}/${A.split_counts.val}/${A.split_counts.test} (students ${A.split_groups.train}/${A.split_groups.val}/${A.split_groups.test})`],
  ["Missing labels / duplicate files", "0 / 0", `${A.label_missing} / ${A.duplicate_raw_files}`],
  ["Licence", "team-authored", "team-authored"],
], [26, 37, 37]));
add(H2("Label distributions (v2)"));
const ld = (d) => Object.entries(A.label_distribution[d]).map(([k, v]) => `${Number(k)}: ${v}`).join(", ");
add(T(["Dimension (range)", "Mark: count"], [["Code (0-8)", ld("code")], ["Explanation (0-8)", ld("explanation")], ["Documentation (0-4)", ld("documentation")]], [25, 75]));
add(H2("v2 label rules (fixed before any model was run; docs/week5_protocol.md)"));
add(B(["Code = round_half_up(8 x points / 3); per task: correct (either implementation) 1, unexecuted correct code 0.5, syntax error / runtime error / wrong logic 0.",
  "Explanation = sum over two prompts of: correct 4, paraphrase 4, partial 2, vague 1, wrong / off-topic / empty 0.",
  "Documentation = comment level (0-2) + 1 if the helper function has a docstring + 1 if any explanation was written.",
  "Overall = code + explanation + documentation (0-20). The ranges match the Week 1 proposal rubric (8/8/4)."]));
add(H1("7. Dataset quality and label limitations"));
add(T(["Check", "Result", "Implication"], [
  ["Planted behaviour vs actual execution", `${b.design_check.inconsistent} inconsistencies in ${b.design_check.rows} task checks (after amendment A1)`, "Each planted defect really occurs when the notebook runs"],
  ["Generator bug found before any model run", "helper defined in a broken cell caused cascading NameErrors (1 notebook, 2 tasks)", "Fixed (amendment A1) and regenerated with the same seed"],
  ["Wrong logic but matching output", `${A.wrong_logic_but_output_matches} tasks (CountVectorizer prints the expected shape)`, "Output comparison alone cannot detect them"],
  ["Explanation texts shared between train and test", `${A.test_explanations_seen_verbatim_in_train} test explanations appear verbatim in training notebooks`, "Pool reuse makes v2 easier than real data; real students write unique text"],
  ["Near-duplicate notebooks merged into one group", `${A.near_duplicate_groups_merged}`, "No pair reached the 0.95 similarity threshold"],
  ["Label source", "rules (v2), simulated annotators (v1)", "No measure of real grader agreement; v1 simulated agreement (alpha) only tests the code"],
], [30, 35, 35]));

// ============================================================================ 8-10
add(H1("8. Experimental design"));
add(B([
  "Protocol written before running any model on v2 (docs/week5_protocol.md), including the models, metrics, split rule, error-case selection rule and robustness criteria.",
  "Two datasets: v2 is the main evaluation; v1 is reported for continuity with Weeks 2-4.",
  "Three views per dataset: validation (used for alpha selection), held-out test (scored once), and leave-one-student-out cross-validation on train+validation (stability check; test untouched).",
  "Controlled robustness tests are reported separately and never mixed into test metrics.",
  `Re-running scripts/run_week5.py reproduced metrics.csv and predictions.csv exactly.`,
]));
add(H1("9. Train/validation/test methodology"));
add(P(`Split unit: the notebook. Groups: synthetic student (both labs of a student stay together) plus near-duplicate clusters. Method: ${CFG.split_v2.method}. Realised proportions train ${pc(CFG.split_v2.realised.train)}, validation ${pc(CFG.split_v2.realised.val)}, test ${pc(CFG.split_v2.realised.test)} of ${CFG.split_v2.n_notebooks} notebooks; an automated test checks that no student crosses splits and that every cell follows its notebook. With ${A.split_counts.test} test notebooks from ${A.split_groups.test} students, test estimates are still imprecise; bootstrap intervals are reported.`));
add(H1("10. Data-leakage safeguards"));
add(T(["Risk", "Safeguard", "Verified by"], [
  ["Same student in train and test", "Group split by student; near-duplicate merging", "test_v2_split_is_notebook_level_grouped_and_fixed"],
  ["Cells of one notebook in different splits", "Cells inherit their notebook's split", "same test"],
  ["TF-IDF vocabulary learns from test", "Fitted on training explanations + references only", "test_tfidf_fit_on_train_only"],
  ["Imputation / scaling learn from test", "Fitted inside the training pipeline", "baseline.py make_model"],
  ["Encoder learns from our data", "Pretrained, frozen, no fine-tuning", "verify_reference; determinism test"],
  ["Model selection on test", `Only Ridge alpha chosen, on validation (selected: ${Object.entries(CFG.alpha_selected_v2.week3_combined).map(([d, a]) => `${d} ${a}`).join(", ")} for the improved model); concept threshold ${CFG.concept_threshold_v2} on authored tune pairs`, "experiment_config.json"],
  ["Week 4 model dropped after seeing results", "Exclusion declared in the protocol before any run", "docs/week5_protocol.md section 5"],
], [26, 46, 28]));

// ============================================================================ 11-12
add(H1("11. Models evaluated"));
add(T(["Model", "Explanation features", "Code / documentation features", "Estimator", "Status"], [
  ["Mean predictor", "none", "none", "training mean per dimension", "Evaluated"],
  ["Week 2 Baseline (week2_tfidf)", "tfidf_cos_mean, tfidf_cos_max, expl_tokens_log, frac_tasks_explained, keyword_coverage", "6 code / 4 documentation features", "Ridge, alpha in {0.1, 1, 10} by validation MAE, standardised, median-imputed", "Evaluated (frozen)"],
  ["Week 3 Semantic-only (ablation)", "sem_cos_mean, sem_cos_max, expl_tokens_log, frac_tasks_explained, sem_concept_coverage", "same as baseline", "same", "Evaluated"],
  ["Week 3 Semantic Model (improved)", "5 baseline features + sem_cos_mean, sem_cos_max, sem_concept_coverage", "same as baseline", "same", "Evaluated"],
  ["Week 4 Improved Model (NLI)", "Week 3 + nli_support, nli_coverage, nli_contradiction", "same", "Ridge or forest (pre-registered rule)", "NOT evaluated: model unavailable"],
], [20, 30, 16, 20, 14]));
add(P(`Encoder: ${CFG.encoder.name}; provenance check against the sentence-transformers README similarity matrix: expected ${JSON.stringify(CFG.encoder.provenance_check.expected)}, obtained ${JSON.stringify(CFG.encoder.provenance_check.got)}, passed = ${CFG.encoder.provenance_check.passed}. ONNX file SHA-256 prefix ${CFG.encoder.onnx_sha256_16}. Seeds: split 42, generator 2026, bootstrap 0.`));
add(H1("12. Evaluation metrics and definitions"));
add(T(["Metric", "Definition", "What it tells us", "Limitation"], [
  ["MAE", "mean |prediction - label|", "Average marks off", "Hides whether errors are systematic"],
  ["RMSE", "sqrt(mean (prediction - label)^2)", "Penalises large errors", "Sensitive to a few outliers on small n"],
  ["Pearson r", "linear correlation", "Do predictions move with labels?", "Undefined if n < 3 or a series is constant; unstable on small n"],
  ["Spearman rho", "rank correlation", "Same ordering of notebooks?", "Same as above"],
  ["Within ±1", "share with |round_half_up(prediction) - label| <= 1", "Usable-agreement rate", "Lenient on the 0-4 documentation scale"],
  ["Exact", "share with round_half_up(prediction) = label", "Strict agreement", "Harsh for continuous predictions"],
  ["MAE 95% CI", "percentile bootstrap, 2,000 resamples (n >= 10)", "Uncertainty of MAE", "Assumes notebooks are exchangeable; wide at n = 12"],
  ["Paired delta MAE", "improved MAE - baseline MAE on the same notebooks; paired bootstrap (5,000) and sign test", "Is the difference distinguishable from noise?", "Low power at n = 12"],
], [14, 32, 26, 28]));

// ============================================================================ 13-15 results
add(H1("13. Quantitative results"));
add(Note("synthetic", "Synthetic development results (v2, rule-derived labels). Held-out test scored once."));
const resRows = (ds, sp) => {
  const rows = [];
  DIMS.forEach((d) => MODELS.forEach((m) => {
    const r = M(ds, sp, d, m);
    if (!r) return;
    const ci = r.mae_ci95_low != null ? ` [${f2(r.mae_ci95_low)}, ${f2(r.mae_ci95_high)}]` : "";
    rows.push([LBL[m], d, r.n, f3(r.MAE) + ci, f3(r.RMSE), f3(r.pearson_r), f3(r.spearman_rho), pc(r.within_1), pc(r.exact)]);
  }));
  return rows;
};
const HDR = ["Model", "Dimension", "N", "MAE [95% CI]", "RMSE", "Pearson", "Spearman", "±1", "Exact"];
const W = [22, 11, 5, 18, 9, 9, 9, 8, 9];
add(H2("13.1 v2 held-out test set"));
add(T(HDR, resRows("synthetic_v2", "test"), W));
add(H2("13.2 v2 validation set (used for alpha selection)"));
add(T(HDR, resRows("synthetic_v2", "val"), W));
add(H2("13.3 v2 leave-one-student-out CV on train+validation (n = 48)"));
add(T(HDR, resRows("synthetic_v2", "cv_train_val"), W));
add(P("Mean-predictor correlations in cross-validation are negative by construction: leaving out a high-scoring student lowers the training mean used to predict that student. They are not a property of the data."));
add(H2("13.4 v1 (original 10-notebook set; test n = 2, CV n = 8)"));
const v1rows = [];
["explanation", "overall"].forEach((d) => ["test", "cv_train_val"].forEach((sp) => MODELS.forEach((m) => {
  const r = M("synthetic_v1", sp, d, m); if (r) v1rows.push([LBL[m], d, sp, r.n, f3(r.MAE), f3(r.RMSE), f3(r.spearman_rho), pc(r.within_1)]);
})));
add(T(["Model", "Dimension", "View", "N", "MAE", "RMSE", "Spearman", "±1"], v1rows, [26, 13, 15, 6, 10, 10, 11, 9]));
const v1e2 = M("synthetic_v1", "cv_train_val", "explanation", "week2_tfidf"), v1e3 = M("synthetic_v1", "cv_train_val", "explanation", "week3_combined");
add(P(`On v1 the direction differs from v2: in cross-validation the Week 3 model has a higher explanation MAE (${f3(v1e3.MAE)}) than the baseline (${f3(v1e2.MAE)}). With 8 notebooks this illustrates how unstable small-sample comparisons are.`));

add(H1("14. Baseline-versus-improved comparison"));
add(P("Same notebooks, labels, split, preprocessing and metrics. Negative delta = improved model has lower error."));
add(T(["View", "Dimension", "N", "MAE Week 2", "MAE Week 3", "Delta", "Relative", "95% CI of delta", "Week 3 closer / Week 2 closer / tie", "Sign test p"],
  b.paired.filter((r) => ["explanation", "overall"].includes(r.dimension)).map((r) => [r.view, r.dimension, r.n, f3(r.MAE_baseline), f3(r.MAE_improved),
    sgn(r.delta_MAE_improved_minus_baseline), `${f2(r.relative_change_pct)}%`, `${f3(r.delta_ci95_low)} to ${f3(r.delta_ci95_high)}`,
    `${r.improved_closer} / ${r.baseline_closer} / ${r.ties}`, f3(r.sign_test_p)]), [10, 10, 5, 9, 9, 9, 9, 15, 14, 10]));
const allCI0 = b.paired.filter((r) => ["explanation", "overall"].includes(r.dimension)).every((r) => r.delta_ci95_low <= 0 && r.delta_ci95_high >= 0);
add(Note("key", `On every view the 95% interval of the difference ${allCI0 ? "includes zero" : "does not always include zero"}. The improved model has the lower explanation MAE on test (${sgn(pe.delta_MAE_improved_minus_baseline)}), validation (${sgn(PR("val", "explanation").delta_MAE_improved_minus_baseline)}) and CV (${sgn(pcv.delta_MAE_improved_minus_baseline)}), but the differences are small relative to their uncertainty, and on v1 the direction reverses. Conclusion: no reliable difference has been demonstrated.`));
add(H2("Ablation: semantic features replacing TF-IDF"));
const ab = (sp) => [M("synthetic_v2", sp, "explanation", "week2_tfidf"), M("synthetic_v2", sp, "explanation", "week3_semantic"), M("synthetic_v2", sp, "explanation", "week3_combined")];
add(T(["View", "Week 2 (TF-IDF)", "Semantic only", "TF-IDF + semantic"], ["test", "val", "cv_train_val"].map((sp) => [sp, ...ab(sp).map((r) => f3(r.MAE))]), [25, 25, 25, 25]));
add(P("Replacing the lexical features with semantic ones (semantic-only) gives a higher explanation MAE than the baseline on every view; only the combination is lower. The semantic features add information but do not substitute for lexical overlap."));

add(H1("15. Results by scoring dimension"));
const tc = (d) => [tex(d, "mean_predictor"), tex(d, "week2_tfidf"), tex(d, "week3_combined")];
add(B([
  `**Code (0-8).** Test MAE ${f3(tc("code")[1].MAE)} for both models vs ${f3(tc("code")[0].MAE)} for the mean predictor; ±1 agreement ${pc(tc("code")[1].within_1)}. Determined by deterministic code-track features (output match, errors, executed cells); unaffected by NLP changes.`,
  `**Explanation (0-8).** Test MAE ${f3(tc("explanation")[1].MAE)} (Week 2) and ${f3(tc("explanation")[2].MAE)} (Week 3) vs ${f3(tc("explanation")[0].MAE)} (mean); ±1 agreement ${pc(tc("explanation")[1].within_1)} and ${pc(tc("explanation")[2].within_1)}. The hardest dimension and the only one the models differ on.`,
  `**Documentation (0-4).** Test MAE ${f3(tc("documentation")[1].MAE)} vs ${f3(tc("documentation")[0].MAE)} (mean). Partly driven by explanation length because md_tokens_log duplicates expl_tokens_log (stress test O).`,
  `**Overall (0-20).** Test MAE ${f3(tex("overall", "week2_tfidf").MAE)} vs ${f3(tex("overall", "week3_combined").MAE)}; mean predictor ${f3(tex("overall", "mean_predictor").MAE)}.`,
]));
add(H2("Concept matching (lexical vs semantic), authored eval pairs"));
const ce = b.concept_calibration.eval;
add(T(["Method", "Precision", "Recall", "F1", "n"], [["Lexical only (Week 2)", f3(ce.lexical_only.precision), f3(ce.lexical_only.recall), f3(ce.lexical_only.f1), ce.lexical_only.n],
  ["Semantic only (threshold " + b.concept_calibration.threshold + ")", f3(ce.semantic_only.precision), f3(ce.semantic_only.recall), f3(ce.semantic_only.f1), ce.semantic_only.n],
  ["Lexical or semantic (used)", f3(ce.lexical_or_semantic.precision), f3(ce.lexical_or_semantic.recall), f3(ce.lexical_or_semantic.f1), ce.lexical_or_semantic.n]], [40, 15, 15, 15, 15]));
add(P(`Threshold chosen on ${b.concept_calibration.n_tune} tune pairs, evaluated on ${b.concept_calibration.n_eval} pairs with different concepts. Semantic matching raises recall and lowers precision.`));

// ============================================================================ 16 error analysis
add(H1("16. Error analysis"));
add(P(`Cases were selected by the rule fixed in the protocol (all test rows with an error above 1 mark, extended with validation rows to reach at least 10, plus the two controls), not chosen by hand. ${b.error_cases.length} cases; all are synthetic v2 notebooks with rule-derived labels; no real-data cases exist.`));
add(T(["Case", "Notebook", "Split", "Dim.", "Label", "W2 pred", "W3 pred", "W2 err", "W3 err", "Category"],
  b.error_cases.map((e) => [e.case_id, e.anon_label, e.source_split, e.dimension, e.human, f2(e.baseline_week2), f2(e.improved_week3), sgn(e.baseline_error), sgn(e.improved_error), e.error_category]),
  [6, 15, 7, 10, 6, 7, 7, 7, 7, 28]));
const cnt = (pred) => b.error_cases.filter(pred).length;
add(P(`Balance: Week 3 closer in ${cnt((e) => Math.abs(e.improved_error) < Math.abs(e.baseline_error))} cases, Week 2 closer in ${cnt((e) => Math.abs(e.baseline_error) < Math.abs(e.improved_error))}, equal in ${cnt((e) => Math.abs(e.baseline_error) === Math.abs(e.improved_error))}; both outside ±1 (rounded) in ${cnt((e) => e.category_vs_pm1 === "both fail")}.`));
add(H2("Case details"));
add(T(["Case", "Planted input", "Root cause (from planted design and measured features)", "Mitigation"],
  b.error_cases.map((e) => [e.case_id, `${e.assignment_id}; code ${e.planted_code}; explanation ${e.planted_explanation}`, e.root_cause, e.suggested_mitigation]), [7, 25, 42, 26]));

// ============================================================================ 17-19 robustness
add(H1("17. Robustness test methodology"));
add(B(["**Notebook stress tests:** a high-quality LAB_SYN1 base notebook is perturbed in one controlled way per test, executed with a real kernel, and scored by the frozen models fitted on v2 train. Pass criteria were written in src/evaluation/stress_tests.py before the run.",
  "**Text robustness:** 5 reference explanations x 8 variants (paraphrase, reversed sentences, scrambled words, irrelevant sentence, 5-word truncation, empty, wrong-but-similar, verbatim copy) scored by TF-IDF and the semantic encoder.",
  "**Probe pairs:** 13 authored (reference, answer) pairs with known correctness.",
  "None of these are mixed into the test metrics; they test behaviour, not accuracy on real data."]));
add(H1("18. Robustness results"));
add(H2(`Notebook stress tests (Week 2 passed ${nPass("week2_tfidf")}/${nCrit("week2_tfidf")}, Week 3 passed ${nPass("week3_combined")}/${nCrit("week3_combined")})`));
const sx = {};
stress.forEach((r) => { (sx[r.test] = sx[r.test] || {})[r.model] = r; });
add(T(["Test", "Condition", "Criterion", "W2 code/expl/doc", "W3 code/expl/doc", "Pass W2 / W3"],
  Object.entries(sx).map(([t, v]) => [t, v.week2_tfidf.input_condition, v.week2_tfidf.criterion,
    `${f2(v.week2_tfidf.pred_code)} / ${f2(v.week2_tfidf.pred_explanation)} / ${f2(v.week2_tfidf.pred_documentation)}`,
    `${f2(v.week3_combined.pred_code)} / ${f2(v.week3_combined.pred_explanation)} / ${f2(v.week3_combined.pred_documentation)}`,
    t === "BASE" ? "reference" : `${v.week2_tfidf.passed ? "pass" : "FAIL"} / ${v.week3_combined.passed ? "pass" : "FAIL"}`]), [18, 26, 20, 13, 13, 10]));
const pinv = b.stress.find((r) => r.test === "P_invalid_notebook_file");
add(P(`P (invalid notebook file): ${pinv.passed ? "passed" : "FAILED"}; the run completed and logged "${pinv.note}". Environment errors were classified as environment errors in test M (${sx.M_missing_package_or_file.week2_tfidf.env_errors_detected} detected).`));
add(H2("Text robustness (mean over 5 cases)"));
add(T(["Variant", "TF-IDF cosine", "Semantic cosine"], b.text_robustness_summary.map((r) => [r.variant, f3(r.tfidf_cos), f3(r.semantic_cos)]), [40, 30, 30]));
add(H2("Probe pairs"));
add(T(["Probe", "Category", "Answer correct?", "TF-IDF", "Semantic"], b.probes.map((p) => [p.id, p.category, p.student_is_correct ? "yes" : "no", f3(p.tfidf_cos), f3(p.semantic_cos)]), [10, 40, 16, 17, 17]));
add(H1("19. Failure cases and code-explanation consistency"));
const F = sx.F_wrong_code_convincing_explanation, E = sx.D_E_correct_code_wrong_explanation, G = sx.G_missing_output;
add(B([
  `**Correct code, wrong explanation (test D/E):** explanation predicted ${f2(E.week2_tfidf.pred_explanation)} (W2) and ${f2(E.week3_combined.pred_explanation)} (W3) although the rule label is 0; semantic cosine ${f3(E.week3_combined.sem_cos_mean)} vs base ${f3(sx.BASE.week3_combined.sem_cos_mean)}. The wrong explanation looks similar to the reference.`,
  `**Wrong code, convincing explanation (test F):** code drops to ${f2(F.week2_tfidf.pred_code)} while explanation stays at ${f2(F.week3_combined.pred_explanation)}. The two scores are computed independently; no flag reports that the explanation describes code that does not work.`,
  `**Missing outputs (test G):** code ${f2(G.week2_tfidf.pred_code)}, explanation unchanged (${f2(G.week3_combined.pred_explanation)}).`,
  "**Consistency check status:** NOT implemented. The system measures similarity to the reference explanation, not correctness of the explanation, and not consistency between explanation and code. These are three different properties; the Group 3 stretch goal (\"correct output, but the explanation does not describe the algorithm\") is therefore not met.",
  `**Paraphrases (test C):** explanation falls from ${f2(sx.BASE.week3_combined.pred_explanation)} to ${f2(sx.C_paraphrased_explanation.week3_combined.pred_explanation)} (W3) for an equally correct answer: bias against differently worded explanations.`,
]));

// ============================================================================ 20 root cause
add(H1("20. Root-cause analysis"));
add(H2("Signed error by planted explanation kind (out-of-fold, train+validation)"));
const kinds = ["correct", "paraphrase", "partial", "vague", "wrong", "empty"];
add(T(["Planted kind", "Task slots", "Week 2", "Semantic only", "Week 3"], kinds.map((k) => [k, kind("week2_tfidf", k).n_task_slots,
  sgn(kind("week2_tfidf", k).mean_signed_error), sgn(kind("week3_semantic", k).mean_signed_error), sgn(kind("week3_combined", k).mean_signed_error)]), [24, 16, 20, 20, 20]));
add(H2("Feature-label correlation on v2 training notebooks (Spearman, explanation)"));
add(T(["Feature", "rho"], Object.entries(b.audit.feature_correlations_train_v2).map(([k, v]) => [k, f3(v)]), [60, 40]));
add(H2("Overfitting check"));
add(T(["Model", "Dimension", "Train in-sample MAE", "CV out-of-fold MAE"], b.overfitting.map((r) => [LBL[r.variant], r.dimension, f3(r.train_in_sample_MAE), f3(r.cv_out_of_fold_MAE)]), [35, 20, 22, 23]));
add(T(["Finding", "Evidence", "Status"], [
  ["Embeddings reward topic, not correctness", `wrong-but-similar mean semantic cosine ${f3(b.text_robustness_summary.find((r) => r.variant === "incorrect_similar").semantic_cos)} > correct paraphrase ${f3(b.text_robustness_summary.find((r) => r.variant === "correct_paraphrase").semantic_cos)}; wrong explanations over-scored`, "Supported"],
  ["Lexical features penalise paraphrases", `paraphrase signed error ${sgn(kind("week2_tfidf", "paraphrase").mean_signed_error)} (W2)`, "Supported"],
  ["Explanation score tracks length", `expl_tokens_log has the highest correlation with the explanation label (${f3(b.audit.feature_correlations_train_v2.expl_tokens_log)})`, "Supported on synthetic data; in the pools wrong answers are long and vague ones short, so length is a confound"],
  ["Code correctness from proxies", `${A.wrong_logic_but_output_matches} wrong-method tasks match the expected output; cases E10, E12`, "Supported"],
  ["Mild overfitting", "in-sample MAE below CV MAE for every model", "Supported; expected with 36 training notebooks"],
  ["Synthetic labels too easy", `${A.test_explanations_seen_verbatim_in_train} test explanations seen verbatim in training; code labels deterministic`, "Supported: real-data error will likely be higher (hypothesis)"],
  ["Rubric ambiguity", "unexecuted code = 0.5 points is a design choice", "Hypothesis; needs annotator discussion"],
  ["Task alignment failure", "K1 reordered blocks scored identically to base", "Not observed in tests; headers present in all fixtures"],
], [28, 50, 22]));

// ============================================================================ 21-22
add(H1("21. Fixes implemented in Week 5"));
add(T(["Issue", "Evidence", "Fix", "Tests", "Effect on comparability"], [
  ["Week 3 model never executed", "Week 3/4 reports pending; huggingface.co blocked", "OnnxSentenceEncoder + verify_reference + fetch_models.py (npm package integrity pinned)", "test_onnx_* (4)", "First real Week 3 numbers; same weights as the official model (reference matrix reproduced)"],
  ["Half-to-even rounding in ±1/exact", "rint(2.5) = 2", "round_half_up", "test_round_half_up_and_agreement_definition", `No reported value changed (${b.rounding_fix_changes.length} differences on v1)`],
  ["Invalid notebook crashed run", "NotJSONError propagated", "skip + skipped_notebooks.csv", "test_invalid_notebook_skipped_not_crash; stress test P", "None on valid data"],
  ["Test n = 2", "v1 split", "pre-registered v2 benchmark (60 notebooks)", "design and label-rule tests", "v1 results kept and reported separately"],
  ["Generator cascade bug", "design check: 2 inconsistent tasks", "helper in own cell (amendment A1, before any model run)", "test_v2_planted_design_matches_execution", "None (no results existed)"],
  ["Not fixed: md/expl length duplication", "md_tokens_log = expl_tokens_log", "documented; change would alter the frozen baseline", "stress test O", "Week 6 candidate"],
  ["Not fixed: no consistency check", "stress tests E, F", "documented", "n/a", "Week 6 candidate"],
], [16, 18, 26, 20, 20]));
add(H1("22. Automated test results"));
add(Code(["$ python -m pytest -q -rs tests", `${b.tests.passed} passed, ${b.tests.skipped} skipped, ${b.tests.failed} failed`, ...b.tests.skipped_lines.map((l) => l.slice(0, 110))]));
add(P("The 4 skipped tests load the original Hugging Face models (sentence-transformers and the NLI cross-encoder), which this environment cannot download. The same encoder weights are covered by the ONNX tests."));

// ============================================================================ 23-25
add(H1("23. Responsible AI and privacy"));
add(B([
  "**Advisory only.** Scores support faculty; the faculty member makes the final grading decision. Large errors observed here (up to about 5 marks on explanations) would be unacceptable as automatic grades.",
  "**Privacy.** Anonymisation runs first (roll numbers, e-mails, class-list names, user paths, metadata; salted file hashes). The encoder runs locally; no notebook is sent to an external service. No student data were used in Week 5.",
  "**False positives / negatives.** Wrong but keyword-rich explanations are over-scored (false confidence); correct paraphrases and concise answers are under-scored (penalising students who write in their own words or briefly).",
  "**Bias.** Dependence on vocabulary and length disadvantages concise writers and non-native phrasing (hypothesis for real data; shown for paraphrases here).",
  "**Reference dependence.** Every explanation score depends on one reference explanation and its concept list; a weak reference produces weak scores.",
  "**Environment failures.** Missing packages/files are classified as environment errors in reports but still lower the code score (test M).",
  "**Uncertainty.** The system outputs no confidence yet; bootstrap intervals show test MAE could plausibly be much higher.",
  "**Licensing.** Synthetic data are team-authored. The encoder is Apache-2.0 (sentence-transformers/all-MiniLM-L6-v2); the ONNX copy came from an npm package with a pinned integrity hash and was verified behaviourally. Public notebooks (not used yet) must have their licence checked per source.",
]));
add(H1("24. Limitations"));
add(B([
  "No real student notebooks or human marks: nothing here measures agreement with faculty.",
  "v2 labels are rule-derived from planted properties; explanation texts come from small pools shared between train and test, which makes the task easier than reality.",
  `Test set of ${A.split_counts.test} notebooks: wide intervals; the comparison is inconclusive.`,
  "Week 4 NLI model could not be evaluated.",
  "No code-explanation consistency check, feedback text or confidence score.",
  "Only two labs and one notebook template style; real notebooks vary much more.",
  "Probe, robustness and calibration sets are small and authored by the team.",
]));
add(H1("25. Conclusion"));
add(P(`Week 5 turned previously untested claims into measured evidence. The baseline is reproducible and frozen; the Week 3 semantic model now runs with verified weights; on the synthetic v2 test set it reaches an explanation MAE of ${f3(pe.MAE_improved)} against ${f3(pe.MAE_baseline)} for the baseline, a difference whose 95% interval (${f3(pe.delta_ci95_low)} to ${f3(pe.delta_ci95_high)}) includes zero. The clearest, best-supported finding is qualitative: similarity to a reference, lexical or semantic, does not measure correctness. Wrong explanations that reuse the right words are over-scored by both models, correct paraphrases are under-scored, and the system cannot detect inconsistency between code and explanation.`));

// ============================================================================ 26-28
add(H1("26. Week 6 recommendations"));
add(N([
  "Obtain the faculty-approved class notebooks and run the 3-annotator protocol; re-run scripts/run_week5.py with configs/class_data.yaml.",
  "Implement a code-explanation consistency signal and evaluate it on stress tests E and F.",
  "Make the Week 4 NLI model available (local copy) and evaluate it under the same protocol, since it targets the main failure mode.",
  "Separate documentation features from explanation length (new variant, baseline kept frozen).",
  "Treat missing required constructs as task failures; expose per-task diagnostics in the report.",
  "Add a confidence estimate and route low-confidence or high-similarity/low-correctness cases to human review before building the API.",
]));
add(H1("27. Reproducibility instructions"));
add(Code(["pip install -r requirements.txt", "python -m ipykernel install --user",
  "python scripts/fetch_models.py          # MiniLM ONNX export, integrity + provenance check",
  "python scripts/make_synthetic_v2.py     # optional: regenerate v2 (deterministic, seed 2026)",
  "python scripts/run_week5.py             # all experiments -> results/week5/",
  "python scripts/export_week5_bundle.py   # collect results for the report",
  "node scripts/build_week5_report.js      # this document",
  "python -m pytest -q -rs tests"]));
add(T(["File", "Content"], [
  ["results/week5/metrics.csv", "every metric, model, split, dimension, n, CI, evidence type"],
  ["results/week5/predictions.csv", "per-notebook labels, predictions and errors (anonymised ids)"],
  ["results/week5/paired_comparison.csv", "baseline vs improved paired statistics"],
  ["results/week5/error_cases.csv", "the error-analysis cases with root cause and mitigation"],
  ["results/week5/robustness_results.csv", "notebook stress tests"],
  ["results/week5/text_robustness_results.csv, probe_results.csv", "text-level robustness and probes"],
  ["results/week5/root_cause_by_explanation_kind.csv, overfitting_check.csv", "root-cause evidence"],
  ["results/week5/dataset_audit.json, design_check.csv", "dataset audit"],
  ["results/week5/experiment_config.json, test_summary.txt", "configuration, seeds, provenance, test output"],
  ["docs/week5_protocol.md", "pre-registered protocol and amendment A1"],
], [45, 55]));
add(H1("28. References and data sources"));
add(B(["NLP Course Project 2026 assignment handout (Sections 5-8, 12-14, 17).", "Group 3 Week 1 proposal and Week 2-4 code and reports.",
  "Reimers & Gurevych (2019), Sentence-BERT; sentence-transformers/all-MiniLM-L6-v2 model and README reference values (github.com/huggingface/sentence-transformers).",
  "ONNX export: npm @alvix/all-minilm-l6-v2@1.0.1 (Xenova export), integrity sha512 pinned in scripts/fetch_models.py.",
  "scikit-learn, nbformat, nbclient, onnxruntime, tokenizers.", "Synthetic datasets v1 and v2: team-authored (this repository)."]));

// ============================================================================ checklist
add(PB());
add(H1("Appendix: checklist against the official assignment"));
add(T(["Requirement", "Evidence", "File / section", "Status"], [
  ["Quantitative evaluation", "MAE, RMSE, Pearson, Spearman, ±1, exact, CIs for 4 models x 4 dimensions", "metrics.csv; s.13", "Met (synthetic data only)"],
  ["Test-set evaluation", `held-out v2 test, n = ${A.split_counts.test}, scored once`, "s.13.1", "Met, low power"],
  ["Baseline vs improved comparison", "paired deltas, bootstrap CIs, sign tests, ablation", "paired_comparison.csv; s.14", "Met for Week 3; Week 4 not evaluated"],
  ["Error analysis", "rule-selected cases with root cause and mitigation", "error_cases.csv; s.16", "Met"],
  ["5-10 error cases", `${b.error_cases.length} cases (${b.error_cases.filter((e) => e.selection_reason.startsWith("failure")).length} failures + controls)`, "s.16", "Met (synthetic, no real cases)"],
  ["Robustness testing", `${Object.keys(sx).length - 1} notebook stress tests + invalid file + text suite + probes`, "robustness_results.csv; s.17-18", "Met"],
  ["Failure cases (Part G)", "D/E, F, C, paraphrase and wrong-explanation cases", "s.19", "Met"],
  ["Dataset documentation (9 items)", "source, size, format, labels, split, preprocessing, quality, privacy, licensing", "s.6-7, 23", "Met for synthetic data; real dataset pending"],
  ["Agreement with human graders", "no human marks exist", "s.24", "NOT met: requires real annotated data"],
  ["Reproducibility", "scripts, seeds, identical rerun", "s.27", "Met"],
  ["Limitations", "listed", "s.24", "Met"],
  ["Responsible AI", "privacy, bias, oversight, uncertainty, licensing", "s.23", "Met"],
  ["Group 3 stretch goal (output correct, explanation wrong)", "stress tests D/E", "s.19", "NOT met: not detected"],
], [24, 36, 20, 20]));

build("Group 3 Week 5: Evaluation Report", "Jupyter Notebook Lab Evaluator · Evaluation & Error Analysis", c, OUT);
