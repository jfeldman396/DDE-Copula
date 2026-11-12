DEEP NO-FLOOR ENDPOINT ANALYSIS

These scripts analyse the completed no-floor study, not the older CP study.
Run from this directory or use full script paths. Install requirements.txt.
Each command below requires explicit external input/output paths.

  python collect_deep_refits.py --root /absolute/study --output /absolute/audit
  python summarize_deep_refits.py --folder /absolute/audit
  python build_nofloor_tables.py --folder /absolute/audit
  python plot_nofloor_study.py --folder /absolute/audit
  python audit_nofloor_checkpoints.py --root /absolute/study --output /absolute/audit
  python nofloor_weight_review.py --root /absolute/study --folder /absolute/audit

The study root must contain results/manuscript_deep_nofloor_20261104 with
full_n{n}_r{rep}_a1_result.mat and recorded failure files, matching inputs in
results/manuscript_deep_pyramid_20261103/inputs, and truth in
results/deeper_paper_rerun/truth.mat. The collector reads scored endpoint
exports (metrics and hp), not raw result.mat files from arbitrarily named
run_deep_saved_case directories. Full endpoints are external to this package.

Only the no-floor study is collected. Optional baseline records, when present,
are used for secondary matched analyses and must never be pooled with refits.
Missing baselines are missing values, not failed fits. Main no-floor MSE uses
method_specific_correct/refit rows, all three dimensions correct, and separate
upper Q3+1.5 IQR trimming by n/layer. Raw metrics, counts and excluded IDs are
retained in analysis outputs. The 600 attempts include 599 fits and one known
initialization failure; do not replace the failure by changing initialization.

Final builders reject incomplete or stale audits. --preview produces explicitly
preliminary outputs. Checkpoint diagnostics distinguish stabilization, budget
caps, inner-solver flags and KKT residuals; completion is not convergence.
render_nofloor_weight_review.m renders prepared heatmap inputs; see its function
signature. For bundled aggregate results without endpoints, use the simpler
Scripts/Publication/reproduce_summary_results.py command in Scripts/REPRODUCE.txt.
