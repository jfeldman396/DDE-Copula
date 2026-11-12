Current revision endpoint audit and deep summaries

Run with explicit paths (Python 3 with requirements.txt dependencies):

python collect_revision_runs.py --root /absolute/study/root --output /absolute/audit/output
python summarize_deep_current.py --folder /absolute/audit/output

The study root must contain the exact current results subdirectories named in the collector. Superseded deep studies are deliberately excluded. Raw fits are not bundled here. The collector hashes each endpoint, validates dimensions, finite values and method settings, and records errors and completion status. An audit error prevents summary generation.

Dimension recovery uses every completed fit. Deep MSE requires all three dimensions correct and removes only values above Q3+1.5 IQR separately for each sample size, arm and layer. CSV outputs retain raw MSEs, eligible counts, excluded counts and per-replication decisions. Empty eligible cells have missing means, not zero MSE.

Partial summaries are monitoring outputs, not final manuscript results. Check DEEP_SUMMARY_STATUS.json before using them. A complete deep study has 1800 valid fits, 100 per sample-size/arm cell.

These files summarize the current revision runs; portable simulation entry points are documented separately.

Final output builders (explicit output folder required for portable use):
python summarize_j150_comparators.py --folder /absolute/audit/output
python build_deep_tables.py --folder /absolute/audit/output
python plot_deep_revision.py --folder /absolute/audit/output

Run summarize_deep_current.py immediately before the deep builders. Final deep tables and plots refuse incomplete or stale audit snapshots. For layout review only, add --preview; these outputs have explicit preliminary labels and separate filenames/directories. The comparator summarizer includes only fully completed methods.

Saved-checkpoint diagnostics (run after endpoint collection):
python audit_deep_checkpoints.py --root /absolute/study/root --output /absolute/audit/output

This separately records outer-loop stabilization, iteration caps, inner conditional-maximization flags, and KKT residuals. Stabilized and at-cap counts can overlap when stabilization occurs on the last permitted iteration. Endpoint completeness does not establish convergence; report these diagnostics with the finite-budget results. The audit opens only checkpoints having completed current-run endpoints.

Separate sensitivity studies:
python collect_deep_refits.py --root /absolute/study/root --output /absolute/refit/audit
python summarize_deep_refits.py --folder /absolute/refit/audit

The higher-threshold and original-floor gap+5 studies were stopped at the user's request, with102 and55 completed high-n fits respectively. Their original targets were300 each; do not restart them. The no-floor gap+5 study targets600 fits over all six sample sizes. The no-floor comparator is gap+5 with the original floor for n>=4000; at lower n it is the original pyramid baseline, which also differs in initialization. baseline_study identifies this distinction explicitly. Never pool these runs with each other or superseded studies. Summaries preserve method-specific and both-correct eligibility, full and true-signal MSE, independent trimming counts, and all-pair graph recovery. Partial summaries are monitoring only.

All-sample-size no-floor figures:
python plot_nofloor_study.py --folder /absolute/refit/audit

Final mode requires600 audited no-floor endpoints. Use --preview for explicitly preliminary figures. These show only the no-floor method, avoiding a single comparator curve that would mix baseline definitions across sample sizes.

Unavailable stopped-study comparator endpoints are represented by baseline_available=false and missing metric values, not failed fits. Summaries distinguish completed_refits, available_baseline_pairs, and the graph-specific evaluated denominator. The baseline_correct=false sentinel on an unavailable row must never be interpreted as a dimension-recovery failure. Recovery denominators for the baseline use available_baseline_pairs. Figure count sublabels are omitted; CSV counts and exclusion records remain available.

No-floor sensitivity tables: python build_nofloor_tables.py --folder /absolute/refit/audit. Final mode checks all100 replication IDs at each of six sample sizes. Use --preview for explicitly preliminary tables; these are not automatically inserted into submission documents.

No-floor stopping diagnostics: python audit_nofloor_checkpoints.py --root /absolute/study/root --output /absolute/refit/audit. Reads only checkpoints with completed no-floor endpoints and reports stabilization, cap hits, inner-solver flags and KKT residuals. Do not infer algorithmic convergence from endpoint completion.

Original gap+5 can yield zero top-layer width. Such attempts are recorded as initialization failures, never refit with changed widths. The audit now distinguishes completed fits, initialization_failed and attempts_resolved. Recovery uses resolved attempts including failures; conditional MSE uses valid dimension-correct fits only. Completion requires600 resolved outcomes. Failure IDs and reasons are preserved separately.

No-floor heatmap inputs: python nofloor_weight_review.py --root /absolute/study/root --folder /absolute/refit/audit [--preview]. Validates each matrix against stored replication MSE and matches the retained-set summaries. Layer1 truth is standardized per dataset before averaging. Mean-matrix error is reported separately from mean replication MSE. Final mode requires the completed audit.
