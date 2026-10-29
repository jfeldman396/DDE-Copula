# Publication validation

Publication baseline: `424427fe1bf06d06494172cf278deb7bd4b29463`. Target branch: `estimation-v2` only. The original main files and local study sources/results were not replaced. The unvalidated schedule refactor is excluded.

## Original source and historical evidence

- All 28 entries in the three original frozen manifests matched their source SHA-256 at audit and final preparation. Historical manifests retain predecessor entries even when those engines are intentionally absent from this compact package; `included_in_package` makes this explicit.
- All 50 distributed numerical MATLAB files match the local source bytes. The current engine, initialization, scoring, rank and objective helpers form a complete static closure of the recorded roots. No old top-level utility substitutes for the newer local version.
- The historical LSA validation JSON reports original short-trajectory, objective-history and restart checks. It was copied as evidence, not rerun or promoted into validation of the new wrapper.
- Nine compact evidence files were copied byte for byte from the completed paper-evidence capture (2026-10-29 01:20:03 UTC). They are not a live monitor. Active mechanism diagnostics, bulk fit records/checkpoints and posthoc sensitivities were excluded.

## New checks performed

1. MATLAB R2026a, single computation thread, static `requiredFilesAndProducts` on isolated copies: 50 MATLAB dependencies, all resolving within the package copy. Products: MATLAB and Statistics and Machine Learning Toolbox.
2. `test_v2_packaging`: SHA-256/size checks; weighted logistic objective and gradient equal explicit weighted observation-history calculations within 1e-12; absent configuration decay; objective accumulator serialization followed by the next update; rank relabeling and violation rejection; arm, temperature endpoint and stopping checks. All passed. These are deterministic helper checks, not fitted trajectories.
3. New entry points passed parse checks and guards rejecting an existing destination and duplicate jobs before any fitting. One error-identifier issue in the new guard was corrected and rechecked.
4. Python standard-library reconciliation: 216 primary cells, 100 outcomes per cell, 21,600 total outcomes, 45 optimizer-flagged outcomes retained and zero fatal failures. Source/evidence hashes passed.
5. The fresh-endpoint aggregation path was tested read-only against the saved original endpoint CSV. Reconstructed 216 all-setting rows, 54 tempering+CP rows and 54 per-n matched rows agree with published tables within absolute/relative tolerance 1e-12. The input endpoint CSV is deliberately not published.
6. Base R regenerated all three figure sets from published compact tables. PNGs and rendered PDFs were visually checked for layout and labels. An optional SVG attempt failed due to unavailable Cairo/X11 libraries; that option was removed. The final PNG/PDF run completed without warnings. A transient R default-device artifact from the failed attempt was removed from the source directory; no existing study source or result was altered. PDF rendering required font fallback but completed and was visually checked.

## Limits

No estimator or simulation was run for publication. No full-fit, exact restart, cached-input, endpoint or runtime parity test was run for the new wrapper. It uses unchanged numerical source but different orchestration (fresh runs, no prefix reuse). Do not infer exact end-to-end parity from byte identity or deterministic helper checks.

The serial study entry point has no claim scheduler, automatic resume or retry. The summary entry point refuses incomplete cases. The complete-endpoint table path supports the primary and per-n matched figure tables; supplementary bootstrap intervals, all-six-n cohorts and sensitivity analyses remain supplied historical tables rather than regenerated analyses. CPU is serial-equivalent study accounting, not a controlled hardware comparison. No convergence/mixing claim follows from the finite-run MSE results.

Rerun portable checks from the repository root using the commands in `v2/README.md`. Fits require a separate explicit invocation. The original study remains with its existing simulation owner.
