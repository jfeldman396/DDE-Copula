# Revised DDE estimator

Start with [the reproduction guide](../../Scripts/REPRODUCE.txt) for commands, inputs and outputs. The shallow simulation uses `Scripts/DDE/run_cavi_case.m`: arm 1 tempering + VI, arm 2 VI only, arm 3 tempering only. `cavi/full_cavi_fit.m` is the fitting engine. The deeper no-floor study uses `deep/revision_deep_fit.m` through `Scripts/DDE/run_deep_saved_case.m`.

MATLAB R2026a, Statistics and Machine Learning Toolbox, Optimization Toolbox, and a JVM are required. Comparator entry points also use Parallel Computing Toolbox functions to check for an active pool. Numerical functions retain their research-source names and dependency folders to avoid changing function resolution. Names containing CP or earlier study labels can still identify helpers used by the current estimator; they do not mean the obsolete CP experiment is included.

`verify_revision_sources.m` checks MATLAB code and synthetic truth inputs against the single [package manifest](../../Results/Publication/MANIFEST.json). Numerical kernels were preserved during packaging; whole-study fitting has not been rerun to validate the trimmed package. The saved-result checks are separate from fitting validation and convergence assessment.

For the deep model, input generation uses gap+5 widths and records a zero top-layer width as an initialization failure. Fitting uses beta0=.9, 60 warmup iterations, averaging from iteration 81, spikes [.01 .03 .005], concentrations [36 12 4], threshold 3*n^(-.3) without a floor, and cap 160. Do not replace the known n=500, replication=76 initialization failure with a different initialization.

To score a deep fit, add `Algorithms/DDE` and `Scripts/DDE` to the MATLAB path, then call `score_deep_saved_case(resultFile,inputFile,truthFile)` with absolute paths. The truth is `deep/input_generation/truth.mat`. Scoring includes sequential positive-loading alignment, width padding, intercepts, final-Z scaling of estimates and per-replication truth-Z scaling. Dimension eligibility and upper-IQR trimming occur in the analysis scripts, separately for each sample size and layer.
