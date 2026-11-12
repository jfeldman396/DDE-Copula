# Proposed DDE model: fitting and implementation

The proposed MAP estimator combines **CAVI and tempering during warm-up,
followed by stochastic approximation averaging after burn-in**. Use **arm 1**
(`map`) for the combined method. Arms 2 and 3 are the CAVI-only and
tempering-only warm-up ablations used in the supplement.

## Run one complete example

Start MATLAB in the repository root, with the required toolboxes installed.
This example generates the paper's synthetic data for J=50, n=500 and
replication 1, then fits the combined method at its full published budget:

```matlab
addpath('Algorithms/DDE', 'Scripts/DDE');
result = run_cavi_case(50, 500, 1, 1, '/absolute/new/dde_example');
fit = result.fit;
```

The fourth argument is the arm number: **1 selects CAVI + tempering +
post-burn-in averaging**. Choose a new absolute output directory. Omit the
optional iteration-limit argument for a complete fit. The shallow study's
cap is 160 iterations; the existing stability rule may stop it earlier.

`result.mat` contains the fitted state and dimension-recovery metrics;
`checkpoint.mat` retains the state for reproducibility. The raw state has
`B1`, `B2`, `gamma`, `prop`, latent states and active-dimension indicators.
The first column of each B matrix is its intercept. These are the padded
fitting matrices; the study-specific export/scoring functions handle active
columns, alignment and scaling. `fit.trace` records the schedule and
numerical diagnostics at every iteration. Check `fit.stabilized` separately
from reaching the iteration cap.

## What the combined algorithm does

1. **Warm-up, iterations 1–60.** The inverse temperature beta starts at 0.7
   and reaches 1 at iteration 30 in this shallow example. Binary block
   updates mix the CAVI product distribution with the exact
   block conditional. The exact-conditional mixture weight is
   `lambda = min(1, (iteration - 1)/59)`; CAVI's contribution reaches zero
   at iteration 60. Gaussian latent variables use the group-maxima update.
2. **Post-warm-up burn-in, iterations 61–80.** Updates use beta=1 and
   lambda=1. Parameters are still updated from the current latent draw.
3. **Stochastic averaging, from iteration 81.** The Gaussian sufficient
   statistics and the binary complete-data objective are accumulated and
   used in the penalized conditional maximization steps for MAP estimation.
   With `k = iteration - 80`, the step size is 1 for k=1,...,10, then
   `rho = (k - 9)^(-0.7)`. Thus the accumulator is initialized from iteration
   81, and averaging across successive draws begins at iteration 91.

The averaged quantities include the Gaussian cross-products, the binary
configuration counts/response totals and top-layer occupancies. This is
stochastic approximation of the MAP objective; it does not produce posterior
credible intervals or an arithmetic average of the fitted B matrices.
The separate posterior-sampling scripts handle the supplement's interval
analyses. The excluded variational-interval pilot is separate from the
CAVI warm-up used by the proposed estimator, which remains included.

## Where the implementation lives

- [run_cavi_case.m](../../Scripts/DDE/run_cavi_case.m): runnable shallow
  example, input generation and combined-method selection.
- [full_cavi_fit.m](cavi/full_cavi_fit.m): complete MAP fitting loop,
  warm-up schedule, accumulators and parameter updates.
- [full_cavi_binary.m](cavi/full_cavi_binary.m) and
  [pilot_mix.m](cavi/pilot_mix.m): CAVI/exact-conditional mixture.
- [pfs_arm.m](CopDDE/PaperFactorialStudy/pfs_arm.m): arm settings,
  warm-up length, averaging start and iteration cap.
- [cpdde_engine_partitioned.m](../EmpiricalDDE/cpdde_engine_partitioned.m):
  empirical fitting engine with explicit tuning parameters and partitions.

The shallow runner invokes the engine with `fullAverage=true`,
`forceOne=false` and `useCAVI=true` for arm 1. Keep these settings to retain
averaging of both the Gaussian and binary objectives.

## Reproduce the paper's other settings

Use [Scripts/REPRODUCE.txt](../../Scripts/REPRODUCE.txt): section 3 gives
shallow simulations and comparisons, section 4 the deeper model, section 5
ordinal examples, and section 7 the selected survey model. The selected
survey command is `run_empirical_case(..., 'maxima', 9, ...)`; it also uses
CAVI, tempering and stochastic averaging, with the survey-specific tuning
configuration. Contact Joe Feldman for its original data.

The example above generates a paper simulation; it is not an arbitrary-data
loader. The survey reproduction wrapper checks the exact original data.
Adapting a new dataset requires a suitable initializer and input structure
for the fitting engine; the two example wrappers should not be treated as
a general-purpose `fit(Y)` interface.

MATLAB R2026a, Statistics and Machine Learning Toolbox, Optimization Toolbox, and a JVM are required. Comparator entry points also use Parallel Computing Toolbox functions to check for an active pool. Numerical functions retain their research-source names and dependency folders to avoid changing function resolution. Names containing CP or earlier study labels can still identify helpers used by the current estimator; they do not mean the obsolete CP experiment is included.

`verify_revision_sources.m` checks MATLAB code and synthetic truth inputs against the single [package manifest](../../Results/Publication/MANIFEST.json). Numerical kernels were preserved during packaging; the full simulation grid has not been rerun to completion. The current verification uses a representative subset at the author's request. The saved-result checks are separate from fitting validation and convergence assessment.

For the deep model, input generation uses gap+5 widths and records a zero top-layer width as an initialization failure. Fitting uses beta0=.9, 60 warmup iterations, averaging from iteration 81, spikes [.01 .03 .005], concentrations [36 12 4], threshold 3*n^(-.3) without a floor, and cap 160. Do not replace the known n=500, replication=76 initialization failure with a different initialization.

To score a deep fit, add `Algorithms/DDE` and `Scripts/DDE` to the MATLAB path, then call `score_deep_saved_case(resultFile,inputFile,truthFile)` with absolute paths. The truth is `deep/input_generation/truth.mat`. Scoring includes sequential positive-loading alignment, width padding, intercepts, final-Z scaling of estimates and per-replication truth-Z scaling. Dimension eligibility and upper-IQR trimming occur in the analysis scripts, separately for each sample size and layer.
