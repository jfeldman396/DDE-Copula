# Current estimator and reproduction

## Scope and versions

`CopDDE/LayerSpecificSAEM/lsa_fit.m` is the current estimator, preserved byte for byte (SHA-256 `6464c65b08e0fc1075c70f64203608acdc14b9a4664bdea4c18dc0a70d3ae7ba`). Its current dependency closure is included under the original relative folder names. Historical manifests document predecessor versions without distributing their experimental engines. `provenance/SOURCE_MANIFEST.json` identifies every copied source, its hash and static-analysis roots. The three study manifests retain all component hashes; their file paths alone were normalized to repository-relative paths. Their original manifest hashes are recorded.

Do not mix the old top-level `Utilities/` with `v2/CopDDE/Utilities/`: local `Normal_init_v2.m` and `simulate_gaussian_mixture.m` differ from the GitHub baseline versions. The new entry point controls its source path and restores the caller's path afterward.

## Methods

| Engine and label | Gaussian latent update | Averaged likelihood components |
| --- | --- | --- |
| Current `lsa_fit`, H1 | Two collapsed-maxima sweeps, sequential traversal of rank-group maxima | Gaussian, logistic and top-layer Bernoulli |
| Current `lsa_fit`, H0 | Same collapsed-maxima update | None; raw objectives |
| Current `lsa_fit`, Z | One sequential rank-group Gibbs sweep | Gaussian, logistic and top-layer Bernoulli |
| Historical `pfs_fit`, H1 | Collapsed maxima | Gaussian only |
| Historical `pfs_fit`, Z | Rank-group Gibbs | None |

`cma_column(...,'sequential')` refers to maxima traversal; it is not the rank-group Gibbs algorithm. The default current H1/Z behavior uses full layer-specific averaging. A direct call with `fullAverage=false` changes that behavior and is not the current H1/Z paper protocol. No claim of independent exact posterior draws is implied by a Gibbs/slice transition.

Blocked binary updates use the sampled Z. CP reaches one at iteration 60; tempering reaches one at iteration 30. After warmup the binary conditionals target the untempered full model. `pfs_arm` defines 12 arms: IDs 1–4 H1, 5–8 H0, 9–12 Z, each in order tempering+CP, CP-only, tempering-only, neither. This is not a complete sampler-by-averaging factorial: there is no raw-objective Z arm in the current study.

Warmup is 60 iterations with CP, 30 with tempering alone, and zero otherwise. SA activates at warmup+21. The first ten SA updates have rho=1, followed by `(k-9)^(-0.7)`. `foa_accumulate` retains weighted configuration counts and response sums, decaying configurations absent from the current draw; `foa_cm` uses their weighted logistic objective. Factor identities are preserved during accumulation.

Stopping compares consecutive ten-iteration means: relative Frobenius changes at most 0.5% for B1/B2 and top probabilities, 1% for gamma; active identities must stay fixed across the 20 iterations. Checks start after 40 post-warmup iterations, occur every five iterations and require three consecutive passes. The cap is 100 post-warmup iterations. Stabilization is a numerical criterion, not proof of posterior convergence or mixing.

## Requirements and non-computational checks

Packaging was checked with MATLAB R2026a and Statistics and Machine Learning Toolbox. The static analyzer identified these products; the source also uses MATLAB's JVM for hashes and filesystem locks. Earlier MATLAB versions were not tested. Python 3 standard library is sufficient for the compact table reproduction script.

From the checkout root:

```sh
python3 v2/reproduce_tables.py
python3 v2/reproduce_tables.py --output /absolute/new/primary_table.csv
matlab -singleCompThread -batch "addpath('v2'); test_v2_packaging"
```

The Python check verifies source/evidence hashes and rebuilds the 216-cell primary table from the published compact summary. It does not recompute fits or bootstrap intervals from endpoint records. The MATLAB check verifies the isolated dependency closure and deterministic helper identities; it runs no estimator, simulation or parallel pool. Temporary test files are removed afterward.

## Run one fresh case

This command **starts numerical estimation**; it was not run as part of packaging:

```matlab
addpath('v2');
result = run_v2_case(50, 500, 1, 1, '/absolute/new/dde_case_J50_n500_r1_a1');
```

Use a new absolute destination that does not already exist. The entry point verifies the published source hashes, copies only the source into that destination, generates the paper's synthetic input, and runs one arm using the unchanged `lsa_fit`. All legacy relative paths and configuration-directory side effects are confined to that new destination. It restores the caller's path, current directory and RNG state. Invoke MATLAB with `-singleCompThread` for the study's single-thread execution convention. No pool is started.

Arguments are J in 50/100/150; n in 500/1000/2000/4000/8000/16000; replication in 1:100; arm ID in 1:12. These reproduce the study configuration, seed formulas and numerical engine, not its scheduling infrastructure. Input initialization and caches are rebuilt independently; exact equality to unavailable original cached inputs and endpoints has not been tested. The wrapper intentionally computes a fresh trajectory rather than reusing production prefix checkpoints. Its elapsed cost therefore differs from the original batch's shared-prefix accounting.

The run directory contains `job.mat` (configuration and source manifest), the cached input, `checkpoint.mat` (full numerical restart state) and `result.mat` (fit and metrics). A failed run can leave a checkpoint; do not delete or silently overwrite it. The wrapper refuses pre-existing destinations and is not a restart controller. It also does not claim to reproduce the production collector's endpoint record schema.

## Resume and production boundaries

Direct `lsa_fit` calls can resume its own `record` checkpoint. It deterministically reseeds each iteration phase, but exact resume also requires the same input, arm, source, fullAverage/forceOne options and every latent/CSP/objective/history/stopping field. Its loader does not independently check all these identities and can initialize a missing objective accumulator. Do not migrate predecessor checkpoints or arbitrary records into the current study.

The historical production launcher, claims, release markers and absolute-path freeze checks are not exposed as a ready-to-run grid launcher here. Predecessor engines, shared-prefix orchestration and the old production collector are deliberately excluded. `summarize_v2` reads only the fresh wrapper's result schema and refuses incomplete case lists; it does not reinterpret old checkpoints or silently omit failed cases. Never create validation/release markers merely to bypass the production guards.

The saved `historical_LSA_VALIDATION.json` reports checks previously performed on the original engine, including short trajectories and resume. It is historical evidence, not a new test of this package. New checks are described in [PACKAGING_VALIDATION.md](provenance/PACKAGING_VALIDATION.md). Full trajectory/restart parity for the wrapper and end-to-end regeneration of the study remain untested. Results for unfinished post-warmup diagnostics are deliberately excluded.

## Small public interface

The public surface is `run_v2_case` (simulate the paper input and fit one arm), `run_v2_study` (explicit serial case list), `summarize_v2` (fresh outcomes), `reproduce_tables.py` (frozen-table checks) and `plot_results.R` (paper figures). The 50 internal MATLAB files are the static dependency closure of the current engine, initialization and scoring. Some generic helpers retain unused branches because changing them would invalidate source identity; no exploratory experiment drivers are shipped. Consolidating numerical helpers is deliberately deferred. [METHODS.md](METHODS.md) gives the maxima derivation and implementation pseudocode.

To run a chosen case list (this starts numerical work):

```matlab
addpath('v2');
jobs = table(50,500,1,1,'VariableNames',{'J','n','replication','arm_id'});
run_v2_study('/absolute/new/study',jobs);
summarize_v2('/absolute/new/study','/absolute/new/endpoints.csv');
```

For the full published grid, construct the explicit jobs table below and pass it to `run_v2_study`. This is **21,600 fits**, potentially very expensive in this simple serial launcher; it is not a recommended smoke test and was not run during publication. The launcher has no distributed claims, automatic retry or resume. A failure stops it, preserving partial outputs.

```matlab
[J,N,R,A] = ndgrid([50 100 150],[500 1000 2000 4000 8000 16000],1:100,1:12);
jobs = table(J(:),N(:),R(:),A(:),'VariableNames',{'J','n','replication','arm_id'});
```

Reproduce the three published descriptive figures from compact evidence, without fitting (base R, no extra packages):

```sh
Rscript v2/plot_results.R v2/evidence /absolute/new/figures
```

The script writes PNG and PDF files to a new directory and refuses overwrites. Its plotting calculations match the paper evidence script; only input/output handling is adapted. After a complete fresh grid, export `endpoints.csv` with `summarize_v2`, then rebuild the primary and per-n matched figure tables:

```sh
python3 v2/reproduce_tables.py --endpoints /absolute/new/endpoints.csv --outdir /absolute/new/tables
Rscript v2/plot_results.R /absolute/new/tables /absolute/new/figures
```

The endpoint reaggregator requires the full 21,600 distinct complete cases and retains every outcome. It was validated against the saved historical endpoint CSV: all 216 primary rows, 54 tempering+CP rows and 54 matched rows agree to tolerance 1e-12. No fits were rerun. It does not rebuild the supplementary bootstrap intervals, all-six-n matching or posthoc sensitivity tables; those published uncertainty tables remain historical evidence. Full numerical regeneration of the study is untested.
