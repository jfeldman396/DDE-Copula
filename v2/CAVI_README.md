# Maxima with CAVI warmup

This additive entry point preserves the published baseline kernels and adds the four unchanged CAVI source files used in the completed study. From MATLAB with `v2` on the path, call:

```matlab
result = run_cavi_case(50,500,1,1,'/absolute/new/output');
```

Arms: 1 maxima + tempering + CAVI; 2 maxima + CAVI only; 3 maxima + tempering only. All use full likelihood-objective stochastic averaging. The optional sixth argument limits iterations for smoke checks; such outputs explicitly report `complete=false` unless stabilization or the full cap is reached. Do not count partial smoke checks as study endpoints.

The wrapper isolates source and output, verifies both manifests, and restores MATLAB path, working directory, and RNG state. It generates fresh data using the existing study seed rules; it does not read personal directories or original production caches.

CAVI approximates binary block conditionals only during warmup, with exact block Gibbs from iteration60. Tempering reaches unit inverse temperature by iteration30. Full objective averaging follows warmup and20 further iterations. The original `v2` documentation describes CP, not this CAVI extension.

Isolated fresh-data checks (J=50, n=500, replication 1, three iterations) passed for all three arms. Each matched the production implementation exactly for latent states, loadings, shrinkage states, histories and objective accumulators. Source verification, rank consistency and partial-run labeling also passed. An additional 85-iteration arm 1 check passed exact production parity through CAVI warmup and likelihood-objective accumulation; see `provenance/CAVI_AVERAGING_VALIDATION.json`. The short all-arm checks are recorded in `provenance/CAVI_PACKAGING_VALIDATION.json`. These checks do not establish full-study statistical reproduction. A separate [saved-input deeper fitting and scoring interface](deep/README.txt) is now available with its own validation records. Deep fresh-input generation now passes exact low/high-n checks and reproduces the known initialization failure; see `provenance/DEEP_INPUT_VALIDATION.json`. Saved-input comparator interfaces are available for [original truncated-lasso RL-SAEM](comparators/original_rl/README.txt) and [Poisson](comparators/poisson/README.txt), each with one full-case exact endpoint validation. These checks do not establish reproduction of every study endpoint.
