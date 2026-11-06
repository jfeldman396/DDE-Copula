Deeper no-floor fit: saved-input reproduction

Add the v2 directory to the MATLAB path, then call:
  result = run_deep_saved_case('/absolute/input.mat', '/absolute/new_output', 160);
The output directory must not already exist. The input must contain the
synthetic study's saved input structure, including initial states and seeds.
The wrapper copies the published kernels into the new output directory and
preserves the caller's working directory, path, and random-number state.

Protocol: maxima + tempering + CAVI; beta0=.9; warmup60; SA starts81;
spikes [.01 .03 .005], concentrations [36 12 4], tau=3*n^(-.3), cap160.
The saved input determines the initial widths. This entry point does NOT
regenerate data, choose an architecture, count initialization failures,
or compute aligned MSE. Use the separate analysis tools for final audits.
Do not describe this saved-input entry point as full end-to-end reproduction.

DEEP_SOURCE_MANIFEST.json records byte-identical production fitting and metric kernels.
Baseline and CAVI manifests cover their published dependencies. The first
validation compares three warmup iterations exactly. A separate averaging
validation report is produced only after successful comparison through95
iterations, including the decaying SA schedule. Neither is a full-study rerun.

Do not use this wrapper to resume the explicitly stopped higher-threshold
or original-floor comparison studies. Preserve their partial results.

Scoring an existing fit (no files are modified):
  metrics = score_deep_saved_case(resultFile, inputFile, truthFile);
Use the matching archived synthetic input and truth. The routine reproduces
sequential positive-loading alignment, maximum-width zero padding, final-Z
scaling for layer1, and per-replication truth-Z scaling. Intercepts are
included. MSE on exact model-moment truth is a separate diagnostic.
Eligibility and upper-IQR trimming are applied later by the analysis tools.
