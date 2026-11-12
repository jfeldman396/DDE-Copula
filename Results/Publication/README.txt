PAPER CODE AND AGGREGATE RESULTS HANDOFF

The repository keeps its original Algorithms / Utilities / Scripts / Results
layout. This directory contains small publication results, figure/table maps,
and provenance. Estimation code is separate from saved-result analysis.

STATUS
The native15/5 comparator study is complete and numerically and visually validated.
See data/native15_5/README.txt for captions, timing definitions, and limitations.
All 12 figures are in figures/native15_5. The configuration file retains its
historical native15_5_pending.json filename but records COMPLETE_VALIDATED.
Existing top-level legacy Results files remain historical assets.

CONTENTS AND ENTRY POINTS
* DDE: existing Algorithms/DDE/ estimation code, saved-input interfaces, and provenance.
  See Algorithms/DDE/README.md, Algorithms/DDE/deep/README.txt and Algorithms/DDE/comparators/*/README.txt.
  data/dde contains MAP dimension recovery and native15/5 truth matrices.
  data/deep contains only the no-floor tempering+VI study:599 fitted endpoints
  and1 initialization failure among600 attempts. Stopped studies are excluded.
* Gaussian comparison: Algorithms/GaussianCopula and Utilities/GaussianCopula
  contain the exact research fitter/scorer with path-only portability changes.
  data/factor contains all6 candidates,300 pMSE scores, standardized loading
  matrices, selection records and mixing diagnostics. Candidate6 has the
  minimum mean pMSE in this finite grid; posterior mixing remained inadequate.
* iVAE: Algorithms/iVAE contains the local research source with its MIT notice,
  Gaussian broadcasting patches and an explicit opt-in training entry point.
  It uses ten hidden units in each network and one latent dimension, seed0,
  batch256, Adam learning rate.01, and the original2000-update budget checked
  at epoch boundaries (2004 actual updates). data/ivae contains aggregate
  network weights, plotting matrices, item metadata and training provenance.
* data/uncertainty contains the completed40-dataset summaries and diagnostics;
  all40 failed the mixing screen. Interval enclosure is not established
  posterior coverage. Completion of a fixed budget does not prove convergence.
* FIGURE_TABLE_MAP.csv maps the30 supplement assets/tables, with comparator
  assets linked to the completed native15/5 results. MAIN_PAPER_MAP.csv refers to the supplied PDF15
  snapshot; live manuscript numbering is not claimed. No manuscript is bundled.
  Preserved-only artwork is identified where exact renderers were not verified.

SAFE SAVED-RESULT CHECKS
Run from the repository root, or use absolute script paths. Python3.11+ is
required; the validation runtime and exact tested package versions are recorded.

  python -m pip install -r Scripts/Publication/requirements-render.txt
  python Scripts/Publication/validate_handoff.py
  python Scripts/Publication/render_ivae.py --output-dir /chosen/new/output/ivae
  python Scripts/Publication/render_factor_loadings.py --output-dir /chosen/new/output/factor

These commands validate/replot existing aggregate results only. They do not
train networks, run chains, generate new datasets or touch ongoing workers.
The iVAE display uses the DDE shallow-loading row order. Hidden units are
permuted without changing their fitted signs. The Gaussian comparison permits
sign/column alignment. The two plotting scales are described in the figures.

OPTIONAL FITTING (NOT RUN TO VALIDATE THIS HANDOFF)
MATLAB R2026a and Statistics and Machine Learning Toolbox were used for the
Gaussian comparison. DDE also requires the toolboxes listed in Algorithms/DDE/README.md,
including Optimization Toolbox. Raw survey inputs are external; see
EXTERNAL_INPUTS.json for hashes, expected schemas and preprocessing.

  addpath('Scripts/Publication');
  run_gaussian_study('external_train_input.mat','new_external_result_directory');

This explicitly runs the archived4k stage, copies its new checkpoints, and
continues the same two chains to8k for each of6 candidates. It preserves the
4k rotation reference and fixed seeds. It performs no success-driven extension.
The sampling and scoring bodies are unchanged; path adaptations are recorded
in SOURCE_MANIFEST.json. The handoff has not revalidated full experimental parity.

For iVAE, install Scripts/Publication/requirements-ivae.txt in a separate
environment, then supply the original external survey CSV:

  python Algorithms/iVAE/train.py --train --data external_survey.csv --output-dir /chosen/new/training/output

Without --train this command prints configuration and exits. It checks the
original input hash and uses CPU execution. It exports aggregate network weights
and loss only. No respondent-level embeddings, inputs or training checkpoints
are bundled. Historical bitwise runtime reproducibility is not claimed.

SCIENTIFIC LIMITS
The selected DDE empirical10/1 fit was user-selected and third-ranked in its
grid, not the minimum-pMSE fit. Empirical pMSE is fitted-sample utility, not
held-out performance. The Gaussian comparison retains poor-mixing caveats.
iVAE uses ideology as an auxiliary variable while DDE excludes it; this is a
descriptive comparison with unequal information. The latent dimension is one;
the ten displayed columns are hidden-unit weights.

The completed native15/5 MSE comparison uses Jx16 and15x6 matrices including
intercepts; MAP requires jointly correct10/3 dimensions, comparators use all
valid replications, and upperQ3+1.5IQR exclusions are method/J/n/layer-specific.
Counts, excluded IDs, untrimmed errors and both matrix-mean versions must remain
available. Poisson log-mean coefficients are a different estimand from Gaussian
latent-scale coefficients. No cropped wider-model fits substitute for reruns.

PUBLICATION SCOPE
Only this additive handoff and its entry points are new. Existing repository
history/files are preserved. Raw survey data, respondent embeddings, personal
data, private manuscripts, large endpoints/chains, controllers and stopped
experiments are excluded. No new project-wide license is assigned; see
THIRD_PARTY_NOTICES.txt. The iVAE upstream commit is unavailable for the local
research copy; exact file hashes and its existing MIT notice are preserved.

CURRENT DOCUMENT SNAPSHOT (2026-11-12)
DOCUMENT_ASSET_MAP.csv covers all active figure references in the supplied manuscript, supplement, and responses. CONSISTENCY_NOTES.txt records wording conflicts rather than silently changing measurements.
python Scripts/Publication/export_document_figures.py --output-dir /chosen/new/figure_export
python Scripts/Publication/render_native15_5.py --output /chosen/new/native_plots

Additional fitting entry points:
addpath('Algorithms/Native15_5'); run_native15_5_case(inputFile,'rl',newOutputDir)
addpath('Algorithms/Native15_5'); run_native15_5_case(inputFile,'poisson',newOutputDir)
addpath('Algorithms/PosteriorGibbs'); run_posterior_gibbs(a,fullMapState,options,newOutputDir)
These require the full saved input/state, not only aggregate CSVs. Sampler options are seed, iterations, spike1, spike2, alpha1, alpha2. Simulation seed formula: 930031+J*10000+n+chain*1000000+replication*17000000. The B5 reported first chain uses its documented independent replacement seed, not the original seed.
