# DDE-Copula: estimation-v2

This branch publishes the current **layer-specific SAEM rank-likelihood estimator**, its frozen source versions, and compact results from the completed 21,600-fit simulation study. The original `main` implementation remains unchanged. The original top-level `Algorithms/`, `Utilities/`, `Scripts/` and `Results/` directories are retained for historical use; use **`v2/`** for the current estimator.

## Start here

- [Estimator and reproducibility guide](v2/README.md): dependencies, one-case entry point, method definitions, checkpoints and limitations.
- [Algorithm derivation and pseudocode](v2/METHODS.md): CP, exact-target updates, collapsed maxima and layer-specific averaging.
- [Results and interpretation](v2/evidence/README.md): completed-study tables, denominators, flags and caveats.
- [Primary table, all settings](v2/evidence/primary_all_settings.csv) and [tempering + CP](v2/evidence/primary_tempering_CP.csv).
- [Source manifest](v2/provenance/SOURCE_MANIFEST.json) and [publication validation](v2/provenance/PACKAGING_VALIDATION.md).

The numerical source was copied without modification from the audited local study. All 28 frozen-manifest entries match. New packaging and deterministic helper checks passed; the packaging entry point has **not** been validated by rerunning the full estimator or simulation grid. The proposed schedule refactor is **not included**.

Quick checks (no simulations):

```sh
python3 v2/reproduce_tables.py
matlab -singleCompThread -batch "addpath('v2'); test_v2_packaging"
```

A fit is an explicit separate action; see the guide. Bulk fit files, checkpoints, private/local data, machine credentials and ongoing diagnostic results are not part of this publication.
