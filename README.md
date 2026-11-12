# DDE-Copula: estimation-v2

This branch publishes the current **layer-specific SAEM rank-likelihood estimator**, its frozen source versions, and compact results from the completed 21,600-fit simulation study. The original `main` implementation remains unchanged. The original `Algorithms/`, `Utilities/`, `Scripts/` and `Results/` layout is retained. The revised DDE estimator is in **`Algorithms/DDE/`**, alongside the other algorithms.

## Start here

- [Estimator and reproducibility guide](Algorithms/DDE/README.md): dependencies, one-case entry point, method definitions, checkpoints and limitations.
- [Algorithm derivation and pseudocode](Algorithms/DDE/METHODS.md): CP, exact-target updates, collapsed maxima and layer-specific averaging.
- [Results and interpretation](Algorithms/DDE/evidence/README.md): completed-study tables, denominators, flags and caveats.
- [Primary table, all settings](Algorithms/DDE/evidence/primary_all_settings.csv) and [tempering + CP](Algorithms/DDE/evidence/primary_tempering_CP.csv).
- [Source manifest](Algorithms/DDE/provenance/SOURCE_MANIFEST.json) and [publication validation](Algorithms/DDE/provenance/PACKAGING_VALIDATION.md).

The numerical source was copied without modification from the audited local study. All 28 frozen-manifest entries match. New packaging and deterministic helper checks passed; the packaging entry point has **not** been validated by rerunning the full estimator or simulation grid. The proposed schedule refactor is **not included**.

Quick checks (no simulations):

```sh
python3 Algorithms/DDE/reproduce_tables.py
matlab -singleCompThread -batch "addpath('Algorithms/DDE'); test_v2_packaging"
```

A fit is an explicit separate action; see the guide. Bulk fit files, checkpoints, private/local data, machine credentials and ongoing diagnostic results are not part of this publication.


## Publication code and aggregate results

The handoff follows the original `Algorithms`, `Utilities`, `Scripts`, and `Results` layout. Start with [Results/Publication/README.txt](Results/Publication/README.txt) for DDE, Gaussian copula factor-model and 10/1 iVAE code, saved results, configuration/seeds and figure/table mappings. Completed native 15/5 comparator plots, per-replication errors, average matrices, and J=100 MAP timings are available in Results/Publication/data/native15_5 and Results/Publication/figures/native15_5. Existing legacy files remain historical assets.

## Revised DDE (tempering + VI)

```matlab
addpath('Algorithms/DDE');
run_cavi_case(50,500,1,1,'/absolute/new/output');
```

Arm 1 is tempering + VI; arm 2 is VI only; arm 3 is tempering only. See [the DDE guide](Algorithms/DDE/CAVI_README.md). Existing function names containing `v2` are retained for compatibility; there is no separate top-level `v2` folder.
