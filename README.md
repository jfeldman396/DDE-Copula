# DDE-Copula: revision and supplement reproduction

**Start with [Scripts/REPRODUCE.txt](Scripts/REPRODUCE.txt)** for the result-by-result commands, required inputs, output filenames and reproduction limits.

This branch retains the original repository's algorithms, utilities, scripts and results, and adds the material used by the revision and supplement:

- `Algorithms/`: revised DDE, Poisson comparison, Gaussian copula and iVAE implementations, and posterior Gibbs sampling. The truncated-lasso comparator reuses the original algorithm.
- `Scripts/`: simulation entry points, saved-result analysis and figure rendering.
- `Results/Publication/`: numerical results, settings needed by the Gaussian study, publication figures and result mappings.
- `Utilities/`: original helpers and Gaussian copula helpers.

The older CP simulation package, superseded comparator copies and historical development reports are omitted. Related candidate/dataset records are consolidated without dropping their numerical values or mixing diagnostics. Source/result integrity is checked with `python Scripts/Publication/validate_handoff.py`.

The original repository baseline is commit `424427fe1bf06d06494172cf278deb7bd4b29463`; its 46 files remain, with only additive ignore rules in `.gitignore`. No manuscript source or raw survey data is included. Full refits needing external data/checkpoints are identified in the guide; the package does not claim end-to-end reproduction of every result from aggregate files alone.
