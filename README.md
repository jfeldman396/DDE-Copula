# DDE-Copula: revision and supplement reproduction

**Start with [Scripts/REPRODUCE.txt](Scripts/REPRODUCE.txt)** for the result-by-result commands, required inputs, output filenames and reproduction limits.

To run all packaged saved-result workflows after installing the rendering requirements:

```sh
python Scripts/reproduce.py --output-dir reproduced/revision
```

The output `RESULT_INDEX.csv` distinguishes regenerated figures, exported numerical inputs and preserved artwork. The guide explicitly lists missing external inputs and analysis routes for full refits.

This branch retains the original repository's algorithms, utilities, scripts and results, and adds the material used by the revision and supplement:

- `Algorithms/`: revised DDE, Poisson comparison, Gaussian copula and iVAE implementations, and posterior Gibbs sampling. The truncated-lasso comparator reuses the original algorithm.
- `Scripts/`: simulation entry points, saved-result analysis and figure rendering.
- `Results/Publication/`: numerical results, settings needed by the Gaussian study, publication figures and result mappings.
- `Utilities/`: original helpers and Gaussian copula helpers.

The older CP simulation package, superseded comparator copies and historical development reports are omitted. Related candidate/dataset records are consolidated without dropping their numerical values or mixing diagnostics. Source/result integrity is checked with `python Scripts/Publication/validate_handoff.py`.

The original repository baseline is commit `424427fe1bf06d06494172cf278deb7bd4b29463`; its 46 files remain, with only additive ignore rules in `.gitignore`. No manuscript source or raw survey data is included. Full refits needing external data/checkpoints are identified in the guide; the package does not claim end-to-end reproduction of every result from aggregate files alone.

## Why the added files remain

The current package has 264 tracked files: the original 46 plus 218 additions.

| Added files | Count | Purpose |
| --- | ---: | --- |
| Fitting code, dependencies and truth inputs | 105 | Revised DDE, Poisson/Gaussian/iVAE comparisons, posterior sampler, and required helpers |
| Reproduction scripts, settings and instructions | 34 | Case runners, seed/configuration files, analysis, rendering and validation |
| Saved result records | 40 | Estimates, scores, timings, denominators, exclusions and mixing diagnostics |
| Reference figure files | 33 | Exact supplied figures, including artwork without a verified renderer |
| Root documentation, mappings and integrity files | 6 | Result/source mapping, external-input descriptions, notices and hashes |

Most fitting helpers are separate MATLAB functions. Combining them merely to reduce the count would require a numerical-code refactor and executable MATLAB validation. Duplicates that exactly match original helpers are reused; the DDE run setup reconstructs the same isolated source tree. Reference artwork is retained because regenerating some historical layouts is not yet verified. The single entry point avoids requiring readers to navigate those internal files.
