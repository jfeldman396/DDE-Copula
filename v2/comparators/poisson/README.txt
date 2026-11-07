Original Poisson comparator kernels, copied without modification from upstream commit45f7360c0017bb8b24ceb72f745f7845320dac7e.
Portable saved-input wrapper run_poisson_saved_case now passes exact prop/B1/B2/iteration parity for a full50-iteration J150,n500,rep1 case. See provenance/POISSON_PORTABLE_VALIDATION.json.
The comparator uses fixed initialized widths and truncated-lasso regularization, not CSP rank-Gibbs.
Poisson log-mean coefficients and Gaussian latent-scale coefficients are different estimands; coefficient MSE follows the original paper convention.
Do not add this tree recursively beside other estimator implementations: function names overlap. The forthcoming wrapper must isolate path resolution and restore the caller path.

Usage: add v2 to the MATLAB path, then run_poisson_saved_case(absoluteInputFile,absoluteNewOutputDir). Input MAT must contain the archived synthetic record with data/algorithm seeds and initialization. Existing output destinations are refused. The wrapper verifies source hashes, regenerates paired count data and checks ranks, isolates estimator paths, and restores the caller path, working directory and RNG. Output is an unaligned coefficient endpoint; aggregate scoring/eligibility remain separate.
