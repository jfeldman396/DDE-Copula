Original truncated-lasso RL-SAEM comparator, not CSP rank-Gibbs.
Production J150 rerun uses fixed50/16 widths, Normal_init_v2 with seed data_seed+700000, algorithm_seed for fitting, temperature.7, cap50, tau=max(.3,3*n^(-.3)), penalties n^.25.
The parfor implementation was run with parallel pool automatic creation disabled; a portable wrapper must preserve this execution policy and restore caller settings.
Source kernels are byte-preserved. Portable run_original_rl_saved_case passed exact prop,B1,B2,gamma,Z and iteration equality for completed J150 n500rep1. See provenance/ORIGINAL_RL_PORTABLE_VALIDATION.json.

Usage: run_original_rl_saved_case(absoluteInputFile,absoluteNewOutputDir). The input must contain the archived synthetic record. Run in a fresh MATLAB session without an active parallel pool. The wrapper refuses existing outputs, verifies source hashes, checks paired-data ranks, restores path/cwd/RNG and pool auto-create preference. Output contains unaligned coefficients and latent state; aggregate scoring remains separate.
