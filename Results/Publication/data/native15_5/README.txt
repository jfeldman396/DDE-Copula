PAPER FIGURES: NATIVE 15/5 SIMULATION COMPARISONS

12 figures are retained as vector PDFs in Results/Publication/figures/native15_5. The renderer regenerates PDFs and PNG previews in its new output directory.

For each J = 50, 100, 150:
  J*_B1_main_MSE.pdf and J*_B2_main_MSE.pdf: entry-wise coefficient MSE versus sample size.
  J*_B1_est_updated.pdf and J*_B2_est_updated.pdf: mean coefficient matrices at n = 1000, 4000, 16000, alongside truth.

MSE CAPTION (replace J and layer as appropriate)
Entry-wise mean squared error for B^(1) or B^(2) at J = [J], plotted on logarithmic axes. Each method/setting starts with 100 replications. MAP: tempering + VI is restricted to replications recovering both true dimensions (10,3); truncated-lasso RL and Poisson DDE include all valid replications before trimming. The upper Q3 + 1.5 IQR exclusion rule is applied separately by method, J, sample size, and layer. Errors include intercepts on common J-by-16 and 15-by-6 coefficient grids. Poisson log-mean coefficients and Gaussian latent-scale coefficients are different estimands; this comparison follows the paper's descriptive coefficient-error convention. Eligibility and exclusion counts accompany the numerical data.

HEATMAP CAPTION (replace J and layer as appropriate)
Average estimates of B^(1) or B^(2) at J = [J] for n = 1000, 4000, and 16000, compared with data-generating coefficients. Rows show MAP: tempering + VI, truncated-lasso RL, and Poisson DDE. All fits initialize with widths (15,5); column zero contains the intercept. Each estimated panel averages the exact layer-specific retained replications used in the corresponding MSE curve. Each figure shares a symmetric color scale set by the maximum absolute truth coefficient; values beyond its limits saturate, as indicated by the colorbar extensions. Exact values and saturation counts are supplied. Poisson coefficients retain their distinct log-mean scale.

VALIDATION AND SCOPE
2,997 new comparator endpoints passed independent source, seed, dimensions, finite-value, alignment, and MSE checks. Collection uses 603 predeclared comparator references and 1,800 MAP fits: 5,400 unique method/setting/replication records, summarized in 108 method/J/n/layer groups. Three extra saved files for reuse-designated cases are preserved but not counted or used; the predeclared references remain selected. A completed finite-budget fit is not evidence of convergence; 118 newly audited Poisson fits reached the original 50-iteration cap.

SUPPORTING DATA
MSE_SUMMARY.csv: retained and untrimmed MSE means and sample counts.
REPLICATE_MSE_UNTRIMMED_ARCHIVE.csv: all per-replication errors and eligibility/retention flags.
AVERAGE_WEIGHT_MATRICES.npz and weight_matrices/*.csv: retained and all-eligible means, including true matrices in the NPZ.
ELIGIBILITY_AND_EXCLUSIONS.json: exact replication selections.
FIGURE_SOURCES.json and FIGURE_RENDER_AUDIT.json: plotted-data mapping and saturation counts.

Existing manuscripts and older figure packages were preserved. These are the completed native-15/5 simulation figures, not a revised manuscript or a new uncertainty study.

Re-render from the repository root:
python Scripts/Publication/render_native15_5.py --output /tmp/dde_native15_5_figures
