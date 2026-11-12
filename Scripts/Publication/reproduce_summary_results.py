"""Export paper table inputs, recompute J=100 timings, and plot saved deep MSEs."""
from pathlib import Path
import argparse, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import output
ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
out = output(a.output_dir)
data = ROOT / 'Results/Publication/data'
# Preserve all denominators and diagnostics in the table exports.
for source in ['dde/MAIN_DIMENSION_RECOVERY.csv',
               'deep/DEEP_REFIT_DIMENSION_RECOVERY.csv',
               'deep/DEEP_REFIT_MSE_SUMMARY.csv',
               'uncertainty/COVERAGE_SUMMARY.csv',
               'empirical/SURVEY_ITEM_RESPONSE_COUNTS.csv']:
    shutil.copy2(data/source, out/Path(source).name)
r = pd.read_csv(data/'native15_5/J100_MAP_RUNTIME_REPLICATES.csv')
assert len(r) == 600 and not r.duplicated(['n','replication']).any()
rows = []
for n, g in r.groupby('n', sort=True):
    assert set(g.replication) == set(range(1,101))
    t = g.numerical_wall
    assert np.allclose(t, g.warmup_wall + g.post_wall)
    rows.append(dict(n=n, replications=len(g), mean_seconds=t.mean(),
                     sd_seconds=t.std(ddof=1), median_seconds=t.median(),
                     mean_CPU_seconds=g.CPU.mean(),
                     mean_warmup_seconds=g.warmup_wall.mean(),
                     mean_post_seconds=g.post_wall.mean(),
                     mean_worker_attempt_seconds=g.actual_worker_attempt_wall.mean(),
                     mean_minutes=t.mean()/60))
summary = pd.DataFrame(rows)
reference = pd.read_csv(data/'native15_5/J100_MAP_RUNTIME_SUMMARY.csv')
assert np.allclose(summary[reference.columns], reference, rtol=1e-10, atol=1e-10)
summary.to_csv(out/'J100_MAP_RUNTIME_SUMMARY.csv', index=False)
s = pd.read_csv(data/'deep/DEEP_REFIT_MSE_SUMMARY.csv')
s = s.query("variant == 'nofloor' and method == 'refit' and population == 'method_specific_correct'")
for layer in (1,2,3):
    g = s[s.layer == layer].sort_values('n')
    assert len(g) == 6
    fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
    ax.plot(g.n, g.trimmed_mean, 'o-', label='Trimmed mean')
    ax.plot(g.n, g.untrimmed_mean, 's--', label='Untrimmed mean')
    ax.set(xscale='log', xlabel='Sample size (n)', ylabel='Entry-wise MSE',
           title=f'No-floor DDE: layer {layer}')
    ax.set_xticks(g.n, g.n.astype(str)); ax.grid(alpha=.2); ax.legend()
    fig.savefig(out/f'B{layer}_MSE.pdf'); plt.close(fig)
# Known generating matrices are model specifications, not fitted results.
truth=np.load(data/'dde/TRUTH_MATRICES_15_5.npz')
for J in [50,100,150]:
    for layer in [1,2]:
        matrix=truth[f'J{J}_B{layer}'];bound=np.max(np.abs(matrix))
        fig,ax=plt.subplots(figsize=(5.7,4),layout='constrained')
        im=ax.imshow(matrix,aspect='auto',cmap='viridis',vmin=-bound,vmax=bound,interpolation='nearest')
        ax.set(title=f'J={J}: B{layer} truth',xlabel='Factor (0 = intercept)',ylabel='Item' if layer==1 else 'First-layer feature')
        fig.colorbar(im,ax=ax)
        fig.savefig(out/f'J{J}_B{layer}_truth.pdf');plt.close(fig)
(out/'README.txt').write_text(
    'Table CSVs are exported from saved aggregates, not recomputed from fits.\n'
    'J=100 timing statistics are recomputed from all 100 replications per n.\n'
    'Times are numerical warmup + post-warmup wall seconds, excluding overhead.\n'
    'Deep plots re-render saved numerical summaries; historical typography may differ.\n'
    'Six truth heatmaps are regenerated from the exact generating matrices.\n'
    'Deep MSE conditions on all dimensions correct; upper Q3+1.5 IQR per n/layer.\n'
    'Uncertainty enclosure is not calibrated coverage: all 40 datasets failed mixing checks.\n')
print(out)
