"""Compare fresh B.4-B.5 fits with packaged historical references; no fitting."""
from pathlib import Path
import argparse
import json
import sys
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Algorithms/DDE'))
from dde_predict import DDEModel, states


def require_b45(group):
    if 'csp_update_version' not in group or ''.join(chr(int(x)) for x in group['csp_update_version'][()].ravel()) != 'B4_B5_v1':
        raise ValueError('Expected a B.4-B.5 fit, not a historical checkpoint.')


def scalar(group, name):
    return np.asarray(group[name]).ravel()[0].item()


def simulation_comparison(folder, output):
    reference = pd.read_csv(ROOT / 'Results/Publication/data/dde/CSP_AVERAGING_CHECK.csv')
    records = []
    for n in [500, 1000, 2000, 4000, 8000, 16000]:
        for rep in range(1, 6):
            file = folder / f'J100_n{n}_r{rep}' / 'result.mat'
            with h5py.File(file) as f:
                result = f['result']
                if not scalar(result, 'complete'):
                    raise ValueError(f'Incomplete fit: {file}')
                m, fit = result['metrics'], result['fit']
                require_b45(fit)
                row = dict(J=100, n=n, replication=rep,
                           b45_iterations=scalar(fit, 'iteration'),
                           b45_stabilized=bool(scalar(fit, 'stabilized')))
                for public, field in [('K1', 'K1'), ('K2', 'K2'),
                                      ('MSE1', 'B1_full'), ('MSE2', 'B2_full')]:
                    row['b45_' + public] = scalar(m, field)
                row['b45_CM_flags'] = sum(scalar(f[ref], 'flags')
                                          for ref in fit['trace'][()].ravel())
                records.append(row)
    paired = reference.merge(pd.DataFrame(records), on=['J', 'n', 'replication'], validate='one_to_one')
    if len(paired) != 30:
        raise ValueError('Expected all 30 paired cases; no filtering is allowed.')
    summaries = []
    for n, rows in paired.groupby('n'):
        row = dict(n=int(n), replications=len(rows),
                   b45_both_dimensions_correct=int(((rows.b45_K1 == 10) & (rows.b45_K2 == 3)).sum()),
                   b45_CM_flags=int(rows.b45_CM_flags.sum()))
        for prefix, label in [('old', 'original'), ('new', 'penalty_only')]:
            row[label + '_dimension_changes'] = int(((rows[prefix + '_K1'] != rows.b45_K1) |
                                                     (rows[prefix + '_K2'] != rows.b45_K2)).sum())
            for layer in [1, 2]:
                before, after = rows[f'{prefix}_MSE{layer}'], rows[f'b45_MSE{layer}']
                row[f'{label}_mean_MSE{layer}'] = float(before.mean())
                row[f'b45_mean_MSE{layer}'] = float(after.mean())
                row[f'{label}_mean_MSE{layer}_change_percent'] = float(100 * (after.mean() / before.mean() - 1))
                row[f'{label}_worst_individual_MSE{layer}_change_percent'] = float((100 * (after / before - 1)).max())
        summaries.append(row)
    paired.to_csv(output / 'SIMULATION_PAIRED_RESULTS.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'SIMULATION_SUMMARY.csv', index=False)
    return dict(completed_fits=30, summary_by_n=summaries,
                cap_without_stabilization=int((~paired.b45_stabilized).sum()),
                scope='Five full fits per setting; every case retained. MSE includes intercepts, original padding and alignment.')


def survey_comparison(folder, output):
    completed = json.loads((folder / 'COMPLETE.json').read_text())
    models = sorted((folder / 'fit').glob('MODEL_it*_target*.mat'))
    if not models or not completed['complete']:
        raise ValueError('A completed survey fit is required.')
    with h5py.File(models[-1]) as f:
        g = f['model']
        require_b45(g)
        widths = g['widths'][()].ravel().astype(int).tolist()
        prop = g['prop'][()].ravel() if widths[1] else np.zeros(0)
        model = DDEModel(g['B1'][()].T, g['B2'][()].T, g['gamma'][()].ravel(), prop)
    shallow = states(widths[0])
    logprior = model.lower_logprior(shallow)
    prior = np.exp(logprior - logsumexp(logprior))
    mean = prior @ shallow
    cov = (shallow * prior[:, None]).T @ shallow - np.outer(mean, mean)
    sd = np.sqrt(model.gamma + np.einsum('jk,kl,jl->j', model.B1[:, 1:], cov, model.B1[:, 1:]))
    # Match the normalization of the public original factor-comparison CSV.
    new = model.B1[:, 1:] * np.sqrt(mean * (1 - mean))[None, :] / sd[:, None]
    original = pd.read_csv(ROOT / 'Results/Publication/data/ivae/all_standardized_loadings.csv')
    old = original[[f'DDE_{k}' for k in range(1, 11)]].to_numpy()
    norms = np.outer(np.linalg.norm(old, axis=0), np.linalg.norm(new, axis=0))
    cos = np.divide(old.T @ new, norms, out=np.zeros_like(norms), where=norms > 0)
    left, right = linear_sum_assignment(-abs(cos))
    matches = [dict(original_factor=int(i + 1), b45_factor=int(j + 1), signed_cosine=float(cos[i, j]))
               for i, j in zip(left, right)]
    pd.DataFrame(matches).to_csv(output / 'SURVEY_LOADING_MATCHES.csv', index=False)
    pd.DataFrame(new, columns=[f'B45_{k+1}' for k in range(widths[0])]).to_csv(output / 'SURVEY_STANDARDIZED_LOADINGS.csv', index=False)
    new_scores = pd.read_csv(folder / 'PMSE_REPLICATES.csv')
    old_scores = pd.read_csv(ROOT / 'Results/Publication/data/empirical/PAIRED_PMSE.csv')
    paired = new_scores.merge(old_scores[['replicate', 'pMSE_dde']], on='replicate', validate='one_to_one')
    if len(paired) != 50:
        raise ValueError('Expected all 50 paired predictive replicates.')
    change = paired.pMSE - paired.pMSE_dde
    se = float(change.std(ddof=1) / np.sqrt(50))
    interval = change.mean() + np.array([-1, 1]) * t.ppf(.975, 49) * se
    paired['pMSE_change'] = change
    paired.to_csv(output / 'SURVEY_PAIRED_SCORES.csv', index=False)
    return dict(widths=widths, iterations=completed['iterations'], stabilized=completed['stabilized'],
                CM_flags=completed['CM_flags'], loading_matches=matches,
                unmatched_original_factors=sorted(set(range(1, 11)) - {int(i+1) for i in left}),
                unmatched_b45_factors=sorted(set(range(1, widths[0]+1)) - {int(i+1) for i in right}),
                original_pMSE_mean=float(paired.pMSE_dde.mean()), b45_pMSE_mean=float(paired.pMSE.mean()),
                pMSE_change=float(change.mean()), paired_MC_SE=se, paired_MC_interval=interval.tolist(),
                b45_AUC_mean=float(paired.AUC.mean()), b45_Brier_mean=float(paired.Brier.mean()),
                scope='Selected configuration only; no tuning-grid reranking or posterior-chain rerun. MC interval describes predictive-replicate variation, not population uncertainty.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulations', type=Path)
    parser.add_argument('--survey-fit', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.simulations and not args.survey_fit:
        parser.error('Supply --simulations, --survey-fit, or both.')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {}
    if args.simulations:
        report['simulation'] = simulation_comparison(args.simulations, args.output_dir)
    if args.survey_fit:
        report['survey'] = survey_comparison(args.survey_fit, args.output_dir)
    (args.output_dir / 'COMPARISON.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
