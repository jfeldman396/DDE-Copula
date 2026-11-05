"""Compare refits against matched baseline IDs, with explicit eligibility populations."""
from pathlib import Path
import json,argparse
import numpy as np,pandas as pd
parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,required=True);out=parser.parse_args().folder
a=json.loads((out/'DEEP_REFITS_AUDIT.json').read_text());assert not a['errors']
d=pd.read_csv(out/'DEEP_REFITS_PAIRED_REPLICATES.csv');rows=[];exclusions=[]
for (variant,n),g in d.groupby(['variant','n']):
 for population in ['method_specific_correct','both_correct']:
  for method in ['refit','baseline']:
   eligible=g[g[f'{method}_correct']] if population=='method_specific_correct' else g[g.refit_correct & g.baseline_correct]
   for layer in [1,2,3]:
    v=eligible[f'{method}_MSE{layer}'];q1,q3=v.quantile([.25,.75]) if len(v) else (np.nan,np.nan);cut=q3+1.5*(q3-q1);keep=v<=cut
    rows.append(dict(variant=variant,n=n,population=population,method=method,layer=layer,completed_refits=len(g),available_baseline_pairs=int(g.baseline_available.sum()),eligible=len(v),retained=int(keep.sum()),untrimmed_mean=v.mean(),trimmed_mean=v[keep].mean(),upper_cutoff=cut))
    for idx,x in v.items():exclusions.append(dict(variant=variant,n=n,population=population,method=method,layer=layer,replication=int(g.loc[idx,'replication']),MSE=x,excluded=bool(x>cut)))
pd.DataFrame(rows).to_csv(out/'DEEP_REFIT_MSE_SUMMARY.csv',index=False);pd.DataFrame(exclusions).to_csv(out/'DEEP_REFIT_MSE_EXCLUSIONS.csv',index=False)
(out/'DEEP_REFIT_SUMMARY_STATUS.json').write_text(json.dumps(dict(source_checked_at=a['checked_at'],studies=a['studies'],notice='Interim until the corresponding study is complete. Method-specific eligibility matches manuscript rule; both-correct is a secondary matched-case diagnostic. Trimming is separate by variant/n/method/layer/population; untrimmed means retained.'),indent=2))
print('Wrote refit summaries and exclusion audit.')

# Signal-only errors use their own exclusion thresholds, not full-MSE exclusions.
signal_rows=[];signal_exclusions=[];graph_rows=[]
for (variant,n),g in d.groupby(['variant','n']):
 for method in ['refit','baseline']:
  for layer in [1,2,3]:
   graph_rows.append(dict(variant=variant,n=n,method=method,layer=layer,completed_refits=len(g),available_baseline_pairs=int(g.baseline_available.sum()),evaluated=int(g[f'{method}_exact_graph{layer}'].notna().sum()),exact_graph_count=int(g[f'{method}_exact_graph{layer}'].sum()),mean_missing_signals=g[f'{method}_missing_signals{layer}'].mean(),mean_accuracy=g[f'{method}_graph_accuracy{layer}'].mean()))
 for population in ['method_specific_correct','both_correct']:
  for method in ['refit','baseline']:
   eligible=g[g[f'{method}_correct']] if population=='method_specific_correct' else g[g.refit_correct & g.baseline_correct]
   for layer in [1,2,3]:
    v=eligible[f'{method}_signal_MSE{layer}'];q1,q3=v.quantile([.25,.75]) if len(v) else (np.nan,np.nan);cut=q3+1.5*(q3-q1);keep=v<=cut
    signal_rows.append(dict(variant=variant,n=n,population=population,method=method,layer=layer,eligible=len(v),retained=int(keep.sum()),untrimmed_mean=v.mean(),trimmed_mean=v[keep].mean(),upper_cutoff=cut))
    for idx,x in v.items():signal_exclusions.append(dict(variant=variant,n=n,population=population,method=method,layer=layer,replication=int(g.loc[idx,'replication']),signal_MSE=x,excluded=bool(x>cut)))
pd.DataFrame(signal_rows).to_csv(out/'DEEP_REFIT_SIGNAL_MSE_SUMMARY.csv',index=False)
pd.DataFrame(signal_exclusions).to_csv(out/'DEEP_REFIT_SIGNAL_MSE_EXCLUSIONS.csv',index=False)
pd.DataFrame(graph_rows).to_csv(out/'DEEP_REFIT_GRAPH_SUMMARY.csv',index=False)

# Recovery denominators include resolved initialization failures; MSE does not.
failure_rows=a.get('initialization_failures',[])
recovery=[]
for (variant,n),g in d.groupby(['variant','n']):
 failed=sum(x['variant']==variant and x['n']==n for x in failure_rows)
 denominator=len(g)+failed
 row=dict(variant=variant,n=int(n),successful_fits=len(g),initialization_failed=failed,attempts_resolved=denominator)
 for layer,truth in enumerate([18,6,2],1):
  correct=int(g[f'refit_K{layer}'].eq(truth).sum())
  row[f'K{layer}_correct']=correct;row[f'K{layer}_correct_percent']=100*correct/denominator
 row['joint_correct']=int(g.refit_correct.sum());row['joint_correct_percent']=100*row['joint_correct']/denominator
 recovery.append(row)
pd.DataFrame(recovery).to_csv(out/'DEEP_REFIT_DIMENSION_RECOVERY.csv',index=False)
