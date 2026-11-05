"""Build standalone sensitivity tables; final mode requires all 600 audited fits."""
from pathlib import Path
import argparse,json
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--preview',action='store_true');args=p.parse_args();o=args.folder
au=json.loads((o/'DEEP_REFITS_AUDIT.json').read_text());st=json.loads((o/'DEEP_REFIT_SUMMARY_STATUS.json').read_text())
assert not au['errors'] and au['checked_at']==st['source_checked_at']
d=pd.read_csv(o/'DEEP_REFITS_PAIRED_REPLICATES.csv').query("variant=='nofloor'")
ns=[500,1000,2000,4000,8000,16000]
f=pd.DataFrame(au.get('initialization_failures',[]),columns=['variant','n','replication','reason'])
assert not d.duplicated(['n','replication']).any()
complete=all((set(d.loc[d.n==n,'replication'])|set(f.loc[f.n==n,'replication']))==set(range(1,101)) for n in ns)
if not complete and not args.preview:raise SystemExit('Final no-floor tables withheld: all 600 audited outcomes required, including recorded initialization failures.')
m=pd.read_csv(o/'DEEP_REFIT_MSE_SUMMARY.csv').query("variant=='nofloor' and method=='refit' and population=='method_specific_correct'")
notice='PRELIMINARY: incomplete study. ' if args.preview else ''
rows=[r'\begin{table}[p]',r'\centering\small',r'\begin{tabular}{rrrrrr}',r'\toprule',r'$n$ & Attempts & $K_1=18$ & $K_2=6$ & $K_3=2$ & All \\',r'\midrule']
for n,g in d.groupby('n'):
 den=len(g)+int((f.n==n).sum())
 vals=[100*g[f'refit_K{k}'].eq(truth).sum()/den for k,truth in enumerate([18,6,2],1)]+[100*g.refit_correct.sum()/den]
 rows.append(f'{int(n)} & {den} & '+' & '.join(f'{v:.1f}' for v in vals)+r' \\')
rows += [r'\bottomrule',r'\end{tabular}',r'\caption{'+notice+r'No-floor sensitivity study: dimension recovery percentages for maxima with tempering and CAVI, using gap-plus-five initialization and $\tau_n=3n^{-0.3}$. Every resolved attempt contributes; recorded initialization failures count as unsuccessful recovery.}',r'\label{tab:nofloor_recovery}',r'\end{table}']
for layer in [1,2,3]:
 rows += [r'\begin{table}[p]',r'\centering\small',r'\begin{tabular}{rrrrr}',r'\toprule',r'$n$ & Untrimmed MSE & Trimmed MSE & Eligible & Excluded \\',r'\midrule']
 for x in m[m.layer==layer].sort_values('n').itertuples():
  fmt=lambda v:'---' if pd.isna(v) else f'{v:.6f}'
  rows.append(f'{int(x.n)} & {fmt(x.untrimmed_mean)} & {fmt(x.trimmed_mean)} & {int(x.eligible)} & {int(x.eligible-x.retained)} '+r'\\')
 rows += [r'\bottomrule',r'\end{tabular}',r'\caption{'+notice+rf'No-floor sensitivity study: layer {layer} entry-wise MSE conditional on correct recovery of all three dimensions. Upper outliers exceed $Q_3+1.5\,\mathrm{{IQR}}$ within sample size and layer. Missing means indicate no eligible fits.'+'}',rf'\label{{tab:nofloor_mse{layer}}}',r'\end{table}']
target=o/('NOFLOOR_TABLES_PREVIEW.tex' if args.preview else 'NOFLOOR_TABLES_FINAL.tex');target.write_text('\n'.join(rows)+'\n');print(target)
