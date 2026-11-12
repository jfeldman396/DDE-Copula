"""Build deep-study LaTeX tables, withholding final output until audit completion."""
from pathlib import Path
import argparse,json
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');p.add_argument('--folder',type=Path);args=p.parse_args()
out=args.folder or Path(__file__).resolve().parents[3]/'output/manuscript_update_20261103'
a=json.loads((out/'CURRENT_RUN_AUDIT.json').read_text());status=json.loads((out/'DEEP_SUMMARY_STATUS.json').read_text())
assert not a['errors'];assert status['source_checked_at']==a['checked_at'],'Refresh deep summaries first'
if not a['studies']['deep']['complete'] and not args.preview:raise SystemExit('Final deep tables withheld: study incomplete. Use --preview only for layout review.')
r=pd.read_csv(out/'DEEP_CURRENT_DIMENSION_RECOVERY.csv');m=pd.read_csv(out/'DEEP_CURRENT_MSE_SUMMARY.csv');names={1:'Tempering + CAVI',2:'CAVI only',3:'Tempering only'}
lines=[r'\begin{table}[p]',r'\centering\small',r'\begin{tabular}{rlrrrrr}',r'\toprule',r'$n$ & Maxima variant & Fits & $K_1=18$ & $K_2=6$ & $K_3=2$ & All \\',r'\midrule']
for x in r.itertuples():
 lines.append(f'{x.n} & {names[x.arm]} & {x.completed} & {x.K1_correct_percent:.1f} & {x.K2_correct_percent:.1f} & {x.K3_correct_percent:.1f} & {x.joint_correct_percent:.1f} '+r'\\')
notice='PRELIMINARY LAYOUT PREVIEW: incomplete study. ' if args.preview else ''
lines += [r'\bottomrule',r'\end{tabular}',r'\caption{'+notice+r'Three-layer dimension recovery percentages at $J=108$, by sample size and algorithm. All completed replications contribute; no MSE-based exclusions are applied.}',r'\label{tab:deep_updated_recovery}',r'\end{table}']
for layer in [1,2,3]:
 lines += [r'\begin{table}[p]',r'\centering\small',r'\begin{tabular}{rlrrr}',r'\toprule',r'$n$ & Maxima variant & Untrimmed MSE & Trimmed MSE & Kept/eligible \\',r'\midrule']
 for x in m[m.layer==layer].itertuples():
  fmt=lambda v:'---' if pd.isna(v) else f'{v:.5f}'
  lines.append(f'{x.n} & {names[x.arm]} & {fmt(x.untrimmed_mean)} & {fmt(x.trimmed_mean)} & {x.retained}/{x.eligible} '+r'\\')
 lines += [r'\bottomrule',r'\end{tabular}',r'\caption{'+notice+rf'Three-layer study: layer {layer} MSE, conditional on recovery of all three true dimensions. Trimming removes values above $Q_3+1.5\,\mathrm{{IQR}}$ separately within each sample size, algorithm and layer. Empty eligible cells are unavailable, not zero.'+'}',rf'\label{{tab:deep_updated_mse{layer}}}',r'\end{table}']
name='DEEP_TABLES_PREVIEW.tex' if args.preview else 'DEEP_TABLES_FINAL.tex'
(out/name).write_text('\n'.join(lines)+'\n');print(name)
