"""Plot current deep MSE; incomplete results require explicit preview mode."""
from pathlib import Path
import argparse,json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');p.add_argument('--folder',type=Path);args=p.parse_args()
out=args.folder or Path(__file__).resolve().parents[3]/'output/manuscript_update_20261103';a=json.loads((out/'CURRENT_RUN_AUDIT.json').read_text());s=json.loads((out/'DEEP_SUMMARY_STATUS.json').read_text());assert not a['errors'] and s['source_checked_at']==a['checked_at']
if not a['studies']['deep']['complete'] and not args.preview:raise SystemExit('Final deep plots withheld: incomplete study')
folder=out/('deep_preview_figures' if args.preview else 'deep_final_figures');folder.mkdir(exist_ok=True);d=pd.read_csv(out/'DEEP_CURRENT_MSE_SUMMARY.csv')
plt.rcParams.update({'font.family':'Arial','font.size':17,'axes.titlesize':20,'axes.titleweight':'bold','axes.labelsize':20,'pdf.fonttype':42})
for layer in [1,2,3]:
 fig,ax=plt.subplots(figsize=(12,9));fig.subplots_adjust(left=.12,right=.97,bottom=.20,top=.87);notes=[]
 for arm,c,marker,label in [(1,'#0072BD','o','Maxima: tempering + CAVI'),(2,'#D95319','s','Maxima: CAVI only'),(3,'#EDB120','D','Maxima: tempering only')]:
  g=d[(d.arm==arm)&(d.layer==layer)].sort_values('n');ax.plot(g.n,g.trimmed_mean,color=c,marker=marker,mfc='none',ms=10,mew=2,lw=2,label=label);notes.append(label+': '+', '.join(f'{int(x.retained)}/{int(x.eligible)}' for x in g.itertuples()))
 ax.set_xscale('log');ns=[500,1000,2000,4000,8000,16000];ax.set_xticks(ns,[str(n) for n in ns]);ax.set_xlabel('Sample Size (n)');ax.set_ylabel('Trimmed Average Entry-wise MSE');ax.set_title(rf'$J=108$: $B^{{({layer})}}$ Entry-wise MSE');ax.grid(alpha=.22);ax.tick_params(direction='in',top=True,right=True);ax.legend(fontsize=12)
 if args.preview:fig.text(.5,.96,'PRELIMINARY: incomplete study',ha='center',color='#A00000',fontsize=17)
 fig.text(.12,.025,'All three dimensions correct; upper Q3 + 1.5 IQR exclusion per method / n / layer.\nEmpty eligible cells are missing, not zero.',fontsize=10)
 for ext in ['png','pdf']:fig.savefig(folder/f'D3_B{layer}_MSE.{ext}',dpi=180)
 plt.close(fig)
print(folder)
