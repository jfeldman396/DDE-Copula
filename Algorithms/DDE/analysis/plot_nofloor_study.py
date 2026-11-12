"""All-n no-floor results, with mandatory preliminary marking until complete."""
from pathlib import Path
import argparse,json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');p.add_argument('--folder',type=Path,required=True);args=p.parse_args()
o=args.folder;a=json.loads((o/'DEEP_REFITS_AUDIT.json').read_text());s=json.loads((o/'DEEP_REFIT_SUMMARY_STATUS.json').read_text());assert not a['errors'] and s['source_checked_at']==a['checked_at']
if not a['studies']['nofloor']['complete'] and not args.preview:raise SystemExit('Final no-floor plots withheld until600 audited outcomes (fits or recorded initialization failures).')
d=pd.read_csv(o/'DEEP_REFIT_MSE_SUMMARY.csv');d=d[(d.variant=='nofloor')&(d.method=='refit')&(d.population=='method_specific_correct')];folder=o/('nofloor_preview_figures' if args.preview else 'nofloor_final_figures');folder.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':15,'pdf.fonttype':42})
for layer in [1,2,3]:
 g=d[d.layer==layer].sort_values('n');fig,ax=plt.subplots(figsize=(10,7));fig.subplots_adjust(bottom=.20,top=.86)
 ax.plot(g.n,g.trimmed_mean,'o-',color='#0072BD',mfc='none',lw=2,label='Trimmed mean');ax.plot(g.n,g.untrimmed_mean,'s--',color='#D95319',mfc='none',lw=2,label='Untrimmed mean');ax.set_xscale('log');ns=[500,1000,2000,4000,8000,16000];ax.set_xticks(ns,[str(x) for x in ns]);ax.set_xlabel('Sample size (n)');ax.set_ylabel('Entry-wise MSE');ax.set_title(f'Layer {layer}: gap+5, maxima + tempering + CAVI');ax.grid(alpha=.2);ax.legend();
 if args.preview:fig.text(.5,.95,'PRELIMINARY: incomplete study',ha='center',color='#A00000')
 fig.text(.08,.04,'All three dimensions correct; threshold 3 n^(-0.3), no floor.\nUpper Q3 + 1.5 IQR applied separately by n and layer.',fontsize=9,wrap=True)
 for ext in ['png','pdf']:fig.savefig(folder/f'B{layer}_MSE.{ext}',dpi=160)
 plt.close(fig)
print(folder)
