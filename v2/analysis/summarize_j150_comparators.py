"""Prepare J150 comparator summaries; only complete audited methods are published."""
from pathlib import Path
import json,argparse
import pandas as pd
parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path);args=parser.parse_args()
OUT=args.folder or Path(__file__).resolve().parents[2]/'output/manuscript_update_20261103'
audit=json.loads((OUT/'CURRENT_RUN_AUDIT.json').read_text())
assert not audit['errors'],'Endpoint audit must pass'
data=pd.read_csv(OUT/'J150_CURRENT_COMPARATORS.csv');summaries=[];exclusions=[];status={}
for method,arm in [('Original RL-SAEM',5),('Poisson',4)]:
 complete=audit['studies'][method]['complete'];status[method]={'complete':complete,'included_in_final_summary':complete}
 if not complete:continue
 d=data[data.method==method];assert len(d)==600
 for n,g in d.groupby('n'):
  assert len(g)==100 and set(g.replication)==set(range(1,101))
  for layer in [1,2]:
   x=g[f'B{layer}_MSE'];q1,q3=x.quantile([.25,.75]);cut=q3+1.5*(q3-q1);keep=x<=cut
   summaries.append(dict(J=150,n=n,arm=arm,layer=layer,eligible=len(x),retained=int(keep.sum()),excluded=int((~keep).sum()),upper_cutoff=cut,trimmed_MSE=x[keep].mean(),untrimmed_MSE=x.mean(),q1=q1,q3=q3))
   for idx,v in x.items():exclusions.append(dict(method=method,J=150,n=n,layer=layer,replication=int(g.loc[idx,'replication']),MSE=v,upper_cutoff=cut,excluded=bool(v>cut)))
if summaries:
 pd.DataFrame(summaries).to_csv(OUT/'J150_COMPLETED_COMPARATOR_MSE.csv',index=False)
 pd.DataFrame(exclusions).to_csv(OUT/'J150_COMPLETED_COMPARATOR_EXCLUSIONS.csv',index=False)
(OUT/'J150_COMPARATOR_SUMMARY_STATUS.json').write_text(json.dumps({'checked_at':audit['checked_at'],'methods':status,'rule':'All replications eligible; upper Q3+1.5 IQR separately by method/n/layer; incomplete methods withheld'},indent=2)+'\n')
print(json.dumps(status))
