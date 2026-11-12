from pathlib import Path
import numpy as np,pandas as pd,h5py
import argparse,json
from scipy.io import loadmat,savemat
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--folder',type=Path,required=True);p.add_argument('--preview',action='store_true');args=p.parse_args();root=args.root;out=args.folder
a=json.loads((out/'DEEP_REFITS_AUDIT.json').read_text());status=json.loads((out/'DEEP_REFIT_SUMMARY_STATUS.json').read_text());assert not a['errors'] and status['source_checked_at']==a['checked_at']
if not a['studies']['nofloor']['complete'] and not args.preview:raise SystemExit('Final heatmap matrices withheld: no-floor study incomplete.')
dest=out/('nofloor_weight_preview' if args.preview else 'nofloor_weight_final');dest.mkdir(exist_ok=True)
e=pd.read_csv(out/'DEEP_REFIT_MSE_EXCLUSIONS.csv');summ=pd.read_csv(out/'DEEP_REFIT_MSE_SUMMARY.csv');data={};rows=[]
e=e.query("variant=='nofloor' and method=='refit' and population=='method_specific_correct'");summ=summ.query("variant=='nofloor' and method=='refit' and population=='method_specific_correct'")
with h5py.File(root/'results/deeper_paper_rerun/truth.mat') as f: truths=[f[r][()].T for r in f['record/B_padded'][()].ravel()]
for layer in (1,2,3):
 for n in (500,1000,2000,4000,8000,16000):
  g=e[(e.layer==layer)&(e.n==n)&(~e.excluded)];bs=[];ts=[];ms=[]
  for r in g.replication:
   m=loadmat(root/f'results/manuscript_deep_nofloor_20261104/full_n{n:05}_r{r:03}_a1_result.mat',simplify_cells=True)['metrics'];b=m['B_aligned'][layer-1];t=truths[layer-1].copy()
   if layer==1:
    with h5py.File(root/f'results/manuscript_deep_pyramid_20261103/inputs/input_n{n:05}_r{r:03}.mat') as f:
     mu=f['input/truth_marginal_mean'][()].ravel();sd=f['input/truth_marginal_sd'][()].ravel()
    t[:,0]=(t[:,0]-mu)/sd;t[:,1:]/=sd[:,None]
   assert np.isclose(np.mean((b-t)**2),m['MSE'][layer-1],rtol=1e-8)
   bs.append(b);ts.append(t);ms.append(m['MSE'][layer-1])
  sr=summ[(summ.layer==layer)&(summ.n==n)].iloc[0];assert len(bs)==sr.retained and np.isclose(np.mean(ms),sr.trimmed_mean)
  mean=np.mean(bs,axis=0);truth=np.mean(ts,axis=0);data[f'B{layer}_n{n}_mean']=mean;data[f'B{layer}_n{n}_truth']=truth;data[f'B{layer}_n{n}_error']=mean-truth;data[f'B{layer}_n{n}_count']=np.array([len(bs),sr.eligible]);rows.append(dict(layer=layer,n=n,retained=len(bs),eligible=sr.eligible,mean_replication_MSE=np.mean(ms),MSE_of_mean=np.mean((mean-truth)**2)))
savemat(dest/'matrices.mat',data);pd.DataFrame(rows).to_csv(dest/'counts_and_errors.csv',index=False)

(dest/'STATUS.json').write_text(json.dumps({'source_checked_at':a['checked_at'],'preliminary':not a['studies']['nofloor']['complete'],'note':'Mean weight matrices are descriptive; MSE of a mean is not mean replication MSE. Matrices use layer-specific retained sets.'},indent=2)+'\n')
print(dest)
