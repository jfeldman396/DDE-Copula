"""Keep separately authorized refits paired with baseline, never pooled."""
from pathlib import Path
from datetime import datetime,timezone
import json,re,h5py,argparse
import numpy as np,pandas as pd
from scipy.io import loadmat
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.root;out=args.output;out.mkdir(parents=True,exist_ok=True);rows=[];errors=[];counts={};failures=[]
with h5py.File(root/'results/deeper_paper_rerun/truth.mat') as f:truth=[f[x][()].T for x in f['record/B_padded'][()].ravel()]
for variant in ['nofloor']:
 seen=set()
 for p in sorted((root/f'results/manuscript_deep_{variant}_20261104').glob('full*_result.mat')):
  try:
   match=re.fullmatch(r'full_n(\d+)_r(\d+)_a1_result.mat',p.name);assert match
   n,r=map(int,match.groups());assert n in ([500,1000,2000,4000,8000,16000] if variant=='nofloor' else [4000,8000,16000]) and 1<=r<=100 and (n,r) not in seen
   q=loadmat(p,simplify_cells=True);m=q['metrics'];hp=q['hp'];assert hp['cavi'] and np.isclose(hp['beta0'],.9)
   assert np.allclose(hp['tau'],[.3,.4,.4] if variant=='tau' else (3*n**(-.3) if variant=='nofloor' else max(.3,3*n**(-.3))))
   assert np.allclose(hp['spikes'],[.01,.03,.005]) and np.allclose(hp['alpha'],[36,12,4])
   widths=np.asarray(m['initial_widths']);assert len(widths)==3
   if variant=='tau':assert np.array_equal(widths,[27,9,3])
   else:assert widths[1]==widths[0]//3 and widths[2]==widths[1]//3 and 1<=widths[2] and widths[0]<=36
   baseline_folder='manuscript_deep_gap5_20261104' if variant=='nofloor' and n>=4000 else 'manuscript_deep_pyramid_20261103'
   bp=root/'results'/baseline_folder/p.name
   b=loadmat(bp,simplify_cells=True)['metrics'] if bp.exists() else None
   if variant=='nofloor' and n>=4000 and b is not None:assert np.array_equal(widths,b['initial_widths'])
   with h5py.File(root/f'results/manuscript_deep_pyramid_20261103/inputs/input_n{n:05}_r{r:03}.mat') as f:sd=f['input/truth_marginal_sd'][()].ravel()
   row=dict(variant=variant,n=n,replication=r,baseline_study=baseline_folder,baseline_available=b is not None)
   for label,item in [('refit',m),('baseline',b)]:
    if item is None:
     row[label+'_correct']=False
     for d in [1,2,3]:
      for metric in ['MSE','K','signal_MSE','missing_signals','graph_accuracy','exact_graph']:row[f'{label}_{metric}{d}']=np.nan
     continue
    k=np.asarray(item['K']);v=np.asarray(item['MSE']);assert np.isfinite(v).all() and (v>=0).all();correct=bool(np.all(k==[18,6,2]));assert bool(item['correct_all'])==correct;row[label+'_correct']=correct
    for d,shape in enumerate([(108,37),(36,13),(12,5)],1):
     matrix=np.asarray(item['B_aligned'][d-1]);assert matrix.shape==shape and np.isfinite(matrix).all();row[f'{label}_MSE{d}']=v[d-1];row[f'{label}_K{d}']=k[d-1]
     target=truth[d-1][:,1:].copy();mask=target!=0;estimate=matrix[:,1:]
     if d==1:target/=sd[:,None]
     row[f'{label}_signal_MSE{d}']=float(np.mean((estimate[mask]-target[mask])**2));row[f'{label}_missing_signals{d}']=int(np.sum(mask&(estimate==0)));row[f'{label}_graph_accuracy{d}']=float(np.mean(mask==(estimate!=0)));row[f'{label}_exact_graph{d}']=bool(np.array_equal(mask,estimate!=0))
   rows.append(row);seen.add((n,r))
  except Exception as e:errors.append(dict(file=str(p),error=str(e)))
 failed=set()
 if variant=='nofloor':
  for p in sorted((root/'results/manuscript_deep_nofloor_20261104').glob('full*_failure.mat')):
   try:
    match=re.fullmatch(r'full_n(\d+)_r(\d+)_a1_failure.mat',p.name);assert match
    n,r=map(int,match.groups());f=loadmat(p,simplify_cells=True)['failure']
    assert n in [500,1000,2000,4000,8000,16000] and 1<=r<=100 and (n,r) not in seen and (n,r) not in failed
    assert int(f['n'])==n and int(f['replication'])==r and f['status']=='initialization_failed'
    assert 'zero top-layer width' in f['reason']
    failed.add((n,r));failures.append(dict(variant=variant,n=n,replication=r,reason=f['reason']))
   except Exception as e:errors.append(dict(file=str(p),error=str(e)))
 target=600 if variant=='nofloor' else 300
 counts[variant]=dict(completed=len(seen),initialization_failed=len(failed),attempts_resolved=len(seen|failed),expected=target,complete=len(seen|failed)==target)
df=pd.DataFrame(rows);df.to_csv(out/'DEEP_REFITS_PAIRED_REPLICATES.csv',index=False)
summary=[]
if len(df):
 for (variant,n),g in df.groupby(['variant','n']):
  summary.append(dict(variant=variant,n=int(n),completed=len(g),available_baseline_pairs=int(g.baseline_available.sum()),refit_correct=int(g.refit_correct.sum()),baseline_correct=int(g.baseline_correct.sum()),both_correct=int((g.refit_correct & g.baseline_correct).sum())))
for item in summary:
 item['initialization_failed']=sum(f['variant']==item['variant'] and f['n']==item['n'] for f in failures)
 item['attempts_resolved']=item['completed']+item['initialization_failed']
pd.DataFrame(failures,columns=['variant','n','replication','reason']).to_csv(out/'DEEP_REFIT_INITIALIZATION_FAILURES.csv',index=False)
a=dict(initialization_failures=failures,checked_at=datetime.now(timezone.utc).isoformat(),studies=counts,errors=errors,recovery=summary,notice='Partial results until the variant reaches its expected endpoint count; comparisons use matched replication IDs. Correct-dimension MSE populations may differ.')
(out/'DEEP_REFITS_AUDIT.json').write_text(json.dumps(a,indent=2));print(json.dumps(a,indent=2))
