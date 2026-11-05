"""Validate current-run endpoints; never use superseded fits or partial averages as final."""
from pathlib import Path
import argparse,hashlib,json,re
from datetime import datetime,timezone
import h5py,numpy as np,pandas as pd
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[2]
NS=(500,1000,2000,4000,8000,16000)

def collect(root, output):
    output.mkdir(parents=True,exist_ok=True)
    errors=[];sources=[];deep=[];comparators=[]
    base=root/'results/manuscript_deep_pyramid_20261103'
    expected={(n,r,a) for n in NS for r in range(1,101) for a in (1,2,3)};seen=set()
    for p in sorted(base.glob('full*_result.mat')):
        try:
            match=re.fullmatch(r'full_n(\d+)_r(\d+)_a([123])_result.mat',p.name)
            assert match, 'unexpected filename'
            n,rep,arm=map(int,match.groups());key=(n,rep,arm)
            assert key in expected and key not in seen,'unexpected or duplicate endpoint'
            q=loadmat(p,simplify_cells=True);m=q['metrics'];hp=q['hp']
            widths=np.asarray(m['initial_widths'],dtype=int).ravel();K=np.asarray(m['K']).ravel()
            assert len(widths)==3 and widths[0]>=3*widths[1] and widths[1]>=3*widths[2] and widths[2]>=3
            assert np.all(widths<=np.array([36,12,4])) and np.all(K<=widths)
            assert bool(hp['cavi'])==(arm!=3)
            assert np.isclose(hp['beta0'],1 if arm==2 else .9)
            mse=np.asarray(m['MSE']).ravel();assert len(mse)==3 and np.isfinite(mse).all() and (mse>=0).all()
            correct=K==np.array([18,6,2]);assert bool(m['correct_all'])==bool(correct.all())
            for d,shape in enumerate([(108,37),(36,13),(12,5)]):
                matrix=np.asarray(m['B_aligned'][d]);assert matrix.shape==shape and np.isfinite(matrix).all()
            row=dict(n=n,replication=rep,arm=arm,correct_all=bool(correct.all()))
            for d in range(3):row.update({f'K{d+1}':K[d],f'initial_K{d+1}':widths[d],f'MSE{d+1}':mse[d],f'correct{d+1}':bool(correct[d])})
            deep.append(row);seen.add(key);sources.append(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        except Exception as exc:errors.append(dict(path=str(p),error=str(exc)))
    counts={'deep':dict(valid=len(seen),expected=len(expected),complete=seen==expected)}
    for method,folder in [('Poisson','manuscript_poisson_20261103/J150_rerun'),('Original RL-SAEM','manuscript_original_rl_20261103/full')]:
        expected2={(n,r) for n in NS for r in range(1,101)};seen2=set()
        for p in sorted((root/'results'/folder).glob('fit*.mat')):
            try:
                with h5py.File(p,'r') as f:
                    r=f['record'];scalar=lambda obj:float(np.asarray(obj).squeeze())
                    J=int(scalar(r['J']));n=int(scalar(r['n']));rep=int(scalar(r['replication']));m=r['metrics']
                    assert J==150 and (n,rep) in expected2 and (n,rep) not in seen2
                    vals=[scalar(m['B1_full']),scalar(m['B2_full'])];assert np.isfinite(vals).all() and min(vals)>=0
                    for d,shape in [(1,(150,51)),(2,(50,17))]:
                        x=np.asarray(m[f'B{d}_aligned']).T;assert x.shape==shape and np.isfinite(x).all()
                    comparators.append(dict(method=method,J=J,n=n,replication=rep,B1_MSE=vals[0],B2_MSE=vals[1],iterations=int(scalar(r['iterations'])),seconds=scalar(r['seconds'])))
                seen2.add((n,rep));sources.append(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            except Exception as exc:errors.append(dict(path=str(p),error=str(exc)))
        counts[method]=dict(valid=len(seen2),expected=600,complete=seen2==expected2)
    for name,rows in [('DEEP_CURRENT_REPLICATES',deep),('J150_CURRENT_COMPARATORS',comparators)]:
        if rows:pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    audit=dict(checked_at=datetime.now(timezone.utc).isoformat(),studies=counts,errors=errors,all_complete=all(x['complete'] for x in counts.values()) and not errors,sources=sources,notice='Partial results are monitoring outputs, not final study summaries. Superseded studies are excluded.')
    (output/'CURRENT_RUN_AUDIT.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k!='sources'},indent=2))
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path);args=p.parse_args()
    collect(args.root,args.output or args.root/'output/manuscript_update_20261103')
