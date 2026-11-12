"""Audit stopping and numerical diagnostics for completed current deep endpoints."""
from pathlib import Path
from datetime import datetime, timezone
import argparse,json,re
import h5py,numpy as np,pandas as pd

def audit(root,output):
    rows=[];errors=[]
    for endpoint in sorted((root/'results/manuscript_deep_pyramid_20261103').glob('full*_result.mat')):
        path=endpoint.with_name(endpoint.name.replace('_result','_checkpoint'))
        try:
            n,rep,arm=map(int,re.fullmatch(r'full_n(\d+)_r(\d+)_a(\d)_result.mat',endpoint.name).groups())
            with h5py.File(path) as f:
                g=f['f']; scalar=lambda x:float(np.asarray(x).squeeze())
                it=int(scalar(g['iteration']));stable=bool(scalar(g['stabilized']));cap=130 if arm==3 else 160
                assert 0<it<=cap and (stable or it==cap),'neither stabilized nor reached cap'
                traces=[f[ref] for ref in g['trace'][()].ravel()]
                assert len(traces)==it,'trace length mismatch'
                flags=[scalar(t['flags']) for t in traces];kkt=[scalar(t['KKT']) for t in traces]
                last=traces[-1]
                assert np.isfinite(kkt).all(),'nonfinite KKT'
                assert scalar(last['beta'])==1 and scalar(last['lambda'])==1,'warmup not completed'
                rows.append(dict(n=n,replication=rep,arm=arm,iterations=it,stabilized=stable,at_cap=it==cap,iterations_with_flags=sum(x!=0 for x in flags),max_KKT=max(kkt),final_KKT=kkt[-1],total_rejections=sum(scalar(t['rejections']) for t in traces)))
        except Exception as exc:errors.append(dict(path=str(path),error=str(exc)))
    output.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(output/'DEEP_CHECKPOINT_DIAGNOSTICS.csv',index=False)
    groups=[]
    if len(df):
        for (n,arm),g in df.groupby(['n','arm']):groups.append(dict(n=int(n),arm=int(arm),completed=len(g),stabilized=int(g.stabilized.sum()),at_cap=int(g.at_cap.sum()),fits_with_flags=int((g.iterations_with_flags>0).sum()),max_final_KKT=float(g.final_KKT.max())))
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),audited=len(rows),expected=1800,complete=len(rows)==1800 and not errors,errors=errors,groups=groups,interpretation='Reaching the iteration cap is not evidence of convergence. Flags and KKT diagnostics are reported separately from endpoint validity.')
    (output/'DEEP_CHECKPOINT_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.root,a.output)
