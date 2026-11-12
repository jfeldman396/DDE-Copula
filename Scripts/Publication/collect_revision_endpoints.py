"""Aggregate portable endpoint.mat files produced by run_revision_case."""
from pathlib import Path
import argparse, json, itertools
import h5py
import numpy as np
import pandas as pd
from paths import output
NS = [500,1000,2000,4000,8000,16000]

def read(path):
    with h5py.File(path) as h:
        e = h['endpoint']; result = {}
        for k, v in e.items():
            a = v[()]
            kind = v.attrs.get('MATLAB_class', b'')
            if kind == b'char': result[k] = ''.join(map(chr, a.ravel().astype(int)))
            elif a.size == 1: result[k] = a.item()
            else: result[k] = a.T
        return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--study',choices=['shallow','comparison','deep'],required=True)
    p.add_argument('--preview',action='store_true',help='Permit explicitly incomplete diagnostics, never final study results')
    a=p.parse_args(); records={}; sources=[]
    for path in sorted(a.input_dir.rglob('endpoint.mat')):
        e=read(path)
        accepted = e['study']=='deep' if a.study=='deep' else (e['study']=='shallow' if a.study=='shallow' else e['study'] in ('shallow','comparator') and e['method'] in ('map','rl','poisson'))
        if not accepted: continue
        key=(int(e['J']),int(e['n']),int(e['replication']),e['method'])
        assert key not in records, f'Duplicate endpoint: {key}'
        assert e['complete'], f'Incomplete fit is not a study endpoint: {path}'
        if a.study=='comparison': assert e.get('native15_5'), f'Wrong comparison widths: {path}'
        records[key]=e; sources.append(dict(J=key[0],n=key[1],replication=key[2],method=key[3],file=str(path.resolve())))
    methods=['map'] if a.study=='deep' else (['map','vi','tempering'] if a.study=='shallow' else ['map','rl','poisson'])
    js=[108] if a.study=='deep' else [50,100,150]
    expected=set(itertools.product(js,NS,range(1,101),methods)); missing=expected-records.keys()
    assert records and not (records.keys()-expected), 'Empty or unexpected study grid'
    if missing and not a.preview: raise SystemExit(f'Missing {len(missing)} endpoints; final results withheld. Use --preview for diagnostics.')
    out=output(a.output_dir); prefix='PREVIEW_' if a.preview else ''
    pd.DataFrame(sources).to_csv(out/(prefix+'ENDPOINTS.csv'),index=False)
    recovery=[];errors=[];matrices={};truths={}
    for (J,n,rep,method),e in records.items():
        if not e['initialization_failed']:
            for layer in range(1,4 if a.study=='deep' else 3):
                if f'B{layer}' not in e: continue
                B=e[f'B{layer}'];T=e[f'truth{layer}'];value=float(np.mean((B-T)**2))
                assert np.isfinite(B).all() and np.isclose(value,e[f'MSE{layer}'],rtol=1e-10,atol=1e-12)
                errors.append(dict(method=method,J=J,n=n,replication=rep,layer=layer,eligible=bool(e['eligible']),MSE=value))
                matrices[method,J,n,rep,layer]=B
                if a.study=='comparison':
                    if (J,layer) in truths: assert np.array_equal(truths[J,layer],T)
                    truths[J,layer]=T
    for J,n,method in itertools.product(js,NS,methods):
        group=[e for (j,nn,r,m),e in records.items() if (j,nn,m)==(J,n,method)]
        if not group: continue
        good=[e for e in group if not e['initialization_failed']]
        row=dict(J=J,n=n,method=method,attempts=len(group),successful_fits=len(good),initialization_failed=len(group)-len(good),both_correct_pct=100*sum(bool(e['eligible']) for e in good)/len(group))
        if a.study=='shallow':
            row.update(arm=methods.index(method)+1,completed=len(good),K1_correct_pct=100*sum(e['K1']==10 for e in good)/len(group),K2_correct_pct=100*sum(e['K2']==3 for e in good)/len(group))
        if a.study=='deep':
            row.update({f'K{d+1}_correct_pct':100*sum(np.ravel(e['K'])[d]==k for e in good)/len(group) for d,k in enumerate([18,6,2])})
        recovery.append(row)
    pd.DataFrame(recovery).to_csv(out/(prefix+'DIMENSION_RECOVERY.csv'),index=False)
    if errors:
        raw=pd.DataFrame(errors);raw['retained']=False; summaries=[];means={};selections=[]
        for (method,J,n,layer),g in raw.groupby(['method','J','n','layer']):
            v=g[g.eligible];q1,q3=v.MSE.quantile([.25,.75]);cut=q3+1.5*(q3-q1);keep=v[v.MSE<=cut];raw.loc[keep.index,'retained']=True
            summaries.append(dict(method=method,J=J,n=n,layer=layer,attempted=sum(1 for key in records if (key[0],key[1],key[3])==(J,n,method)),eligible=len(v),retained=len(keep),excluded=len(v)-len(keep),q1=q1,q3=q3,upper_cutoff=cut,MSE=keep.MSE.mean(),untrimmed_MSE=v.MSE.mean()))
            selections.append(dict(method=method,J=int(J),n=int(n),layer=int(layer),retained=keep.replication.astype(int).tolist(),excluded=v[v.MSE>cut].replication.astype(int).tolist(),ineligible=g[~g.eligible].replication.astype(int).tolist()))
            for label,subset in [('retained',keep),('all_eligible',v)]:
                if len(subset):means[f'{method}_J{J}_n{n}_B{layer}_{label}']=np.mean([matrices[method,J,n,int(rep),layer] for rep in subset.replication],axis=0)
        raw.to_csv(out/(prefix+'REPLICATE_MSE_UNTRIMMED_ARCHIVE.csv'),index=False)
        pd.DataFrame(summaries).to_csv(out/(prefix+'MSE_SUMMARY.csv'),index=False)
        for (J,layer),T in truths.items(): means[f'truth_J{J}_B{layer}']=T
        np.savez_compressed(out/(prefix+'AVERAGE_WEIGHT_MATRICES.npz'),**means)
        (out/(prefix+'ELIGIBILITY_AND_EXCLUSIONS.json')).write_text(json.dumps(selections,indent=2)+'\n')
    # New-fit timings are measured anew; identical hardware timings are not expected.
    timing=[dict(n=n,replication=rep,**{k:e[k] for k in ['CPU','numerical_wall','warmup_wall','post_wall']}) for (J,n,rep,method),e in records.items() if J==100 and method=='map' and all(k in e for k in ['CPU','numerical_wall','warmup_wall','post_wall'])]
    if timing:
        t=pd.DataFrame(timing);t.to_csv(out/(prefix+'J100_MAP_RUNTIME_REPLICATES.csv'),index=False)
        rows=[]
        for n,g in t.groupby('n'):
            assert np.allclose(g.numerical_wall,g.warmup_wall+g.post_wall)
            rows.append(dict(n=n,replications=len(g),mean_seconds=g.numerical_wall.mean(),sd_seconds=g.numerical_wall.std(ddof=1),median_seconds=g.numerical_wall.median(),mean_CPU_seconds=g.CPU.mean(),mean_warmup_seconds=g.warmup_wall.mean(),mean_post_seconds=g.post_wall.mean(),mean_minutes=g.numerical_wall.mean()/60))
        pd.DataFrame(rows).to_csv(out/(prefix+'J100_MAP_RUNTIME_SUMMARY.csv'),index=False)
    if a.study=='deep' and errors:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        summary=pd.DataFrame(summaries)
        for layer in [1,2,3]:
            g=summary[summary.layer==layer].sort_values('n');fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
            ax.plot(g.n,g.MSE,'o-',label='Trimmed mean');ax.plot(g.n,g.untrimmed_MSE,'s--',label='Untrimmed mean')
            ax.set(xscale='log',xlabel='Sample size (n)',ylabel='Entry-wise MSE',title=f'No-floor DDE: layer {layer}');ax.set_xticks(g.n,g.n.astype(str));ax.grid(alpha=.2);ax.legend()
            fig.savefig(out/f'{prefix}B{layer}_MSE.pdf');plt.close(fig)
    (out/(prefix+'STATUS.json')).write_text(json.dumps(dict(study=a.study,complete=not missing,preview=a.preview,expected=len(expected),available=len(records),missing=len(missing),fitting_started=False),indent=2)+'\n')
    print(out)

if __name__=='__main__': main()
