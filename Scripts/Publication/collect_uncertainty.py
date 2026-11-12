"""Read-only exact-MAP chain collector; writes a separate complete-protocol audit."""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,hashlib,csv,datetime
import h5py,numpy as np
from scipy.special import expit
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata,norm
# Paths are assigned only by the explicit CLI below; importing cannot start work.
INPUTS=CHAINS=OUT=None

def locate(root,name,filename):
 matches=[p.parent for p in root.rglob(filename) if p.parent.name==name]
 if len(matches)!=1:raise ValueError(f'Expected exactly one {name}/{filename}; found {len(matches)}')
 return matches[0]

def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
 if isinstance(x,np.integer):return int(x)
 return x
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n');t.replace(p)
def split(x):return np.concatenate([x[:,:x.shape[1]//2],x[:,x.shape[1]//2:]],axis=0)
def ranknorm(x):return norm.ppf((rankdata(x.reshape(-1,x.shape[-1]),axis=0)-.375)/(x.shape[0]*x.shape[1]+.25)).reshape(x.shape)
def rh(x):
 y=split(x);n=y.shape[1];w=y.var(1,ddof=1).mean(0);v=(n-1)/n*w+y.mean(1).var(0,ddof=1)
 return np.sqrt(np.divide(v,w,out=np.full_like(v,np.nan),where=w>0))
def ess(x):
 y=split(x);m,n,p=y.shape;w=y.var(1,ddof=1).mean(0);v=(n-1)/n*w+y.mean(1).var(0,ddof=1)
 f=np.fft.rfft(y-y.mean(1,keepdims=True),n=2*n,axis=1);ac=np.fft.irfft(f*np.conjugate(f),axis=1)[:,:n]/n
 rho=1-np.divide(w[None,:]-ac.mean(0),v[None,:],out=np.full((n,p),np.nan),where=v[None,:]>0);rho[0]=1
 vals=np.full(p,np.nan)
 for j in range(p):
  if v[j]<=0 or not np.isfinite(v[j]):continue
  pairs=[]
  for k in range(0,n-1,2):
   q=rho[k,j]+rho[k+1,j]
   if not np.isfinite(q) or q<0:break
   pairs.append(min(q,pairs[-1]) if pairs else q)
  vals[j]=m*n/max(-1+2*sum(pairs),1/np.log10(m*n))
 return vals
def finite_summary(x,kind):
 good=x[np.isfinite(x)]
 return {'defined':len(good),'undefined':int(x.size-len(good)),kind:float((good.max() if kind=='max' else good.min())) if len(good) else None,'median':float(np.median(good)) if len(good) else None}
def diagnostics(x):
 x=x.reshape(4,1000,-1);r=[];b=[];t=[]
 for k in range(0,x.shape[2],64):
  a=x[:,:,k:k+64];z=ranknorm(a);r.extend(np.maximum(rh(z),rh(ranknorm(abs(a-np.median(a,axis=(0,1)))))));b.extend(ess(z));q=np.quantile(a,[.05,.95],axis=(0,1));t.extend(np.minimum(ess((a<=q[0]).astype(float)),ess((a<=q[1]).astype(float))))
 r,b,t=map(np.asarray,(r,b,t));sd=x.reshape(-1,x.shape[-1]).std(0,ddof=1);delta=x[:,500:].mean(1)-x[:,:500].mean(1);scaled=np.divide(abs(delta),sd,out=np.full_like(delta,np.nan),where=sd>0)
 result={'coordinates':x.shape[-1],'rank_folded_split_Rhat':finite_summary(r,'max'),'bulk_ESS':finite_summary(b,'min'),'tail_ESS':finite_summary(t,'min'),'Rhat_above_1_01':int(np.sum(r>1.01)),'Rhat_above_1_1':int(np.sum(r>1.1)),'bulk_ESS_below_400':int(np.sum(b<400)),'tail_ESS_below_400':int(np.sum(t<400)),'max_retained_half_mean_difference':float(abs(delta).max()),'max_retained_half_difference_in_pooled_SD':float(np.nanmax(scaled)) if np.isfinite(scaled).any() else None,'all_diagnostics_defined':bool(np.isfinite(r).all() and np.isfinite(b).all() and np.isfinite(t).all())}
 result['screen_pass']=bool(result['all_diagnostics_defined'] and (r<=1.01).all() and (b>=400).all() and (t>=400).all())
 return result,{'rhat':r,'bulk_ess':b,'tail_ess':t,'half_mean_difference':delta}
def dataset(name):
 paths=[locate(CHAINS,f'{name}_chain{i}','CHAIN.mat') for i in range(1,5)]
 if not all((p/'COMPLETE.json').exists() for p in paths):return None
 fingerprints=[{'path':str(p.resolve()),'chain_bytes':(p/'CHAIN.mat').stat().st_size,'chain_mtime_ns':(p/'CHAIN.mat').stat().st_mtime_ns} for p in paths]
 dest=OUT/f'{name}_SUMMARY.json'
 if dest.exists():
  old=json.loads(dest.read_text());assert old['input_fingerprints']==fingerprints,'Completed chain changed';return old
 with h5py.File(locate(INPUTS,name,'INPUT.mat')/'INPUT.mat') as h:truth=h['truth/B1_scaled'][:].T[:,1:11]
 aa=[];vv=[];dd=[];align=[]
 for path in paths:
  assert json.loads((path/'COMPLETE.json').read_text())['iterations']==2000
  with h5py.File(path/'CHAIN.mat') as h:
   assert int(h['it'][0,0])==2000
   B=h['drawB'][1000:].transpose(0,2,1);C=h['drawC'][1000:].transpose(0,2,1);P=h['drawp'][:].T[1000:];G=h['drawgamma'][:].T[1000:]
   dim=np.stack([h['drawactive1'][:].T[1000:].sum(1),h['drawactive2'][:].T[1000:].sum(1)],axis=1)
  assert all(np.isfinite(v).all() for v in [B,C,P,G]) and (G>0).all() and B.shape[0]==1000
  u=(np.arange(2**P.shape[1])[:,None]>>np.arange(P.shape[1]))&1;W=[];V=[]
  for b,c,p,g in zip(B,C,P,G):
   pu=np.prod(np.where(u,p,1-p),axis=1);q=expit(c[:,0]+u@c[:,1:].T);mu=pu@q;v=(q*pu[:,None]).T@q-np.outer(mu,mu);np.fill_diagonal(v,mu*(1-mu));variance=g+np.einsum('jk,kl,jl->j',b,v,b);assert (variance>0).all();w=b/np.sqrt(variance)[:,None];W.append(w);V.append((w@v@w.T)[np.triu_indices(len(b),1)])
  W=np.asarray(W);m=W.mean(0);cos=m.T@truth/np.maximum(np.linalg.norm(m,axis=0)[:,None]*np.linalg.norm(truth,axis=0)[None,:],1e-30);ii,jj=linear_sum_assignment(-abs(cos));a=np.full((1000,*truth.shape),np.nan)
  for i,j in zip(ii,jj):a[:,:,j]=W[:,:,i]*(1 if cos[i,j]>=0 else -1)
  assert np.isfinite(a).all();aa.append(a);vv.append(V);dd.append(dim);align.append({'source_slots':ii+1,'truth_slots':jj+1,'signs':np.where(cos[ii,jj]>=0,1,-1)})
 a=np.asarray(aa);v=np.asarray(vv);d=np.asarray(dd);lo,hi=np.quantile(a.reshape(-1,*truth.shape),[.025,.975],axis=0);covered=(lo<=truth)&(hi>=truth);nz=abs(truth)>1e-12
 result={'dataset':name,'input_fingerprints':fingerprints,'chains':4,'burnin':1000,'retained_per_chain':1000,'alignment':align,'estimand':'Normalized shallow non-intercept coefficients matched to 10 true factors using one fixed chain-mean assignment per chain; remaining candidate columns are not additional coverage entries. No dimension conditioning.','strata':{}}
 for label,mask in [('all',np.ones(truth.shape,bool)),('nonzero',nz),('zero',~nz)]:result['strata'][label]={'enclosed':int(covered[mask].sum()),'total':int(mask.sum()),'fraction':float(covered[mask].mean()),'mean_interval_width':float((hi-lo)[mask].mean())}
 arrays={}
 for label,x in [('loadings',a),('invariant_correlations',v),('active_dimensions',d)]:
  result[label],detail=diagnostics(x)
  for k,val in detail.items():arrays[f'{label}_{k}']=val
 pairs,counts=np.unique(d.reshape(-1,2),axis=0,return_counts=True);result['dimension_probabilities']=[{'K1':int(pair[0]),'K2':int(pair[1]),'probability':int(count)/4000} for pair,count in zip(pairs,counts)]
 result['dimension_chain_means']=d.mean(1);result['dimension_retained_half_means']=np.stack([d[:,:500].mean(1),d[:,500:].mean(1)],axis=1)
 result['interpretation']='Finite-budget diagnostics; failed mixing precludes a reliable posterior calibration claim. Undefined constant-series diagnostics remain explicitly undefined, not automatic passes.'
 np.savez_compressed(OUT/f'{name}_DETAILS.npz',**arrays,lower=lo,upper=hi,truth=truth,dimensions=d)
 save(dest,result);print('COLLECTED',name,'loading Rhat',result['loadings']['rank_folded_split_Rhat']['max'],flush=True);return clean(result)
def aggregate(results):
 rows=[]
 for j,n in [(50,1000),(50,2000),(100,1000),(100,2000)]:
  ds=[s for s in results if s['dataset'].startswith(f'J{j}_n{n}_')]
  for label in ['all','nonzero','zero']:
   cov=np.array([s['strata'][label]['fraction'] for s in ds]);wid=np.array([s['strata'][label]['mean_interval_width'] for s in ds]);rows.append({'J':j,'n':n,'stratum':label,'available_datasets':len(ds),'planned_datasets':10,'mean_coverage':float(cov.mean()) if len(ds) else None,'dataset_MC_SE':float(cov.std(ddof=1)/np.sqrt(len(ds))) if len(ds)>1 else None,'mean_interval_width':float(wid.mean()) if len(ds) else None,'width_dataset_MC_SE':float(wid.std(ddof=1)/np.sqrt(len(ds))) if len(ds)>1 else None,'loading_diagnostic_pass_datasets':sum(s['loadings']['screen_pass'] for s in ds),'correlation_diagnostic_pass_datasets':sum(s['invariant_correlations']['screen_pass'] for s in ds)})
 with open(OUT/'COVERAGE_SUMMARY.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 save(OUT/'PROGRESS.json',{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'datasets_collected':len(results),'planned_datasets':40,'summary':rows,'MC_SE_scope':'Between-dataset sample SD divided by sqrt(available datasets), not MCMC error; incomplete subsets are provisional. No trimming/exclusion of poor mixing.','diagnostics':'Rank/folded split Rhat; rank-normalized bulk ESS; 5/95-percent indicator tail ESS using FFT autocovariance and Geyer positive monotone paired sequence; retained-half differences; active dimensions.','screen':'All diagnostics defined, Rhat<=1.01, bulk/tail ESS>=400. Heuristic diagnostic screen, not proof of convergence.'})
def export_diagnostics(results):
 rows=[];dimensions=[]
 for s in results:
  for target in ['loadings','invariant_correlations','active_dimensions']:
   x=s[target];rows.append(dict(dataset=s['dataset'],target=target,coordinates=x['coordinates'],max_rank_folded_Rhat=x['rank_folded_split_Rhat']['max'],undefined_Rhat=x['rank_folded_split_Rhat']['undefined'],min_bulk_ESS=x['bulk_ESS']['min'],undefined_bulk_ESS=x['bulk_ESS']['undefined'],min_tail_ESS=x['tail_ESS']['min'],undefined_tail_ESS=x['tail_ESS']['undefined'],Rhat_above_1_01=x['Rhat_above_1_01'],bulk_ESS_below_400=x['bulk_ESS_below_400'],tail_ESS_below_400=x['tail_ESS_below_400'],max_half_mean_difference=x['max_retained_half_mean_difference'],max_half_difference_in_pooled_SD=x['max_retained_half_difference_in_pooled_SD'],screen_pass=x['screen_pass']))
  for item in s['dimension_probabilities']:dimensions.append(dict(dataset=s['dataset'],**item))
 for name,table in [('DATASET_DIAGNOSTICS.csv',rows),('ACTIVE_DIMENSION_PROBABILITIES.csv',dimensions)]:
  if table:
   with (OUT/name).open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)

if __name__=='__main__':
 from paths import output
 ap=argparse.ArgumentParser(description='Compute aligned intervals, coverage, mixing diagnostics and dimension probabilities from raw chains; never starts sampling.')
 ap.add_argument('--datasets-dir',type=Path,required=True);ap.add_argument('--chains-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--limit',type=int)
 args=ap.parse_args();INPUTS=args.datasets_dir.resolve();CHAINS=args.chains_dir.resolve();OUT=output(args.output_dir);results=[]
 for rep in range(1,11):
  for j,n in [(50,1000),(50,2000),(100,1000),(100,2000)]:
   s=dataset(f'J{j}_n{n}_r{rep}')
   if s:results.append(s)
   if args.limit and len(results)>=args.limit:break
  if args.limit and len(results)>=args.limit:break
 aggregate(results);export_diagnostics(results)
 if len(results)!=40:print('PROVISIONAL: only',len(results),'of40 complete datasets; do not use as final study results.')
