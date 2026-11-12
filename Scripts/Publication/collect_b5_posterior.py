from pathlib import Path
import os,json,csv
os.environ.setdefault('MPLCONFIGDIR','/tmp/dde-mpl')
import h5py,numpy as np
from scipy.special import expit
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from paths import output,PACKAGE
p=argparse.ArgumentParser(description='Align full B5 chains and summarize sweeps1001–2000 conditional on10/1 dimensions.')
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--chains-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args();out=output(args.output_dir)
with h5py.File(args.checkpoint) as f:
 r=f['record'];rb=r['B1'][:].T[:,1:];rc=r['B2'][:].T;rp=r['prop'][:].ravel();rg=r['gamma'][:].ravel()
slots=np.flatnonzero(np.linalg.norm(rb,axis=0)>1e-12);assert len(slots)==10
u=(np.arange(16)[:,None]>>np.arange(4))&1
def moments(b,c,p,g):
 pu=np.prod(np.where(u,p,1-p),axis=1);q=expit(c[:,0]+u@c[:,1:].T);mu=pu@q
 v=(q*pu[:,None]).T@q-np.outer(mu,mu);np.fill_diagonal(v,mu*(1-mu));cov=b@v@b.T
 return b/np.sqrt(g+np.diag(cov))[:,None],mu,cov
rw,_,_=moments(rb,rc,rp,rg)
sets=[];audit=[];maxerr=0
for name,path in [(f'chain_{i}',args.chains_dir/f'B5_chain{i}'/'CHAIN.mat') for i in range(1,5)]:
 with h5py.File(path) as f:
  a1=f['drawactive1'][:].T;a2=f['drawactive2'][:].T
  assert int(f['it'][()].item())>=2000,'Chain has not reached the required window'
  ix=np.flatnonzero((np.arange(len(a1))>=1000)&(np.arange(len(a1))<2000)&(a1.sum(1)==10)&(a2.sum(1)==1))
  bs=f['drawB'][:].transpose(0,2,1);cs=f['drawC'][:].transpose(0,2,1);ps=f['drawp'][:].T;gs=f['drawgamma'][:].T;b0s=f['drawb0'][:].T
 ws=[];ds=[];probs=[];matches=[];cosines=[]
 for t in ix:
  b=bs[t].copy();c=cs[t].copy();p=ps[t].copy();b0=b0s[t].copy();w,mu,cov=moments(b,c,p,gs[t]);marg=b0+b@mu
  src=np.flatnonzero(a1[t]);ref=rw[:,slots];cand=w[:,src]
  sim=cand.T@ref/(np.linalg.norm(cand,axis=0)[:,None]*np.linalg.norm(ref,axis=0)[None,:])
  rows,cols=linear_sum_assignment(-abs(sim));order=np.arange(12);sign=np.ones(12)
  for r,k in zip(rows,cols):order[slots[k]]=src[r];sign[slots[k]]=1 if sim[r,k]>=0 else -1
  unused=[k for k in range(12) if k not in src];inactive=[k for k in range(12) if k not in slots]
  order[inactive]=unused
  b=b[:,order];c=c[order,:]
  for k in np.flatnonzero(sign<0):b0+=b[:,k];b[:,k]*=-1;c[k,:]*=-1
  upper=int(np.flatnonzero(a2[t])[0]);uo=[upper]+[k for k in range(4) if k!=upper];c=np.column_stack([c[:,0],c[:,1:][:,uo]]);p=p[uo]
  flip=np.dot(c[slots,1],rc[slots,1])<0
  if flip:c[:,0]+=c[:,1];c[:,1]*=-1;p[0]=1-p[0]
  ww,mm,cc=moments(b,c,p,gs[t]);err=max(np.max(abs(cc-cov)),np.max(abs(b0+b@mm-marg)));maxerr=max(maxerr,float(err));assert err<1e-7
  ws.append(ww[:,slots]);ds.append(c[slots,:2]);probs.append(p[0]);matches.append([int(t+1),*(order[slots]+1).tolist(),upper+1,int(flip)]);cosines.extend(abs(sim[rows,cols]).tolist())
 data={'shallow':np.array(ws),'deep':np.array(ds),'p':np.array(probs),'sweep':ix+1};sets.append(data)
 np.savez_compressed(out/f'{name}_aligned_draws.npz',**data)
 np.savetxt(out/f'{name}_alignment.csv',np.array(matches,dtype=int),delimiter=',',fmt='%d',header='sweep,'+','.join(f'S{k+1}_source' for k in slots)+',upper_source,upper_complemented',comments='')
 audit.append({'chain':name,'conditional_draws':len(ix),'minimum_absolute_matching_cosine':float(min(cosines)),'median_absolute_matching_cosine':float(np.median(cosines))})
def summarize(name,data):
 for kind in ['shallow','deep']:
  z=data[kind];med=np.median(z,axis=0);lo,hi=np.quantile(z,[.025,.975],axis=0);pos=(z>0).mean(0)
  with (out/f'{name}_{kind}.csv').open('w') as f:
   wr=csv.writer(f);wr.writerow(['row','column','mean','median','lower95','upper95','Pr_positive','Pr_negative'])
   for j,k in np.ndindex(med.shape):wr.writerow([j+1 if kind=='shallow' else f'S{slots[j]+1}',f'S{slots[k]+1}' if kind=='shallow' else ['intercept','U1'][k],z[:,j,k].mean(),med[j,k],lo[j,k],hi[j,k],pos[j,k],1-pos[j,k]])
 return med,lo,hi
pool={k:np.concatenate([s[k] for s in sets[:4]]) for k in sets[0]}
np.savez_compressed(out/'original_four_pooled_aligned_draws.npz',**pool)
for name,data in [(a['chain'],s) for a,s in zip(audit,sets)]+[('original_four_pooled',pool)]:summarize(name,data)
z=pool['deep'][:,:,1];med=np.median(z,0);lo,hi=np.quantile(z,[.025,.975],axis=0)
fig,ax=plt.subplots(figsize=(8,6),layout='constrained');y=np.arange(10)
ax.errorbar(med,y,xerr=[med-lo,hi-med],fmt='o',color='#176d91',capsize=3,label='Conditional median and 95% interval');ax.scatter(rc[slots,1],y,marker='x',color='#b45309',label='MAP');ax.axvline(0,color='gray',ls='--');ax.set_yticks(y,[f'S{k+1}–U1' for k in slots]);ax.invert_yaxis();ax.set_xlabel('Raw log-odds coefficient');ax.set_title(f'B5: conditional on 10/1 active dimensions\nNew-seed chain1 + chains2–4 · sweeps1001–2000 · {len(z):,} aligned draws · preliminary');ax.legend(fontsize=9)
fig.savefig(out/'deep_intervals.png',dpi=170);fig.savefig(out/'deep_intervals.pdf')
report={'condition':'CSP active counts 10/1, sweeps 1001–2000; inactive coefficients retained in standardization, no thresholding','alignment':'Per-draw active shallow slots matched to MAP with absolute cosine and Hungarian assignment; binary complements propagated to intercepts and connected coefficients; sole active upper slot aligned to MAP U1 orientation. This is a reference-based summary, not a claim of exchangeable prior symmetry.','pooling':'Concatenate qualifying draws of replacement-seed chain1 and original chains2–4. No equal reweighting of conditional chain samples.','limitations':'Preliminary conditional summaries; filtering does not establish conditional mixing; intervals marginal, not simultaneous; matching assumes correspondence to MAP factors.','chains':audit,'pooled_draws':len(z),'max_mean_covariance_invariance_error':maxerr,'deep_U1':[{'slot':f'S{k+1}','median':float(med[j]),'lower95':float(lo[j]),'upper95':float(hi[j]),'Pr_positive':float((z[:,j]>0).mean())} for j,k in enumerate(slots)]}
(out/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

a=np.load(out/'original_four_pooled_aligned_draws.npz');slots=np.array([1,2,3,4,5,6,7,9,10,12])
with h5py.File(args.checkpoint) as f:rb=f['record/B1'][:].T[:,1:][:,slots-1]
with (PACKAGE/'Results/Publication/data/ivae/item_metadata.csv').open() as f:meta=list(csv.DictReader(f))
order=np.argsort(np.argmax(abs(rb),axis=1),kind='stable');cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('white')
for kind,z in [('shallow',a['shallow']),('deep',a['deep'][:,:,1:])]:
 med=np.median(z,axis=0);lo,hi=np.quantile(z,[.025,.975],axis=0);mask=(lo<=0)&(hi>=0)
 if kind=='shallow':med=med[order];mask=mask[order];labels=[meta[i]['item_id'].replace('BIGFIVE_','Item ') for i in order]
 else:labels=[f'S{k}' for k in slots]
 fig,ax=plt.subplots(figsize=(8,18) if kind=='shallow' else (5,7),layout='constrained');lim=float(abs(med).max());im=ax.imshow(np.ma.array(med,mask=mask),cmap=cmap,vmin=-lim,vmax=lim,aspect='auto')
 ax.set_yticks(range(len(labels)),labels,fontsize=7 if kind=='shallow' else 11);ax.set_xticks(range(med.shape[1]),[f'S{k}' for k in slots] if kind=='shallow' else ['U1'])
 ax.set_xlabel('Shallow factor' if kind=='shallow' else 'Upper factor')
 if kind=='shallow':
  groups=np.argmax(abs(rb[order]),axis=1)
  for j in range(1,len(order)):
   if groups[j]!=groups[j-1]:ax.axhline(j-.5,color='#777777',lw=.5)
 else:
  for j in range(len(labels)):
   if not mask[j,0]:ax.text(0,j,f'{med[j,0]:.2f}',ha='center',va='center',color='white' if abs(med[j,0])>.55*lim else 'black')
 ax.set_title(f'B5 {kind} weights · conditional on 10/1\nPosterior medians; white if 95% interval includes zero\nNew-seed chain 1 + chains 2–4 · sweeps 1,001–2,000',fontsize=10)
 fig.colorbar(im,ax=ax,shrink=.6,label='Standardized loading' if kind=='shallow' else 'Raw log-odds coefficient')
 for ext in ['png','pdf']:fig.savefig(out/f'{kind}_interval_masked_weights.{ext}',dpi=170)
 plt.close(fig);print(kind,'white',int(mask.sum()),'total',mask.size)

# Exact final MAP/conditional-posterior figure layout, with document edits omitted.
a=np.load(out/'original_four_pooled_aligned_draws.npz');slots=np.array([1,2,3,4,5,6,7,9,10,12])-1
with h5py.File(args.checkpoint) as f:
 r=f['record'];b=r['B1'][:].T[:,1:];c=r['B2'][:].T;p=r['prop'][:].ravel();g=r['gamma'][:].ravel()
u=(np.arange(16)[:,None]>>np.arange(4))&1;pu=np.prod(np.where(u,p,1-p),1);q=expit(c[:,0]+u@c[:,1:].T);mu=pu@q;v=(q*pu[:,None]).T@q-np.outer(mu,mu);np.fill_diagonal(v,mu*(1-mu));w=b/np.sqrt(g+np.einsum('jk,kl,jl->j',b,v,b))[:,None]
meta=list(csv.DictReader(open(PACKAGE/'Results/Publication/data/ivae/item_metadata.csv')));order=np.argsort(np.argmax(abs(w[:,slots]),axis=1),kind='stable');cm=plt.get_cmap('RdBu_r').copy();cm.set_bad('white')
for kind,z,point in [('shallow',a['shallow'],w[:,slots]),('deep',a['deep'][:,:,1:],c[slots,1:2])]:
 med=np.median(z,0);lo,hi=np.quantile(z,[.025,.975],axis=0);mask=(lo<=0)&(hi>=0)
 if kind=='shallow':point=point[order];med=med[order];mask=mask[order];labels=[meta[j]['item_id'].replace('BIGFIVE_','') for j in order]
 else:labels=[f'S{k+1}' for k in slots]
 fig,axs=plt.subplots(1,2,figsize=(10,14) if kind=='shallow' else (7,5),layout='constrained');lim=max(abs(point).max(),abs(med).max())
 for ax,arr,title in zip(axs,[np.ma.masked_where(point==0,point),np.ma.array(med,mask=mask)],['MAP','Posterior median']):
  im=ax.imshow(arr,cmap=cm,vmin=-lim,vmax=lim,aspect='auto');ax.set_title(title);ax.set_yticks(range(len(labels)),labels,fontsize=6 if kind=='shallow' else 10);ax.set_xticks(range(arr.shape[1]),[f'S{k+1}' for k in slots] if kind=='shallow' else ['U1']);ax.set_ylabel('Item' if kind=='shallow' else 'Shallow factor')
 fig.colorbar(im,ax=axs,shrink=.6,label='Standardized loading' if kind=='shallow' else 'Log-odds coefficient')
 for ext in ['pdf','png']:fig.savefig(out/f'{kind}_map_posterior.{ext}',dpi=180)
 plt.close(fig)
