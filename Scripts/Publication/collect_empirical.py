"""Analyze a fresh selected empirical DDE fit; original survey data on request."""
from pathlib import Path
import argparse,csv,json,sys,time,hashlib
import numpy as np,pandas as pd,h5py
from scipy.io import loadmat
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import output,PACKAGE
sys.path.insert(0,str(PACKAGE/'Algorithms/DDE'))
from dde_predict import DDEModel,calibrate,states
from dde_embeddings import DDEEmbedder
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True,help='MODEL_it0740_target0800.mat from selected maxima config9')
p.add_argument('--data',type=Path,required=True,help='Original author-provided survey CSV')
p.add_argument('--dde-scores',type=Path,required=True,help='Fresh selected DDE PMSE_REPLICATES.csv')
p.add_argument('--gaussian-dir',type=Path,required=True,help='Fresh run_gaussian_study stage8k directory')
p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
expected=json.loads((PACKAGE/'Results/Publication/data/ivae/RUN_MANIFEST.json').read_text())['data_sha256']
assert hashlib.sha256(args.data.read_bytes()).hexdigest()==expected
survey=pd.read_csv(args.data);metadata=PACKAGE/'Results/Publication/data/ivae/item_metadata.csv';meta=pd.read_csv(metadata)
Y=survey[meta.item_id].to_numpy(float);counts=np.stack([(Y==c).sum(0) for c in range(1,6)],axis=1)
prob=(counts+.5)/(counts.sum(1,keepdims=True)+2.5);edges=np.c_[np.zeros(98),np.cumsum(prob,axis=1)];edges[:,-1]=1
assert Y.shape==(1500,98) and np.isnan(Y).sum()==223
data=dict(Y=Y,row_ids=np.arange(1,1501),cdf_edges=edges)
O=output(args.output_dir);out=d=O
# Main-text response-distribution table. The supplied CSV merges missing codes;
# report missingness without inventing a skipped-versus-not-asked distinction.
item=survey['BIGFIVE_5']
pd.DataFrame([dict(item='BIGFIVE_5',response=str(c),count=int((item==c).sum())) for c in range(1,6)] +
             [dict(item='BIGFIVE_5',response='missing',count=int(item.isna().sum()))]).to_csv(O/'SURVEY_ITEM_RESPONSE_COUNTS.csv',index=False)
def weighted_means(Y,W,columns):
    valid=np.isfinite(Y[:,columns]);y=np.where(valid,Y[:,columns],0.)
    mass=W.T@valid.astype(float);sums=W.T@y
    assert (mass>0).all()
    return (sums/mass).T,mass.T,sums.T

with h5py.File(args.model,'r') as f:
    g=f['model'];k1,k2=map(int,g['widths'][()].ravel())
    m=DDEModel(g['B1'][()].T,g['B2'][()].T,g['gamma'][()].ravel(),g['prop'][()].ravel() if k2 else np.zeros(0))
    ids1=g['native_first_layer_ids'][()].ravel().astype(int)
    ids2=g['native_top_layer_ids'][()].ravel().astype(int) if k2 else np.zeros(0,dtype=int)
    assert np.array_equal(g['training_row_ids'][()].ravel(),data['row_ids'])
assert k1>0 and k1<=12 and k2<=4
thresholds,cal=calibrate(m,data['cdf_edges'])
e=DDEEmbedder(m,thresholds,ids1,ids2,max_states=1<<12,memory_budget_mb=256)
W=np.zeros((1500,1<<k2));posterior_means=np.zeros((1500,k1+k2))
started=time.perf_counter()
for i,y in enumerate(Y):
    r=e.embed(y,identity=int(data['row_ids'][i]));W[i]=r['top_posterior_probabilities'];posterior_means[i]=r['posterior_mean']
    assert r['integration_pass'] and abs(W[i].sum()-1)<1e-12
top=states(k2).astype(int)
np.savez_compressed(out/'UPPER_POSTERIORS.npz',row_ids=data['row_ids'],upper_probabilities=W,top_states=top,posterior_means=posterior_means,native_ids1=ids1,native_ids2=ids2,thresholds=thresholds)
# Exact model-implied Z variance provides the same primary normalization.
shallow=states(k1);lp=m.lower_logprior(shallow);prior=np.exp(lp-logsumexp(lp))
mean=prior@shallow;cov=(shallow*prior[:,None]).T@shallow-np.outer(mean,mean)
sd=np.sqrt(m.gamma+np.einsum('jk,kl,jl->j',m.B1[:,1:],cov,m.B1[:,1:]))
slopes=m.B1[:,1:]/sd[:,None]
mu=m.B1[:,0]+m.B1[:,1:]@mean
np.savez_compressed(out/'NORMALIZED_POINT.npz',B1=np.column_stack([(m.B1[:,0]-mu)/sd,slopes]),B2=m.B2,gamma=m.gamma/sd**2,prop=m.prop,thresholds=(thresholds-mu[:,None])/sd[:,None],native_ids1=ids1,native_ids2=ids2)
z=np.load(d/'NORMALIZED_POINT.npz');B=z['B1'][:,1:];meta=pd.read_csv(metadata);anchors=[]
for k in range(10):
 score=np.maximum(0,B[:,k]-np.max(np.delete(B,k,axis=1),axis=1))
 for rank,j in enumerate(np.argsort(-score,kind='stable'),1):
  if score[j]>1e-12:anchors.append(dict(factor=k+1,rank=rank,item=meta.iloc[j].item_id,wording=meta.iloc[j].wording,loading=B[j,k],contrast=score[j]))
a=pd.DataFrame(anchors);a.to_csv(O/'ANCHORS_ALL.csv',index=False);a[a['rank']<=3].to_csv(O/'ANCHORS_TOP3.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(O/f'{name}.{ext}',dpi=240,bbox_inches='tight',facecolor='white')
 plt.close(fig)
post=np.load(d/'UPPER_POSTERIORS.npz');A=post['posterior_means'][:,:10];ids=post['row_ids'].astype(int);frame=survey;Y=data['Y'];assert np.array_equal(frame[meta.item_id.tolist()].to_numpy(float)[ids-1],Y,equal_nan=True)
y=frame.ideo5.to_numpy()[ids-1];order=np.r_[np.flatnonzero(y==1),np.flatnonzero(y==5)];groups=[y==1,y==5];means=np.array([A[g].mean(0) for g in groups]);counts=[int(g.sum()) for g in groups]
fig,ax=plt.subplots(1,2,figsize=(10,6),gridspec_kw={'width_ratios':[1.25,1]},layout='constrained');im=ax[0].imshow(A[order],aspect='auto',cmap='Greys',vmin=0,vmax=1,interpolation='nearest');ax[0].axhline(counts[0]-.5,color='#b2182b',lw=1);ax[0].set_yticks([(counts[0]-1)/2,counts[0]+(counts[1]-1)/2],['Very liberal\n(n=%d)'%counts[0],'Very conservative\n(n=%d)'%counts[1]]);ax[0].set_title('(a) Respondent posterior probabilities');im2=ax[1].imshow(means,aspect=3,cmap='Greys',vmin=0,vmax=1);ax[1].set_yticks([0,1],['Very liberal','Very conservative']);ax[1].set_title('(b) Group means')
for j in range(10):
 for i in range(2):ax[1].text(j,i,f'{means[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if means[i,j]>.55 else 'black')
for axis in ax:axis.set_xticks(range(10),range(1,11));axis.set_xlabel('First-layer factor');axis.spines[:].set_visible(False)
fig.colorbar(im,ax=ax,shrink=.6,label=r'$P(A^{(1)}_{ik}=1\mid y_i,\widehat\theta)$');save(fig,'A1_IDEOLOGY')
pd.DataFrame(A[order],columns=[f'L{k+1}' for k in range(10)]).assign(row_id=ids[order],ideo5=y[order]).to_csv(O/'A1_IDEOLOGY_VALUES.csv',index=False)
pd.DataFrame(means,index=['Very liberal','Very conservative'],columns=[f'L{k+1}' for k in range(10)]).to_csv(O/'A1_IDEOLOGY_MEANS.csv')
# The same positive-contrast paper anchors in the higher-level response view.
top=a[a['rank']<=2];columns=[meta.index[meta.item_id==item][0] for item in top.item];W=post['upper_probabilities'];means,mass,sums=weighted_means(Y,W,columns);delta=means-np.nanmean(Y[:,columns],axis=0)[:,None];bound=np.ceil(abs(delta).max()*10)/10
fig,ax=plt.subplots(figsize=(8,8));im=ax.imshow(delta,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto');ax.set_yticks(range(len(top)),[f'L{r.factor}: {r.wording}' for r in top.itertuples()]);ax.set_xticks([0,1],[f'Upper state {g}\n({W[:,g].mean():.1%} posterior share)' for g in range(2)]);ax.xaxis.tick_top();ax.tick_params(length=0)
for i in range(len(top)):
 for g in range(2):ax.text(g,i,f'{means[i,g]:.2f}',ha='center',va='center',color='white' if abs(delta[i,g])>.65*bound else 'black')
for j in np.arange(1.5,len(top),2):ax.axhline(j,color='white',lw=2)
ax.spines[:].set_visible(False);fig.colorbar(im,ax=ax,orientation='horizontal',pad=.04,label='Difference from overall observed item mean');fig.tight_layout();save(fig,'HIGHER_ORDER_PROFILES')
records=[]
for i,row in enumerate(top.to_dict('records')):
 for g in range(2):records.append(dict(**row,upper_state=g,observed_mean=means[i,g],difference=delta[i,g],observed_weight_mass=mass[i,g]))
pd.DataFrame(records).to_csv(O/'HIGHER_ORDER_VALUES.csv',index=False)


# The same positive-contrast paper anchors in the higher-level response view.
top=a[a['rank']==1];columns=[meta.index[meta.item_id==item][0] for item in top.item];W=post['upper_probabilities'];means,mass,sums=weighted_means(Y,W,columns);delta=means-np.nanmean(Y[:,columns],axis=0)[:,None];bound=np.ceil(abs(delta).max()*10)/10
fig,ax=plt.subplots(figsize=(8,4.8));im=ax.imshow(delta,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto');ax.set_yticks(range(len(top)),[f'L{r.factor}: {r.wording}' for r in top.itertuples()]);ax.set_xticks([0,1],[f'Upper state {g}\n({W[:,g].mean():.1%} posterior share)' for g in range(2)]);ax.xaxis.tick_top();ax.tick_params(length=0)
for i in range(len(top)):
 for g in range(2):ax.text(g,i,f'{means[i,g]:.2f}',ha='center',va='center',color='white' if abs(delta[i,g])>.65*bound else 'black')
for j in np.arange(.5,len(top),1):ax.axhline(j,color='white',lw=2)
ax.spines[:].set_visible(False);fig.colorbar(im,ax=ax,orientation='horizontal',pad=.04,label='Difference from overall observed item mean');fig.tight_layout();save(fig,'HIGHER_ORDER_TOP_ANCHORS')
records=[]
for i,row in enumerate(top.to_dict('records')):
 for g in range(2):records.append(dict(**row,upper_state=g,observed_mean=means[i,g],difference=delta[i,g],observed_weight_mass=mass[i,g]))
pd.DataFrame(records).to_csv(O/'HIGHER_ORDER_TOP_ANCHOR_VALUES.csv',index=False)

# Select the Gaussian candidate by the same mean-pMSE rule, then combine only fresh estimates.
candidates=[]
for i in range(1,7):
 folder=args.gaussian_dir/f'c{i}';scores=pd.read_csv(folder/'PMSE_REPLICATES.csv')
 assert len(scores)==50;candidates.append((float(scores.pMSE.mean()),i,folder))
_,candidate,folder=min(candidates)
g=pd.read_csv(folder/'PMSE_REPLICATES.csv');v=pd.read_csv(args.dde_scores)
assert len(v)==50
paired=v[['replicate','pMSE']].merge(g[['replicate','pMSE']],on='replicate',suffixes=('_dde','_gaussian'));assert len(paired)==50
paired['difference']=paired.pMSE_dde-paired.pMSE_gaussian;paired.to_csv(O/'PAIRED_PMSE.csv',index=False)
G=pd.read_csv(folder/'STANDARDIZED_POINT_LOADINGS.csv',header=None).to_numpy(float)
assert B.shape==(98,10) and G.shape==(98,32),'New selected dimensions differ from the reported10/1 and32-column comparison; preserve and investigate.'
frame=meta.copy()
for k in range(10):frame[f'DDE_{k+1}']=B[:,k]*np.sqrt(mean[k]*(1-mean[k]))
for k in range(32):frame[f'Gaussian_{k+1}']=G[:,k]
frame.to_csv(O/'all_standardized_loadings.csv',index=False)
pd.DataFrame([dict(candidate=i,pMSE_mean=m) for m,i,_ in candidates]).to_csv(O/'GAUSSIAN_SELECTION.csv',index=False)
(O/'SUMMARY.json').write_text(json.dumps(dict(selected_gaussian=candidate,empirical_selection='Frozen author-selected maxima config9; not minimum-pMSE fit',rows=1500,items=98,normalization='Profiles/intervals: beta/sd(Z), binary0/1. Comparison CSV: beta*sd(A)/sd(Z), matching the supplied factor comparison.',interpretation='Descriptive full-sample profiles; poor mixing remains a limitation'),indent=2)+'\n')
print(O)
