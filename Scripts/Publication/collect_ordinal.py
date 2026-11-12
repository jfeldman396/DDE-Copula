"""Recompute all ordinal probabilities and both figures from eight new fits."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import output
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
OUT=output(args.output_dir);F=OUT
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
families=['clayton','survival_clayton','student_t','gaussian'];titles=['Clayton','Survival Clayton','Student t','Gaussian'];models=['Original','rank_Gibbs','group_maxima'];labels=['Original','DDE: Gibbs','DDE: maxima'];colors=['#202020','#0072B2','#D55E00'];markers=['o','s','^']
def save(fig,name):
 fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.png'),dpi=200,bbox_inches='tight');plt.close(fig)
rows=[];joint=[]
for fam in families:
 datasets={}
 for method,model in [('gibbs','rank_Gibbs'),('maxima','group_maxima')]:
  with h5py.File(args.input_dir/f'{fam}_{method}'/'result.mat') as h:
   assert bool(h['result/complete'][()].item()),'Incomplete ordinal fit'
   observed=h['result/data/Ytest'][:].T
   if 'Original' in datasets:assert np.array_equal(datasets['Original'],observed)
   datasets['Original']=observed;datasets[model]=h['result/pred/Y'][:].T
 for model in models:
  y=datasets[model]; assert np.isfinite(y).all() and (y>=0).all()
  z=np.digitize(y,[1,3,5,7],right=True)+1
  n,J=z.shape;ii,jj=np.triu_indices(J,1);assert J==30
  for k in [1,2,3,4]:
   for side in ['lower','upper']:
    a=(z<=k) if side=='lower' else (z>k)
    # Independent vectorized cross-product and rowwise combinatorial calculations agree.
    pairprob=(a.astype(float).T@a/n)[ii,jj].mean()
    independent=np.mean(a.sum(1)*(a.sum(1)-1)/(J*(J-1)))
    assert abs(pairprob-independent)<1e-12
    rows.append(dict(family=fam,model=model,side=side,cutoff=k,probability=pairprob,n=n,pairs=len(ii),marginal_event_rate=a.mean()))
  for a in range(1,6):
   for b in range(1,6):joint.append(dict(family=fam,model=model,category1=a,category2=b,probability=np.mean((z[:,0]==a)&(z[:,1]==b)),n=n))
o=pd.DataFrame(rows);o.to_csv(OUT/'ordinal_tail_probabilities.csv',index=False);j=pd.DataFrame(joint);j.to_csv(OUT/'ordinal_joint_examples.csv',index=False)
fig,axs=plt.subplots(4,2,figsize=(7.2,8.6),sharex=True,sharey=True)
for r,(fam,title) in enumerate(zip(families,titles)):
 for c,side in enumerate(['lower','upper']):
  ax=axs[r,c]
  for mod,lab,col,m in zip(models,labels,colors,markers):
   d=o[(o.family==fam)&(o.side==side)&(o.model==mod)]
   ax.plot(d.cutoff,d.probability,label=lab,color=col,marker=m,ms=4)
  ax.set_title(title+'\n'+('both categories ≤ k' if side=='lower' else 'both categories > k'));ax.grid(alpha=.16);ax.set_xticks([1,2,3,4]);ax.set_ylim(0,1)
  if c==0:ax.set_ylabel('Joint event probability')
  if r==3:ax.set_xlabel('Category threshold k')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.005));fig.tight_layout(rect=[0,.03,1,1]);save(fig,'R1_2_ordinal_tail_probabilities')
fig,axs=plt.subplots(4,3,figsize=(7.2,8.8),sharex=True,sharey=True,layout='constrained'); vmax=j.probability.max()
for r,(fam,title) in enumerate(zip(families,titles)):
 for c,(mod,lab) in enumerate(zip(models,labels)):
  d=j[(j.family==fam)&(j.model==mod)].pivot(index='category2',columns='category1',values='probability')
  assert abs(d.to_numpy().sum()-1)<1e-12
  im=axs[r,c].imshow(d,origin='lower',extent=[.5,5.5,.5,5.5],vmin=0,vmax=vmax,cmap='Blues');axs[r,c].set_xticks(range(1,6));axs[r,c].set_yticks(range(1,6));axs[r,c].set_title(lab if r==0 else '')
  if c==0:axs[r,c].set_ylabel(title+'\nVariable 2 category')
  if r==3:axs[r,c].set_xlabel('Variable 1 category')
fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.6,label='Joint cell probability');save(fig,'R1_2_ordinal_synthetic_examples')
# Paper-facing two-column panels use the reported group-maxima DDE fit.
# The three-method audit panels above retain the matched Gibbs comparison.
paper_models=['Original','group_maxima'];paper_labels=['Original','DDE Copula']
fig,axs=plt.subplots(4,2,figsize=(5.6,8.6),sharex=True,sharey=True,layout='constrained')
for r,(fam,title) in enumerate(zip(families,titles)):
 for c,(model,label) in enumerate(zip(paper_models,paper_labels)):
  d=j[(j.family==fam)&(j.model==model)].pivot(index='category2',columns='category1',values='probability')
  im=axs[r,c].imshow(d,origin='lower',extent=[.5,5.5,.5,5.5],vmin=0,vmax=vmax,cmap='Blues')
  axs[r,c].set_xticks(range(1,6));axs[r,c].set_yticks(range(1,6))
  if r==0:axs[r,c].set_title(label)
  if c==0:axs[r,c].set_ylabel(title+'\nVariable 2 category')
  if r==3:axs[r,c].set_xlabel('Variable 1 category')
fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.6,label='Joint cell probability');save(fig,'ordinal_synthetic_vs_truth')
fig,axs=plt.subplots(4,2,figsize=(6.5,8.6),sharex=True,sharey=True)
for r,(fam,title) in enumerate(zip(families,titles)):
 for c,side in enumerate(['lower','upper']):
  ax=axs[r,c]
  for model,label,color,marker in zip(paper_models,paper_labels,['#202020','#0072B2'],['o','s']):
   d=o[(o.family==fam)&(o.side==side)&(o.model==model)]
   ax.plot(d.cutoff,d.probability,label=label,color=color,marker=marker,ms=4)
  ax.set_title(title+'\n'+('both categories ≤ k' if side=='lower' else 'both categories > k'));ax.grid(alpha=.16);ax.set_xticks([1,2,3,4]);ax.set_ylim(0,1)
  if c==0:ax.set_ylabel('Joint event probability')
  if r==3:ax.set_xlabel('Category threshold k')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False);fig.tight_layout(rect=[0,.035,1,1]);save(fig,'ordinal_tail_probabilities')
print(OUT)
