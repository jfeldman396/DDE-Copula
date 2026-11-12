from pathlib import Path
import json,hashlib,shutil
import numpy as np,pandas as pd,h5py
from scipy.special import expit
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from paths import configure, inputs, output
parser=configure(argparse.ArgumentParser(description='Render saved standardized DDE/Gaussian loadings; no model fitting.'))
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--loadings',type=Path,help='Fresh combined DDE/Gaussian CSV; omit for saved loadings')
args=parser.parse_args();settings=inputs(args.config);O=output(args.output_dir)
frame=pd.read_csv(args.loadings or settings['ivae']/'all_standardized_loadings.csv')
D=frame[[f'DDE_{k}' for k in range(1,11)]].to_numpy()
G=frame[[f'Gaussian_{k}' for k in range(1,33)]].to_numpy()
cos=D.T@G/(np.linalg.norm(D,axis=0)[:,None]*np.linalg.norm(G,axis=0)[None,:]);ii,jj=linear_sum_assignment(-abs(cos));sgn=np.sign(cos[ii,jj]);M=G[:,jj]*sgn;un=np.array([k for k in range(32) if k not in jj]);un=un[np.argsort(-np.linalg.norm(G[:,un],axis=0))]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(f,n):
 f.savefig(O/(n+'.pdf'),bbox_inches='tight');f.savefig(O/(n+'.png'),dpi=300,bbox_inches='tight');plt.close(f)
group=np.argmax(abs(D),axis=1);order=np.lexsort((np.arange(98),-abs(D[np.arange(98),group]),group));lim=max(abs(D).max(),abs(G).max())
f,axs=plt.subplots(1,2,figsize=(5.5,7.4),gridspec_kw={'width_ratios':[1,1]},layout='constrained')
for ax,A,title,xt in zip(axs,[D[order],M[order]],['DDE Copula\n10 first-layer features','Gaussian copula\n10 matched columns'],[np.arange(1,11),jj+1]):
 im=ax.imshow(A,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest');ax.set_title(title,fontsize=10);ax.set_xticks(range(len(xt)),xt,rotation=90,fontsize=7);ax.set_xlabel('Feature / loading column');ax.set_yticks([])
 for k in range(1,98):
  if group[order[k]]!=group[order[k-1]]:ax.axhline(k-.5,color='#444444',lw=.5)
centers=[np.flatnonzero(group[order]==k).mean() for k in range(10) if np.any(group==k)];labels=[f'D{k+1}' for k in range(10) if np.any(group==k)]
axs[0].set_yticks(centers,labels);axs[0].set_ylabel('98 items grouped by largest absolute DDE loading')
f.colorbar(im,ax=axs,orientation='horizontal',fraction=.045,pad=.04,label='Standardized structural loading');save(f,'weight_matrix_comparison')
