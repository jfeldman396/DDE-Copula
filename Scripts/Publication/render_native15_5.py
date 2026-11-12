"""Re-render completed native15/5 figures from packaged numerical summaries."""
from pathlib import Path
import argparse,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dde_publication_mpl')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from paths import output
ROOT=Path(__file__).resolve().parents[2]
W=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--data-dir',type=Path,default=ROOT/'Results/Publication/data/native15_5');args=parser.parse_args()
O=output(args.output);(O/'images').mkdir()
D=args.data_dir.resolve()
s=pd.read_csv(D/'MSE_SUMMARY.csv');means=np.load(D/'AVERAGE_WEIGHT_MATRICES.npz')
truths={(j,l):means[f'truth_J{j}_B{l}'] for j in [50,100,150] for l in [1,2]}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42})
ns=[500,1000,2000,4000,8000,16000];styles=[('map','MAP: tempering + VI','o'),('rl','Truncated-lasso RL','s'),('poisson','Poisson DDE','^')]
cmap=LinearSegmentedColormap.from_list('parula_reference',json.loads((W/'parula_reference.json').read_text()),N=256)
visual_audit=[];figure_sources=[]
for J in [50,100,150]:
 for l in [1,2]:
  f,ax=plt.subplots(figsize=(7,5.3));f.subplots_adjust(left=.14,right=.97,top=.90,bottom=.29)
  ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks(ns,[str(n) for n in ns]);ax.set_xlabel('Sample size (n)');ax.set_ylabel('Entry-wise MSE (log scale)');ax.set_title(rf'$J={J}$: $B^{{({l})}}$ entry-wise MSE');ax.grid(alpha=.22)
  for method,label,marker in styles:
   g=s.query('method==@method and J==@J and layer==@l').sort_values('n');assert len(g)==6 and (g.MSE>0).all();ax.plot(g.n,g.MSE,marker=marker,mfc='none',lw=1.6,label=label)
   figure_sources.append(dict(figure=f'J{J}_B{l}_main_MSE.pdf',method=method,layer=l,J=J,source='MSE_SUMMARY.csv',sample_sizes=g.n.astype(int).tolist(),plotted_values=g.MSE.tolist(),retained_counts=g.retained.astype(int).tolist(),eligibility_and_exclusions='ELIGIBILITY_AND_EXCLUSIONS.json',ordinate='MSE on logarithmic y axis; values are not log-transformed before plotting'))
  ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),frameon=False,fontsize=10)
  for ext in ['pdf','png']:f.savefig(O/'images'/f'J{J}_B{l}_main_MSE.{ext}',dpi=180)
  plt.close(f)
  truth=truths[J,l];vmax=float(np.max(np.abs(truth)));nr,nc=truth.shape
  f,axs=plt.subplots(3,4,figsize=(16,8));f.subplots_adjust(left=.10,right=.91,top=.87,bottom=.10,wspace=.18,hspace=.38)
  for ri,(method,label,_) in enumerate(styles):
   for ci,n in enumerate([1000,4000,16000,None]):
    b=truth if n is None else means[f'{method}_J{J}_n{n}_B{l}_retained'];ax=axs[ri,ci]
    im=ax.imshow(b,aspect='auto',origin='upper',extent=(-.5,nc-.5,nr+.5,.5),cmap=cmap,vmin=-vmax,vmax=vmax,interpolation='nearest')
    ax.set_title('Truth' if n is None else f'n={n}',fontsize=15,fontweight='bold');ax.set_xticks([0,5,10,15] if l==1 else list(range(6)));ax.set_yticks([1,nr//2,nr]);ax.tick_params(direction='in',top=True,right=True)
    if ci==0:ax.set_ylabel(label,fontsize=11)
    if ri==2:ax.set_xlabel('Factor (0 = intercept)',fontsize=11)
    visual_audit.append(dict(J=J,layer=l,method=method,n=n,rows=nr,columns_including_intercept=nc,below_color_limit=int((b < -vmax).sum()),above_color_limit=int((b>vmax).sum())))
    figure_sources.append(dict(figure=f'J{J}_B{l}_est_updated.pdf',method=method,layer=l,J=J,n=n,source='AVERAGE_WEIGHT_MATRICES.npz',array_key=f'truth_J{J}_B{l}' if n is None else f'{method}_J{J}_n{n}_B{l}_retained',panel_row=ri+1,panel_column=ci+1,shape=list(b.shape),color_limits=[-vmax,vmax]))
  cax=f.add_axes([.935,.18,.012,.61])
  cb=f.colorbar(im,cax=cax,extend='both');cb.set_label('Coefficient (saturated beyond limits)',fontsize=10)
  f.suptitle(rf'$J={J}$: $B^{{({l})}}$ estimates and truth (15/5 initialization)',fontsize=16,y=.97)
  for ext in ['pdf','png']:f.savefig(O/'images'/f'J{J}_B{l}_est_updated.{ext}',dpi=180)
  plt.close(f)
(O/'FIGURE_RENDER_AUDIT.json').write_text(json.dumps(visual_audit,indent=2)+'\n')
(O/'FIGURE_SOURCES.json').write_text(json.dumps(figure_sources,indent=2)+'\n')
