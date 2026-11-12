"""Summarize audited current deep fits; keep eligibility and trimming transparent."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

def summarize(folder):
    audit=json.loads((folder/'CURRENT_RUN_AUDIT.json').read_text())
    if audit['errors']: raise ValueError('Resolve endpoint audit errors before summarizing')
    data=pd.read_csv(folder/'DEEP_CURRENT_REPLICATES.csv')
    assert len(data)==audit['studies']['deep']['valid']
    recovery=[];mse=[];exclusions=[]
    for (n,arm),g in data.groupby(['n','arm']):
        row=dict(n=int(n),arm=int(arm),completed=len(g),target=100)
        for layer,truth in enumerate([18,6,2],1):
            k=g[f'K{layer}'];row.update({f'K{layer}_correct':int((k==truth).sum()),f'K{layer}_under':int((k<truth).sum()),f'K{layer}_over':int((k>truth).sum()),f'K{layer}_correct_percent':100*(k==truth).mean()})
        row['joint_correct']=int(g.correct_all.sum());row['joint_correct_percent']=100*g.correct_all.mean();recovery.append(row)
        eligible=g[g.correct_all]
        for layer in [1,2,3]:
            values=eligible[f'MSE{layer}'];q1,q3=values.quantile([.25,.75]) if len(values) else (np.nan,np.nan)
            cutoff=q3+1.5*(q3-q1);keep=values<=cutoff
            mse.append(dict(n=int(n),arm=int(arm),layer=layer,completed=len(g),eligible=len(values),excluded=int((~keep).sum()),retained=int(keep.sum()),upper_cutoff=cutoff,untrimmed_mean=values.mean(),trimmed_mean=values[keep].mean()))
            for idx,value in values.items():
                exclusions.append(dict(n=int(n),arm=int(arm),layer=layer,replication=int(g.loc[idx,'replication']),MSE=value,upper_cutoff=cutoff,excluded=bool(value>cutoff)))
    pd.DataFrame(recovery).to_csv(folder/'DEEP_CURRENT_DIMENSION_RECOVERY.csv',index=False)
    pd.DataFrame(mse).to_csv(folder/'DEEP_CURRENT_MSE_SUMMARY.csv',index=False)
    pd.DataFrame(exclusions).to_csv(folder/'DEEP_CURRENT_MSE_EXCLUSIONS.csv',index=False)
    (folder/'DEEP_SUMMARY_STATUS.json').write_text(json.dumps(dict(source_checked_at=audit['checked_at'],complete=audit['studies']['deep']['complete'],notice='Monitoring only until all 1800 endpoints pass audit. Recovery uses all completed fits. MSE conditions on all three true widths; upper Q3+1.5 IQR calculated separately by n/arm/layer. Raw MSEs remain in DEEP_CURRENT_REPLICATES.csv. No superseded fits included.'),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=Path(__file__).resolve().parents[3]/'output/manuscript_update_20261103');summarize(p.parse_args().folder)
