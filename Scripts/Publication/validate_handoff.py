"""Read-only integrity, disclosure, and aggregate-result checks. No fitting."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = json.loads((ROOT/'Results/Publication/MANIFEST.json').read_text())
    for record in manifest['files']:
        p = ROOT/record['path']
        assert p.is_file(), p
        assert hashlib.sha256(p.read_bytes()).hexdigest() == record['sha256'], p
        assert p.stat().st_size == record['bytes'], p
    included = {x['path'] for x in manifest['files']} | {'Results/Publication/MANIFEST.json'}
    managed = ['Algorithms/GaussianCopula','Algorithms/iVAE','Algorithms/Native15_5','Algorithms/PosteriorGibbs','Utilities/GaussianCopula','Scripts/Publication','Results/Publication']
    actual = {p.relative_to(ROOT).as_posix() for name in managed for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert actual == included, ('Unexpected files', actual-included)
    for name in sorted(actual):
        p = ROOT/name
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        assert p.stat().st_size < 5_000_000, p
        assert p.suffix not in ('.ipynb', '.pt', '.pkl', '.zip'), p
        if p.suffix=='.mat': assert 'Native15_5/frozen/truth/' in name, p
        assert p.name not in ('data.csv', 'Z_embeddings.csv', 'training_evidence.npz', 'main_draft.pdf'), p
        if p.suffix in ('.py','.m','.csv','.json','.txt','.md'):
            text=p.read_text()
            assert not re.search(r'/(?:Users|home)/[^\s"\n]+',text),p
            assert not re.search(r'(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_]{15,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',text),p
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
    factor=pd.read_csv(ROOT/'Results/Publication/data/ivae/all_standardized_loadings.csv')
    matrices=np.load(ROOT/'Results/Publication/data/ivae/iVAE_plot_matrices_10_1.npz',allow_pickle=False)
    assert matrices['encoder'].shape==matrices['decoder'].shape==(98,10)
    selected=np.loadtxt(ROOT/'Results/Publication/data/factor/c6/STANDARDIZED_POINT_LOADINGS.csv',delimiter=',')
    assert np.allclose(factor[[f'Gaussian_{k}' for k in range(1,33)]],selected)
    summaries=[]
    for i in range(1,7):
        scores=pd.read_csv(ROOT/f'Results/Publication/data/factor/c{i}/PMSE_REPLICATES.csv')
        summary=json.loads((ROOT/f'Results/Publication/data/factor/c{i}/COMPLETE.json').read_text())
        assert len(scores)==50
        assert np.isclose(scores.pMSE.mean(),summary['pMSE_mean'],rtol=1e-12)
        summaries.append(summary)
    assert min(summaries,key=lambda x:x['pMSE_mean'])['id']==6
    deep=pd.read_csv(ROOT/'Results/Publication/data/deep/DEEP_REFIT_DIMENSION_RECOVERY.csv')
    assert len(deep)==6 and (deep.variant=='nofloor').all()
    assert deep.successful_fits.sum()==599 and deep.initialization_failed.sum()==1
    seeds=pd.read_csv(ROOT/'Scripts/Publication/config/paired_simulation_seeds.csv')
    assert len(seeds)==1800 and not seeds.duplicated(['J','n','replication']).any()
    pending=json.loads((ROOT/'Scripts/Publication/config/native15_5_pending.json').read_text())
    assert pending['status']=='COMPLETE_VALIDATED' and not pending['partial_results_included']
    native=pd.read_csv(ROOT/'Results/Publication/data/native15_5/MSE_SUMMARY.csv')
    assert len(native)==108 and (native.attempted==100).all()
    assert len(list((ROOT/'Results/Publication/figures/native15_5').glob('*.pdf')))==12
    maprows=list(csv.DictReader((ROOT/'Results/Publication/FIGURE_TABLE_MAP.csv').open()))
    for row in maprows:
        if row['asset']:assert (ROOT/row['asset']).is_file(), row
        for path in row['saved_inputs'].split('; '):assert (ROOT/path).is_file(),row
    print(json.dumps(dict(status='passed',files=len(manifest['files']),figure_table_items=len(maprows),factor_candidates=6,factor_scores=300,ivae_shape=[98,10],deep_completed=599,deep_initialization_failures=1,simulation_seed_rows=1800,experiments_started=False,scope='Saved-result validation; no end-to-end fitting or convergence claim')))


if __name__=='__main__':main()
