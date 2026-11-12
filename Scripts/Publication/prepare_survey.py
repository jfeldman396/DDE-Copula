"""Prepare the original survey items for MATLAB; data are obtained from the author."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.io import savemat
from paths import output
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
expected=json.loads((ROOT/'Results/Publication/data/ivae/RUN_MANIFEST.json').read_text())['data_sha256']
assert hashlib.sha256(a.data.read_bytes()).hexdigest()==expected,'Request the original survey CSV from the author; input bytes do not match the study.'
f=pd.read_csv(a.data);columns=np.setdiff1d(np.arange(2,102),[45,52]);Y=f.iloc[:,columns].to_numpy(float);X=np.where(np.isnan(Y),3.,Y)
assert Y.shape==(1500,98) and np.isnan(Y).sum()==223 and np.isfinite(X).all()
# Portable identity independent of the timestamp/header in a .mat container.
payload=np.isnan(Y).astype('uint8').tobytes(order='F')+np.nan_to_num(Y,nan=0.).astype('<f8').tobytes(order='F')+X.astype('<f8').tobytes(order='F')
identity=hashlib.sha256(payload).hexdigest();cfg=json.loads((ROOT/'Scripts/Publication/config/factor.json').read_text());assert identity==cfg['prepared_values_sha256']
out=output(a.output_dir);savemat(out/'train_input.mat',dict(Y=Y,X=X,row_ids=np.arange(1,1501),source_csv_sha256=expected),do_compression=True)
(out/'PREPARATION.json').write_text(json.dumps(dict(source_csv_sha256=expected,prepared_values_sha256=identity,rows=1500,items=98,missing=223,imputation='MATLAB DDE/Gaussian: replace missing values with3; iVAE train.py separately uses item means',data_access='Contact the author for the original survey CSV'),indent=2)+'\n');print(out)
