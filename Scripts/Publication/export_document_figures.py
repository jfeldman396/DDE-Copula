"""Export packaged graphics under the relative names used by the supplied LaTeX."""
from pathlib import Path
import argparse,csv,shutil,hashlib
from paths import output
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();out=output(a.output_dir)
for r in csv.DictReader((ROOT/'Results/Publication/DOCUMENT_ASSET_MAP.csv').open()):
 src=ROOT/r['repository_asset'];assert hashlib.sha256(src.read_bytes()).hexdigest()==r['sha256']
 dst=out/r['document'].removesuffix('.tex')/r['requested_asset'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
print(out)
