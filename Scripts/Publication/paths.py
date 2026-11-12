"""Configuration paths are relative to the config file, never to the working directory."""
from pathlib import Path
import json
PACKAGE = Path(__file__).resolve().parents[2]
def configure(parser):
    parser.add_argument('--config', type=Path, default=PACKAGE/'Scripts/Publication/config.json')
    return parser

def inputs(config):
    config = config.resolve()
    settings = json.loads(config.read_text())
    result = {k: (config.parent/v).resolve() for k, v in settings.items()}
    for key, path in result.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Configured {key} directory is missing: {path}")
    return result

def output(path):
    path = Path(path).resolve()
    for protected in ('Results/Publication', 'Algorithms', 'Utilities', 'Scripts', 'v2'):
        root = PACKAGE/protected
        if path == root or root in path.parents:
            raise ValueError(f"Output must not overwrite packaged inputs: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Use a new or empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path
