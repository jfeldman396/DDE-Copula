"""Run all packaged saved-result workflows; no fitting or external data required."""
from pathlib import Path
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'Scripts/Publication'
DATA = ROOT / 'Results/Publication/data'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New directory for regenerated plots, exports and the result index')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    for folder in ('Algorithms', 'Utilities', 'Scripts', 'Results'):
        protected = ROOT/folder
        if out == protected or protected in out.parents:
            parser.error('Output must be outside the packaged source/results folders')
    if out.exists():
        parser.error('Choose a new output directory; existing output is not overwritten')
    env = os.environ.copy()
    env.setdefault('MPLCONFIGDIR', '/tmp/dde_reproduction_mpl')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run([sys.executable, str(SCRIPTS/'validate_handoff.py')], check=True, env=env)
    out.mkdir(parents=True)
    jobs = [
        ('render_native15_5.py', '--output', 'native15_5'),
        ('reproduce_summary_results.py', '--output-dir', 'summaries'),
        ('render_factor_loadings.py', '--output-dir', 'factor'),
        ('render_ivae.py', '--output-dir', 'ivae'),
        ('export_document_figures.py', '--output-dir', 'document_figures'),
    ]
    completed = []
    for script, flag, folder in jobs:
        print(f'Running {script}', flush=True)
        subprocess.run([sys.executable, str(SCRIPTS/script), flag, str(out/folder)],
                       check=True, cwd=ROOT, env=env)
        completed.append(script)
    shutil.copytree(DATA, out/'numeric_inputs')
    # Make consolidated candidate results directly usable as tables.
    candidates = json.loads((DATA/'factor/CANDIDATES.json').read_text())['candidates']
    tables = {
        'gaussian_candidate_summary.csv': [c['summary'] for c in candidates],
        'gaussian_candidate_scores.csv': [dict(candidate=c['id'], **row)
                                         for c in candidates for row in c['scores']],
    }
    for name, rows in tables.items():
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (out/'summaries'/name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    # Record the exact distinction between rerendered plots and preserved artwork.
    index = []
    for row in csv.DictReader((ROOT/'Results/Publication/DOCUMENT_ASSET_MAP.csv').open()):
        source = row['repository_asset']; name = Path(source).name
        generated = ''
        if '/native15_5/' in source:
            generated = f'native15_5/images/{name}'
        elif name in ('B1_MSE.pdf','B2_MSE.pdf','B3_MSE.pdf'):
            generated = f'summaries/{name}'
        elif name == 'weight_matrix_comparison.pdf':
            generated = f'factor/{name}'
        elif name == 'iVAE_comparison_DDE_oriented.pdf':
            generated = f'ivae/images/{name}'
        truth = re.match(r'J(50|100|150)_B([12])_true', name, re.IGNORECASE)
        if truth:
            generated = f'summaries/J{truth[1]}_B{truth[2]}_truth.pdf'
        if generated:
            assert (out/generated).is_file(), generated
        saved = f"document_figures/{row['document'].removesuffix('.tex')}/{row['requested_asset']}"
        assert (out/saved).is_file(), saved
        index.append(dict(document=row['document'], result=row['requested_asset'],
                          status='rendered_from_saved_numbers' if generated else 'preserved_artwork_only',
                          regenerated_output=generated, saved_output=saved,
                          numerical_inputs='numeric_inputs/; see Scripts/REPRODUCE.txt'))
    for row in csv.DictReader((ROOT/'Results/Publication/FIGURE_TABLE_MAP.csv').open()):
        if row['type'] != 'table':
            continue
        inputs = [str(Path('numeric_inputs')/Path(s).relative_to('Results/Publication/data'))
                  for s in row['saved_inputs'].split('; ')]
        assert all((out/p).is_file() for p in inputs)
        document = 'template_v2.tex' if row['labels'] in ('tab:recovery_resultsJ100','tab:keyitems','fig:b5_qs') else 'supplementary_materials_v2.tex'
        index.append(dict(document=document, result=row['labels'] or 'iVAE loading table',
                          status='saved_numerical_table_inputs', regenerated_output='', saved_output='',
                          numerical_inputs='; '.join(inputs)))
    with (out/'RESULT_INDEX.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index[0]))
        writer.writeheader(); writer.writerows(index)
    report = dict(status='completed', experiments_started=False, completed_scripts=completed,
                  document_figure_references=39, index_entries=len(index),
                  scope='Saved numerical results and preserved artwork; not full model refits',
                  limitations='Survey data are available from the author. Fresh-fit routes and their validation limits are in Scripts/REPRODUCE.txt; this command only uses saved results.')
    (out/'REPRODUCTION_REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Completed. Open {out / "RESULT_INDEX.csv"}')


if __name__ == '__main__':
    main()
