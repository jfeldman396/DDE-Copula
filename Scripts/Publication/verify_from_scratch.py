"""Run the published full fitting budgets in isolated, restartable case folders.

No existing research fits are read. Completion is separate from numerical
comparison, and no shortened run can be marked complete by this driver.
"""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ['clayton', 'survival_clayton', 'student_t', 'gaussian']


def source_digest():
    files = sorted(p for folder in ['Algorithms', 'Utilities', 'Scripts']
                   for p in (ROOT / folder).rglob('*')
                   if p.is_file() and p.suffix in {'.m', '.py', '.json', '.mat', '.csv'})
    entries = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in files}
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(), entries


def jobs_for(studies):
    jobs = []
    def add(kind, name, J=0, n=0, rep=0, method='', family=''):
        jobs.append(dict(kind=kind, id=name, J=J, n=n, rep=rep, method=method, family=family))
    if 'ordinal_selection' in studies:
        for i, family in enumerate(FAMILIES, 1):
            add('ordinal_selection', f'selection_{family}', rep=i, family=family)
    if 'ordinal' in studies:
        for family in FAMILIES:
            for method in ['gibbs', 'maxima']:
                add('ordinal', f'{family}_{method}', method=method, family=family)
    if 'gaussian' in studies:
        add('gaussian', 'gaussian')
    if 'b5' in studies:
        add('b5', 'b5')
    if 'empirical' in studies:
        for method in ['gibbs', 'maxima']:
            for index in range(12):
                add('empirical', f'empirical_{method}_{index}', rep=index, method=method)
    if 'uncertainty' in studies:
        for rep in range(1, 11):
            for J in [50, 100]:
                for n in [1000, 2000]:
                    add('uncertainty', f'uncertainty_J{J}_n{n}_r{rep}', J, n, rep)
    # Interleave methods and sizes, so early output exercises all configurations.
    for rep in range(1, 101):
        for n in [500, 1000, 2000, 4000, 8000, 16000]:
            for J in [50, 100, 150]:
                if 'shallow' in studies:
                    for method in ['map', 'vi', 'tempering']:
                        add('shallow', f'shallow_J{J}_n{n}_r{rep}_{method}', J, n, rep, method)
                if 'comparison' in studies:
                    for method in ['rl', 'poisson']:
                        add('comparator', f'comparator_J{J}_n{n}_r{rep}_{method}', J, n, rep, method)
            if 'deep' in studies:
                add('deep', f'deep_J108_n{n}_r{rep}', 108, n, rep, 'map')
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--survey-data', type=Path)
    p.add_argument('--matlab', default='matlab')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--studies', nargs='+', default=['all'], choices=[
        'all', 'ordinal_selection', 'ordinal', 'gaussian', 'b5', 'empirical',
        'uncertainty', 'shallow', 'comparison', 'deep'])
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    if not 1 <= args.workers <= 16:
        p.error('--workers must be between 1 and 16')
    out = args.output_dir.resolve()
    if out == ROOT or ROOT in out.parents:
        p.error('Use an output directory outside the checkout')
    if out.exists() and not args.resume:
        p.error('Output already exists; choose a new directory or explicitly --resume')
    studies = args.studies
    if 'all' in studies:
        studies = ['ordinal_selection', 'ordinal', 'gaussian', 'b5', 'empirical',
                   'uncertainty', 'shallow', 'comparison', 'deep']
    needs_survey = bool(set(studies) & {'gaussian', 'b5', 'empirical'})
    if needs_survey and not args.survey_data:
        p.error('These studies require --survey-data with the author-provided original CSV')
    digest, entries = source_digest()
    planfile = out / 'PLAN.json'
    if args.resume:
        plan = json.loads(planfile.read_text())
        if plan['source_sha256'] != digest or plan['studies'] != studies:
            p.error('Source or study list changed; use a new run to preserve provenance')
        # Failed and interrupted cases require explicit diagnosis; never overwrite.
        stale = [x for x in (out/'claims').iterdir()
                 if not (out/'status'/f'{x.name}.json').exists()
                 or json.loads((out/'status'/f'{x.name}.json').read_text())['status'] in ['RUNNING', 'FAILED']]
        if stale:
            p.error(f'{len(stale)} failed/interrupted cases need diagnosis; their files are retained')
    else:
        out.mkdir(parents=True)
        for folder in ['claims', 'status', 'runs', 'logs']:
            (out/folder).mkdir()
        survey_input = ''
        if needs_survey:
            subprocess.run([sys.executable, str(ROOT/'Scripts/Publication/prepare_survey.py'),
                            '--data', str(args.survey_data.resolve()), '--output-dir', str(out/'survey')], check=True)
            survey_input = str(out/'survey/train_input.mat')
        plan = dict(source_sha256=digest, source_files=entries, studies=studies,
                    output=str(out), survey_input=survey_input, jobs=jobs_for(studies),
                    inference='Full-budget execution does not establish convergence or numerical parity')
        planfile.write_text(json.dumps(plan, indent=2)+'\n')
    print(f"Planned {len(plan['jobs'])} complete cases; results: {out}", flush=True)
    if args.plan_only:
        return
    executable = shutil.which(args.matlab)
    if not executable:
        p.error('MATLAB executable not found; pass its path with --matlab')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    escaped = str(planfile).replace("'", "''")
    command = f"restoredefaultpath; addpath('Scripts/Publication'); run_verification_worker('{escaped}');"
    def worker(i):
        with (out/'logs'/f'worker_{i}.log').open('a') as log:
            return subprocess.run([executable, '-singleCompThread', '-batch', command],
                                  cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(worker, range(1, args.workers+1)))
    records = [json.loads(x.read_text()) for x in (out/'status').glob('*.json')]
    completed = sum(x['status']=='COMPLETED_AWAITING_NUMERICAL_COMPARISON' for x in records)
    report = dict(planned=len(plan['jobs']), completed=completed,
                  failed=sum(x['status']=='FAILED' for x in records), worker_exit_codes=codes,
                  all_fits_executed=completed==len(plan['jobs']), numerical_comparison_passed=False)
    (out/'EXECUTION_REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    if not report['all_fits_executed'] or any(codes):
        sys.exit(1)


if __name__ == '__main__':
    main()
