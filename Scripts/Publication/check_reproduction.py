"""Compare complete fresh studies with publication references; never substitute fits.

Pending and failed components prevent an overall pass. Runtime measurements are
checked for completeness but are not expected to equal historical wall times.
The separate variational-interval pilot is excluded at the author's request.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import traceback
import numpy as np
import pandas as pd
from verify_from_scratch import jobs_for

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT/'Results/Publication/data'
STUDIES = ['ordinal_selection', 'ordinal', 'gaussian', 'b5', 'empirical',
           'uncertainty', 'shallow', 'comparison', 'deep']


def numbers(actual, expected):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape, (a.shape, b.shape)
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True)


def frames(actual, expected, keys, columns=None):
    a, b = actual.sort_values(keys).reset_index(drop=True), expected.sort_values(keys).reset_index(drop=True)
    assert not a.duplicated(keys).any() and not b.duplicated(keys).any()
    assert len(a) == len(b), (len(a), len(b))
    for key in keys:
        assert a[key].tolist() == b[key].tolist(), key
    for col in columns or list(b.columns):
        if col in keys:
            continue
        assert col in a, col
        if pd.api.types.is_numeric_dtype(b[col]):
            numbers(a[col], b[col])
        else:
            assert a[col].fillna('').tolist() == b[col].fillna('').tolist(), col


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--ivae-dir', type=Path, required=True)
    parser.add_argument('--survey-data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    assert out != ROOT and ROOT not in out.parents and not out.exists()
    out.mkdir(parents=True)
    jobs = {}
    for run in args.run_dir:
        plan = json.loads((run/'PLAN.json').read_text())
        for job in plan['jobs']:
            assert job['id'] not in jobs, 'Duplicate case in supplied runs'
            receipt = run/'status'/f"{job['id']}.json"
            status = json.loads(receipt.read_text()) if receipt.exists() else {'status': 'PENDING'}
            jobs[job['id']] = dict(job, directory=run/'runs'/job['id'], receipt=status)
    expected = jobs_for(STUDIES)
    assert not (jobs.keys()-{j['id'] for j in expected}), 'Unexpected case IDs'
    report = dict(tolerance=dict(relative=1e-10, absolute=1e-12), components={},
                  excluded='Reviewer1 variational-interval pilot (author request)',
                  interpretation='Numerical reproduction is separate from convergence or statistical validity.')

    def run(script, *arguments):
        command = [sys.executable, str(ROOT/'Scripts/Publication'/script), *map(str, arguments)]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(out/'plot_cache'))
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    def component(name, kinds, action):
        wanted = [j for j in expected if j['kind'] in kinds]
        pending = [j['id'] for j in wanted if j['id'] not in jobs or jobs[j['id']]['receipt']['status'] != 'COMPLETED_AWAITING_NUMERICAL_COMPARISON']
        if pending:
            report['components'][name] = dict(status='PENDING', required_cases=len(wanted), incomplete_cases=len(pending), first_incomplete=pending[:12])
            return
        try:
            detail = action()
            report['components'][name] = dict(status='PASS', cases=len(wanted), detail=detail)
        except Exception:
            report['components'][name] = dict(status='FAIL', error=traceback.format_exc())
        (out/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')

    def view(name, kinds):
        destination = out/(name+'_inputs');destination.mkdir()
        for job in jobs.values():
            if job['kind'] in kinds:
                (destination/job['id']).symlink_to(job['directory'].resolve(), target_is_directory=True)
        return destination

    def ordinal():
        destination = out/'ordinal'
        run('collect_ordinal.py', '--input-dir', view('ordinal', {'ordinal'}), '--output-dir', destination)
        for filename, keys in [('ordinal_tail_probabilities.csv', ['family','model','side','cutoff']),
                               ('ordinal_joint_examples.csv', ['family','model','category1','category2'])]:
            frames(pd.read_csv(destination/filename), pd.read_csv(REFERENCE/'ordinal'/filename), keys)
        return 'All96 tail-probability rows and300 joint-probability rows match; figures regenerated.'

    def selection():
        results = []
        references=json.loads((REFERENCE/'ordinal/ORDINAL_PROTOCOL.json').read_text())['selection_reference']
        for job in jobs.values():
            if job['kind'] == 'ordinal_selection':
                row = json.loads((job['directory']/'SELECTION_CHECK.json').read_text())
                assert row['matches_reported_settings'] and not row['test_used_for_selection']
                assert row['ineligible_D3_screen'] == 10
                reference=next(r for r in references if r['family']==row['family'])
                for key in ['candidate','validation_pmse','test_pmse']:numbers(row[key],reference[key])
                results.append(row)
        return results

    def synthetic(study):
        kinds = {'shallow'} if study=='shallow' else ({'deep'} if study=='deep' else {'shallow','comparator'})
        destination = out/study
        # pathlib.rglob does not follow directory links; link individual endpoints.
        inputs = out/(study+'_endpoints');inputs.mkdir()
        for job in jobs.values():
            if job['kind'] in kinds:
                case = inputs/job['id'];case.mkdir()
                (case/'endpoint.mat').symlink_to((job['directory']/'endpoint.mat').resolve())
        run('collect_revision_endpoints.py', '--study', study, '--input-dir', inputs, '--output-dir', destination)
        recovery = pd.read_csv(destination/'DIMENSION_RECOVERY.csv')
        if study=='shallow':
            target = pd.read_csv(REFERENCE/'dde/MAIN_DIMENSION_RECOVERY.csv')
            frames(recovery, target, ['J','n','arm'], ['completed','K1_correct_pct','K2_correct_pct','both_correct_pct'])
            timing = pd.read_csv(destination/'J100_MAP_RUNTIME_REPLICATES.csv')
            assert len(timing)==600 and (timing.numerical_wall>0).all()
            return 'All54 recovery rows match;600 new J100 timings recorded.'
        summary = pd.read_csv(destination/'MSE_SUMMARY.csv')
        raw = pd.read_csv(destination/'REPLICATE_MSE_UNTRIMMED_ARCHIVE.csv')
        if study=='comparison':
            target = pd.read_csv(REFERENCE/'native15_5/MSE_SUMMARY.csv')
            columns = [c for c in target if c in summary]
            frames(summary, target, ['method','J','n','layer'], columns)
            target = pd.read_csv(REFERENCE/'native15_5/REPLICATE_MSE_UNTRIMMED_ARCHIVE.csv')
            frames(raw, target, ['method','J','n','replication','layer'], ['eligible','MSE','retained'])
            a=np.load(destination/'AVERAGE_WEIGHT_MATRICES.npz');b=np.load(REFERENCE/'native15_5/AVERAGE_WEIGHT_MATRICES.npz')
            assert set(a.files)==set(b.files)
            for key in b.files:numbers(a[key],b[key])
            run('render_native15_5.py','--data-dir',destination,'--output',out/'comparison_figures')
            return 'All108 MSE rows,10800 replication errors/exclusions and222 matrix arrays match.'
        target = pd.read_csv(REFERENCE/'deep/DEEP_REFIT_DIMENSION_RECOVERY.csv').rename(columns={
            'attempts_resolved':'attempts','joint_correct_percent':'both_correct_pct',
            **{f'K{k}_correct_percent':f'K{k}_correct_pct' for k in [1,2,3]}})
        frames(recovery,target,['n'],['attempts','successful_fits','initialization_failed','both_correct_pct','K1_correct_pct','K2_correct_pct','K3_correct_pct'])
        target = pd.read_csv(REFERENCE/'deep/DEEP_REFIT_MSE_SUMMARY.csv').rename(columns={'trimmed_mean':'MSE','untrimmed_mean':'untrimmed_MSE'})
        frames(summary,target,['n','layer'],['eligible','retained','MSE','untrimmed_MSE','upper_cutoff'])
        target = pd.read_csv(REFERENCE/'deep/DEEP_REFIT_MSE_EXCLUSIONS.csv')
        eligible=raw[raw.eligible].copy();eligible['excluded']=~eligible.retained
        frames(eligible,target,['n','layer','replication'],['MSE','excluded'])
        return 'All600 outcomes, recovery/MSE tables and eligible replication errors/exclusions match.'

    def gaussian():
        base=jobs['gaussian']['directory']/'stage8k'
        reference=json.loads((REFERENCE/'factor/CANDIDATES.json').read_text())['candidates']
        for record in reference:
            folder=base/f"c{record['id']}"
            frames(pd.read_csv(folder/'PMSE_REPLICATES.csv'),pd.DataFrame(record['scores']).astype(float),['replicate'],['pMSE','AUC','Brier'])
            numbers(pd.read_csv(folder/'STANDARDIZED_POINT_LOADINGS.csv',header=None),record['loadings'])
            diagnostics=json.loads((folder/'DIAGNOSTICS.json').read_text())
            for key,value in record['diagnostics'].items():
                if isinstance(value,(int,float)):numbers(diagnostics[key],value)
        return 'All6 candidate loading matrices,300 scores and recorded diagnostics match.'

    def uncertainty():
        destination=out/'uncertainty'
        # Collector recursively locates dataset and chain names in the actual run tree.
        inputs=out/'uncertainty_inputs';inputs.mkdir()
        for job in jobs.values():
            if job['kind']!='uncertainty':continue
            for kind in ['datasets','chains']:
                for p in (job['directory']/kind).iterdir():
                    d=inputs/p.name;d.mkdir()
                    for file in p.iterdir():
                        if file.is_file():(d/file.name).symlink_to(file.resolve())
        run('collect_uncertainty.py','--datasets-dir',inputs,'--chains-dir',inputs,'--output-dir',destination)
        for filename,keys in [('COVERAGE_SUMMARY.csv',['J','n','stratum']),('DATASET_DIAGNOSTICS.csv',['dataset','target']),('ACTIVE_DIMENSION_PROBABILITIES.csv',['dataset','K1','K2'])]:
            frames(pd.read_csv(destination/filename),pd.read_csv(REFERENCE/'uncertainty'/filename),keys)
        return 'All40 datasets,120 diagnostic rows, dimension occupancies and12 enclosure summaries match.'

    def b5():
        base=jobs['b5']['directory'];destination=out/'b5'
        run('collect_b5_posterior.py','--checkpoint',base/'selected/fit/RAW_CHECKPOINT.mat','--chains-dir',base/'chains','--output-dir',destination)
        for layer in ['shallow','deep']:
            filename=f'original_four_pooled_{layer}.csv'
            frames(pd.read_csv(destination/filename),pd.read_csv(REFERENCE/'b5'/filename),['row','column'])
        return 'All980 shallow and20 deep posterior summaries match; both plots regenerated.'

    def empirical():
        base=jobs['empirical_maxima_9']['directory'];destination=out/'empirical'
        reference=json.loads((REFERENCE/'empirical/SELECTION.json').read_text())
        for candidate in reference['grid']:
            expected=candidate['summary'];folder=jobs[f"empirical_{expected['sampler']}_{expected['config_id']}"]['directory']
            actual=json.loads((folder/'COMPLETE.json').read_text())
            assert actual['status']==expected['status']
            numbers(actual['widths'],[expected['K1'],expected['K2']])
            for key in ['iterations','CM_flags']:numbers(actual[key],expected[key])
            numbers(actual['pMSE_mean'],expected['pMSE']);numbers(actual['pMSE_SD'],expected['pMSE_SD'])
            frames(pd.read_csv(folder/'PMSE_REPLICATES.csv'),pd.DataFrame(candidate['scores']),['replicate'],['pMSE','AUC','Brier'])
        run('collect_empirical.py','--model',base/'fit/MODEL_it0740_target0800.mat','--data',args.survey_data,
            '--dde-scores',base/'PMSE_REPLICATES.csv','--gaussian-dir',jobs['gaussian']['directory']/'stage8k','--output-dir',destination)
        for filename,keys in [('HIGHER_ORDER_VALUES.csv',['factor','rank','upper_state']),('HIGHER_ORDER_TOP_ANCHOR_VALUES.csv',['factor','rank','upper_state']),('PAIRED_PMSE.csv',['replicate']),('SURVEY_ITEM_RESPONSE_COUNTS.csv',['item','response'])]:
            frames(pd.read_csv(destination/filename),pd.read_csv(REFERENCE/'empirical'/filename),keys)
        frames(pd.read_csv(destination/'all_standardized_loadings.csv'),pd.read_csv(REFERENCE/'ivae/all_standardized_loadings.csv'),['item_id'])
        run('render_factor_loadings.py','--loadings',destination/'all_standardized_loadings.csv','--output-dir',out/'factor_figures')
        run('render_ivae.py','--weights-dir',args.ivae_dir,'--dde-normalized',destination/'NORMALIZED_POINT.npz','--loadings',destination/'all_standardized_loadings.csv','--output-dir',out/'ivae_figures')
        assert json.loads((out/'ivae_figures/key_items.json').read_text())==json.loads((REFERENCE/'ivae/key_items.json').read_text())
        return 'All24 grid outcomes and1200 scores, survey profiles, response counts, paired scores, loadings and all30 key-item entries match.'

    component('ordinal_selection',{'ordinal_selection'},selection)
    component('ordinal',{'ordinal'},ordinal)
    component('shallow',{'shallow'},lambda:synthetic('shallow'))
    component('comparison',{'shallow','comparator'},lambda:synthetic('comparison'))
    component('deep',{'deep'},lambda:synthetic('deep'))
    component('gaussian',{'gaussian'},gaussian)
    component('uncertainty',{'uncertainty'},uncertainty)
    component('b5',{'b5'},b5)
    component('empirical',{'empirical','gaussian'},empirical)
    try:
        weights=json.loads((REFERENCE/'ivae/NETWORK_WEIGHTS.json').read_text())
        for key,value in weights.items():numbers(pd.read_csv(args.ivae_dir/f'{key}.csv'),value['values'])
        numbers(np.load(args.ivae_dir/'training_elbo.npy'),pd.read_csv(REFERENCE/'ivae/TRAINING_ELBO.csv').ELBO)
        report['components']['ivae_training']=dict(status='PASS',weight_arrays=8,epochs=334)
    except Exception:
        report['components']['ivae_training']=dict(status='FAIL',error=traceback.format_exc())
    report['all_included_numerical_results_verified']=all(r['status']=='PASS' for r in report['components'].values())
    (out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value['status'] for key,value in report['components'].items()},indent=2))
    if not report['all_included_numerical_results_verified']:sys.exit(1)


if __name__=='__main__':main()
