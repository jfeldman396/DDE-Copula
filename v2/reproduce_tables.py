"""Check frozen compact summaries and regenerate the primary publication table.
Standard library only; no fitting and no access to original study directories.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, statistics, collections
ROOT = Path(__file__).resolve().parent

def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))

def verify():
    repo = ROOT.parent
    for filename in ['SOURCE_MANIFEST.json', 'EVIDENCE_MANIFEST.json']:
        manifest = json.loads((ROOT / 'provenance' / filename).read_text())
        for entry in manifest['files']:
            payload = (repo / entry['path']).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == entry['sha256'], entry['path']
            assert len(payload) == entry['bytes'], entry['path']
    reference = read(ROOT / 'evidence' / 'primary_all_settings.csv')
    summaries = read(ROOT / 'evidence' / 'COMMON_SUMMARY.csv')
    fields = list(reference[0])
    rows = []
    for s in summaries:
        count = int(s['datasets'])
        r = dict(J=s['J'], n=s['n'], arm_id=s['arm_id'], method=s['scheme'], setting=s['setting'], N=count,
                 joint_correct=round(float(s['joint_correct'])*count), conditional_N=int(s['conditional_count']))
        for layer in ['B1','B2']:
            r[f'{layer}_MSE_all'] = s[f'{layer}_all_MSE']
            r[f'{layer}_MSE_conditional'] = s[f'{layer}_conditional_MSE']
        r.update(median_CPU_s=s['median_CPU'], cap_count=round(float(s['cap_rate'])*count),
                 stabilized_count=round(float(s['stabilized_rate'])*count),
                 optimizer_flagged_fits=int(s['optimizer_flagged_fits']), failed=int(s['failed_fit_count']))
        rows.append(r)
    key = lambda r: tuple(int(r[k]) for k in ['J','n','arm_id'])
    expected = {key(r): r for r in reference}
    assert len(rows) == len(expected) == 216
    assert {key(r) for r in rows} == {(j,n,a) for j in [50,100,150] for n in [500,1000,2000,4000,8000,16000] for a in range(1,13)}
    for row in rows:
        assert row['N'] == 100
        for field in fields:
            a, b = str(row[field]), expected[key(row)][field]
            if a == b:
                continue
            assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (key(row),field,a,b)
    assert sum(r['N'] for r in rows) == 21600
    assert sum(r['optimizer_flagged_fits'] for r in rows) == 45
    assert sum(r['failed'] for r in rows) == 0
    print('PASS: source/evidence hashes; 216 cells; 21,600 outcomes; compact-summary reconciliation; 45 flagged outcomes retained.')
    return fields, rows

def summarize_endpoints(source, destination):
    rows = read(source)
    key = lambda r: tuple(int(r[k]) for k in ['J','n','replication','arm_id'])
    expected = {(j,n,r,a) for j in [50,100,150] for n in [500,1000,2000,4000,8000,16000] for r in range(1,101) for a in range(1,13)}
    assert len(rows) == 21600 and {key(r) for r in rows} == expected, 'Require all 21,600 distinct endpoints; no omissions.'
    yes = lambda r,k: r[k].lower() in ['1','true']
    groups = collections.defaultdict(list)
    for r in rows:
        assert all(math.isfinite(float(r[k])) for k in ['B1_MSE','B2_MSE','CPU'])
        if 'failed' in r: assert not yes(r,'failed'), 'Failed outcome requires explicit analysis, not exclusion.'
        groups[(int(r['J']),int(r['n']),int(r['arm_id']))].append(r)
    summary = []
    for (j,n,a),g in sorted(groups.items()):
        selected = [r for r in g if yes(r,'joint_correct')]
        row = dict(J=j,n=n,arm_id=a,method=g[0].get('method',g[0].get('scheme')),setting=g[0]['setting'],N=len(g),joint_correct=len(selected),conditional_N=len(selected))
        for layer in ['B1','B2']:
            metric = layer+'_MSE'
            row[metric+'_all'] = statistics.mean(float(r[metric]) for r in g)
            row[metric+'_conditional'] = statistics.mean(float(r[metric]) for r in selected) if selected else ''
        row.update(median_CPU_s=statistics.median(float(r['CPU']) for r in g),cap_count=sum(yes(r,'cap_reached') for r in g),stabilized_count=sum(yes(r,'stabilized') for r in g),optimizer_flagged_fits=sum(float(r.get('optimizer_flags',r.get('flags')))>0 for r in g),failed=0)
        summary.append(row)
    matched=[]
    for j in [50,100,150]:
        for n in [500,1000,2000,4000,8000,16000]:
            sets=[{int(r['replication']) for r in groups[j,n,a] if yes(r,'joint_correct')} for a in [1,5,9]]
            eligible=set.intersection(*sets)
            for a,method in [(1,'H1'),(5,'H0'),(9,'Z')]:
                selected=[r for r in groups[j,n,a] if int(r['replication']) in eligible]
                row=dict(scope='per_n',J=j,n=n,method=method,N=len(selected))
                for metric in ['B1_MSE','B2_MSE']:
                    row[metric]=statistics.mean(float(r[metric]) for r in selected) if selected else ''
                matched.append(row)
    destination.mkdir(parents=True,exist_ok=False)
    def write(name, records):
        with (destination/name).open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write('primary_all_settings.csv',summary)
    write('primary_tempering_CP.csv',[r for r in summary if r['setting']=='temp_CP'])
    write('matched_three_methods.csv',matched)
    print('PASS: complete endpoint grid; all outcomes retained; primary and per-n matched figure tables written.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new CSV destination for frozen summary reconciliation.')
    parser.add_argument('--endpoints', type=Path, help='Complete fresh endpoint CSV from summarize_v2 (21,600 rows).')
    parser.add_argument('--outdir', type=Path, help='New directory for primary and matched tables from endpoints.')
    args = parser.parse_args()
    if args.endpoints:
        if not args.outdir or args.output: parser.error('--endpoints requires --outdir and forbids --output')
        summarize_endpoints(args.endpoints,args.outdir)
    else:
        if args.outdir: parser.error('--outdir requires --endpoints')
        fields, rows = verify()
        if args.output:
            with args.output.open('x', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
