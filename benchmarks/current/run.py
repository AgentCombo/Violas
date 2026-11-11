"""Explicit entry points for the measured research implementation.

Run from the repository root: python -m benchmarks.current.run --help
No benchmark is executed by importing this module.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import runpy
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE/'runtime'
REPO = HERE.parents[1]
DATASETS = ('caltech101','cub200','flowers102','20news','ohsumed','yahoo')
RECIPES = {
    'hard-baselines': 'run_final_violas_bench',
    'postfilter-ann': 'run_final_postfilter_ann',
    'conditional-vg': 'run_integrated_retrieval_v6',
    'yahoo-optimized': 'benchmark_yahoo_search',
    'maintenance': 'run_maintenance_extended_v6',
    'crud': 'run_single_crud',
    'embedding': 'run_pure_embedding_final',
    'soft-entity': 'run_experiment4_current',
}


def measurements():
    with gzip.open(HERE/'results/measurements.json.gz', 'rt', encoding='utf8') as f:
        return json.load(f)


def verify():
    manifest=json.loads((HERE/'packaged-manifest.json').read_text())
    for item in manifest['files']:
        path=HERE/item['path']
        assert hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n') if path.suffix not in ('.gz','.png','.pdf') else path.read_bytes()).hexdigest()==item['sha256'], str(path)
    checks=0
    for path, record in measurements().items():
        rows=record.get('rows', [])
        if not rows or 'aggregate' not in record:
            continue
        for name, aggregate in record['aggregate'].items():
            # Some stored records have repeated trials nested under a query.
            values=[r['methods'][name] for r in rows if name in r.get('methods',{})]
            if not values or not all('ms' in v for v in values):
                continue
            actual=statistics.fmean(v['ms'] for v in values)
            expected=aggregate.get('latency',{}).get('mean_ms')
            if expected is not None:
                assert abs(actual-expected)<0.00011,(path,name,actual,expected)
                checks+=1
            for metric in ['objective_ratio','recall_at_3']:
                if metric in aggregate and all(metric in v for v in values):
                    assert abs(statistics.fmean(v[metric] for v in values)-aggregate[metric])<1e-8,(path,name,metric)
                    checks+=1
    print(json.dumps(dict(payload_files=len(manifest['files']), aggregate_checks=checks, status='PASS')))


def bootstrap(workspace=None, path_map=None, historical=False):
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    sys.path[:0]=[str(RUNTIME/'scripts'),str(RUNTIME),str(REPO)]
    if path_map:
        path_map=Path(path_map).resolve()
        os.environ["VIOLAS_PATH_MAP_FILE"]=str(path_map)
    if workspace:
        os.chdir(Path(workspace).resolve())
    import run_final_violas_bench as bench
    import storage.vectormap as storage_module
    for module in (bench, bench.soft, storage_module):
        assert Path(module.__file__).resolve().is_relative_to(RUNTIME.resolve()), module.__file__
    original_load=bench.load
    def load(dataset):
        result=original_load(dataset)
        if os.environ.get('VIOLAS_QDRANT_URL'):
            result[1].qdrant_url=os.environ['VIOLAS_QDRANT_URL']
        return result
    bench.load=load
    if historical:
        import adaptive_hdmg_before_yahoo
        sys.modules['adaptive_hdmg']=adaptive_hdmg_before_yahoo
    return bench


def check_data(args):
    bench=bootstrap(args.workspace,args.path_map)
    import numpy as np
    from adaptive_hdmg import AdaptiveHDMG
    total=0
    for dataset in ([args.dataset] if args.dataset else DATASETS):
        previous,config,source,_,_=bench.load(dataset)
        stored=bench.read(bench.ROOT/'maintenance_v6'/dataset/'vg_hdmg.json')
        saved=bench.soft.bench.load_pickle(Path(stored['full_snapshot_path']),config.module_path)
        corpus=bench.soft.Corpus(saved['cache'],saved['directory'])
        frozen=bench.read(bench.ROOT/dataset/'results.json')
        expected=bench.read(bench.ROOT/'integrated_v6'/dataset/'results.json')
        expected_rows={r['id']:r for r in expected['rows']}
        binding=bench.soft.bench.choose_queries(previous['rows'],source,set(corpus.ids))
        engine=AdaptiveHDMG(corpus)
        for row in frozen['rows']:
            plan=row['resolved_plan']
            ids,scores,_=engine.search(binding[row['id']]['normalized'],plan,budget=32,score_batch=4)
            assert all(corpus.eligible(corpus.locator[oid],plan) for oid in ids)
            value=bench.soft.core.objective(ids,scores,corpus.objects,plan['facet_dimension'],plan['lambda'],plan['category_preferences'])
            old=expected_rows[row['id']]['methods']['vg_hdmg']
            assert np.isclose(value,old['objective'],rtol=0,atol=2e-6),(dataset,row['id'],value,old['objective'])
            total+=1
        print('VERIFIED',dataset,len(frozen['rows']),'frozen objectives',flush=True)
    print('PASS',total,'queries; no timing results or reports written')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('verify',help='Verify source hashes and recorded aggregate arithmetic (stdlib only).')
    sub.add_parser('demo',help='Execute the actual selector/HDMG on synthetic data, checked by exhaustive enumeration.')
    sub.add_parser('test',help='Check selector, graph construction and incremental topology.')
    for command in ['check-data','run']:
        s=sub.add_parser(command)
        s.add_argument('--workspace',required=True,type=Path,help='Prepared benchmark workspace; run writes results into it.')
        s.add_argument('--path-map',type=Path,help='JSON map of original absolute path prefixes to local directories.')
        s.add_argument('--dataset',choices=DATASETS,required=(command=='run'))
        if command=='run':
            s.add_argument('--experiment',choices=RECIPES,required=True)
            s.add_argument('--method',choices=['vg_hdmg','milvus','qdrant','chroma'],default='vg_hdmg')
    args=p.parse_args()
    if args.command=='verify':return verify()
    if args.command in ['demo','test']:
        for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
            os.environ[name]='1'
        sys.path.insert(0,str(REPO))
        scripts=['test_adaptive_hdmg'] if args.command=='demo' else ['test_adaptive_hdmg','test_fast_hdmg','test_hot_hdmg_v6','test_core_store','test_artifact_paths']
        for name in scripts:
            runpy.run_path(str(RUNTIME/'scripts'/f'{name}.py'),run_name='__main__')
        return
    if args.command=='check-data':return check_data(args)
    if not args.workspace.is_dir():p.error('workspace does not exist; see benchmarks/current/README.md')
    # The original conditional table predates Yahoo's optimized engine.
    historical=args.experiment=='conditional-vg'
    bootstrap(args.workspace,args.path_map,historical=historical)
    if args.experiment=='embedding':
        source=REPO/'violas/retrieval/embedding_hdmg_native.cpp'
        subprocess.run(['g++','-O3','-march=native','-fopenmp-simd','-shared','-fPIC',str(source),'-o',str(source.with_name('libembedding_hdmg_native.so'))],check=True)
    script=RUNTIME/'scripts'/f'{RECIPES[args.experiment]}.py'
    sys.argv=[str(script)]
    if args.experiment!='yahoo-optimized':sys.argv+=['--dataset',args.dataset]
    elif args.dataset!='yahoo':p.error('yahoo-optimized only supports yahoo')
    if args.experiment=='maintenance':sys.argv+=['--method',args.method]
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':
    main()
