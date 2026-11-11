#!/usr/bin/env python3
"""Remeasure only unfiltered native ANN -> hard filter -> identical SELECT.

Keep original full-scan results immutable; store an independently measured overlay.
"""
import argparse
from pathlib import Path
from statistics import fmean
from time import perf_counter_ns

from pymilvus import MilvusClient
from run_final_violas_bench import DATASETS, DBS, ROOT, load, read, soft, write
from run_caltech_filter_order import fetch, select_fetched

OUT = ROOT / 'postfilter_ann'


def run(dataset):
    baseline = read(ROOT / dataset / 'results.json')
    previous, config, source, cache, directory = load(dataset)
    corpus = soft.Corpus(cache, directory)
    bindings = soft.bench.choose_queries(previous['rows'], source, set(corpus.ids))
    out = OUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'checkpoint.json'
    rows = read(checkpoint)['rows'] if checkpoint.exists() else []
    qdrant, qinfo = soft.setup_qdrant(config, corpus)
    milvus = MilvusClient(str(Path(f'cache/five_system/{dataset}_milvus_array.db').resolve()))
    try:
        milvus.load_collection(f'{dataset}_array')
        index = milvus.describe_index(f'{dataset}_array', 'vector')
        if 'IVF_FLAT' not in str(index):
            raise ValueError(f'Unexpected index {index}')
        _, chroma = soft.backends.open_chroma(Path(f'cache/five_system/{dataset}_chroma'), f'{dataset}_chroma')
        clients = dict(milvus=milvus, qdrant=qdrant, chroma=chroma)
        for number, original in enumerate(baseline['rows']):
            if number < len(rows):
                assert rows[number]['id'] == original['id']
                assert rows[number]['resolved_plan'] == original['resolved_plan']
                continue
            plan = original['resolved_plan']
            bound = bindings[original['id']]
            assert bound['object_id'] == original['query_object']['object_id']
            query = bound['normalized']
            methods = {}
            offset = number % len(DBS)
            for db in DBS[offset:] + DBS[:offset]:
                start = perf_counter_ns()
                # pre=False sends NO entity or facet predicate to the native index.
                fetched = fetch(db, clients[db], config, corpus, query, plan, pre=False)
                fetched_at = perf_counter_ns()
                ids, scores, stats = select_fetched(fetched, corpus, query, plan, pre=False)
                stopped = perf_counter_ns()
                ms = (stopped - start) / 1e6
                if len(fetched) != 10 or len({oid for oid, _ in fetched}) != 10:
                    raise ValueError(f'Expected ten distinct candidates: {dataset}/{original["id"]}/{db}')
                stats.update(candidate_ids=[oid for oid, _ in fetched],
                             fetch_ms=(fetched_at-start)/1e6, select_ms=(stopped-fetched_at)/1e6)
                oracle_ids = [r['object_id'] for r in original['oracle']['results']]
                result = soft.result(ids, scores, stats, ms, oracle_ids, original['oracle']['objective'],
                                     plan, corpus, original['eligible_count'])
                assert result['metrics']['hard_condition_validity'] == 1
                assert result['metrics']['objective_ratio'] <= 1 + 2e-6
                result['samples'] = [{'ms':ms, 'metrics':result['metrics'], 'stats':stats}]
                methods[f'post_{db}'] = result
            rows.append({**{k:v for k,v in original.items() if k != 'methods'}, 'methods':methods})
            if (number+1) % 10 == 0:
                write(checkpoint, {'rows':rows})
                print(f'PROGRESS ANN postfilter {dataset} {number+1}/100', flush=True)
        aggregate = {}
        for db in DBS:
            method = f'post_{db}'
            entries = [r['methods'][method] for r in rows]
            aggregate[method] = {k:fmean(e['metrics'][k] for e in entries) for k in entries[0]['metrics']}
            aggregate[method]['latency'] = soft.bench.latency([e['samples'][0]['ms'] for e in entries])
            aggregate[method]['scored_objects'] = fmean(e['stats']['unique_member_scores'] for e in entries)
        write(out / 'results.json', {'dataset':dataset, 'rows':rows, 'aggregate':aggregate,
            'data':baseline['data'], 'config':{**vars(config), 'repeats':1, 'milvus_index':index,
            'qdrant':qinfo, 'postfilter':'unfiltered native IVF/HNSW top10 -> hard filter -> same SELECT',
            'workload_sha256':baseline['config']['workload_sha256'], 'active_warmup':False}})
        print(f'COMPLETE ANN postfilter {dataset}', flush=True)
    finally:
        milvus.close()
        qdrant.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    run(parser.parse_args().dataset)
