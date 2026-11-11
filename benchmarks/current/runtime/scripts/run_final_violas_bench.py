#!/usr/bin/env python3
"""Frozen six-dataset benchmark: native VG, prefilter, exact postfilter, filtered ANN."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np
from pymilvus import MilvusClient
from qdrant_client import models

import run_soft_entity_bench as soft
from run_caltech_filter_order import fetch, select_fetched
from label_caltech_local_mvp import write

ROOT = Path('artifacts/violas_final')
DATASETS = ('caltech101', 'cub200', 'flowers102', '20news', 'ohsumed', 'yahoo')
DBS = ('milvus', 'qdrant', 'chroma')
METHODS = ['vg_hdmg', 'vg_rep'] + [f'{mode}_{db}' for mode in ('pre', 'post', 'index') for db in DBS]


def read(path):
    from artifact_paths import rewrite
    return rewrite(json.loads(Path(path).read_text(encoding='utf-8')))


def load(dataset):
    previous = read(f'artifacts/six_dataset_soft_entity_online_cosine/{dataset}/results.json')
    config = SimpleNamespace(**previous['config'])
    config.index_tag = dataset
    config.candidate_k = 10
    config.group_budget = 8
    config.nav_budget = 12
    source, cache, directory = [soft.bench.load_pickle(Path(p), config.module_path) for p in (
        config.source_cache, config.hdmg_cache, config.directory_cache)]
    return previous, config, source, cache, directory


def fetch_all(db, client, config):
    tag = config.index_tag
    if db == 'milvus':
        iterator = client.query_iterator(f'{tag}_array', filter='', batch_size=1000,
                                         output_fields=['object_id', 'vector'])
        rows = []
        try:
            while True:
                page = iterator.next()
                if not page:
                    break
                rows.extend((r['object_id'], r['vector']) for r in page)
        finally:
            iterator.close()
        return rows
    if db == 'qdrant':
        rows, offset = [], None
        while True:
            page, offset = client.scroll(f'soft_{tag}', offset=offset, limit=1000,
                                         with_vectors=True, with_payload=['object_id'])
            rows.extend((p.payload['object_id'], p.vector) for p in page)
            if offset is None:
                return rows
    response = client.get(include=['embeddings'])
    return list(zip(response['ids'], response['embeddings'], strict=True))


def fetch_index(db, client, config, query, plan):
    tag, entity = config.index_tag, plan['required_entity']
    rules = plan['hard_filters']
    if db == 'milvus':
        clauses = [f'entity == {json.dumps(entity)}']
        if rules:
            tokens = [f"{r['key']}={r['value']}" for r in rules]
            clauses.append(f'ARRAY_CONTAINS_ALL(facet_tokens, {json.dumps(tokens)})')
        response = client.search(f'{tag}_array', data=[query.tolist()], anns_field='vector',
                                 filter=' and '.join(clauses), limit=10, output_fields=['object_id', 'vector'],
                                 search_params={'metric_type': 'COSINE', 'params': {'nprobe': config.nprobe}})
        return [(h['entity']['object_id'], h['entity']['vector']) for h in response[0]]
    if db == 'qdrant':
        conditions = [models.FieldCondition(key='entity', match=models.MatchValue(value=entity))]
        conditions.extend(models.FieldCondition(key='facet_tokens', match=models.MatchValue(
            value=f"{r['key']}={r['value']}")) for r in rules)
        response = client.query_points(f'soft_{tag}', query=query.tolist(), limit=10,
                                       query_filter=models.Filter(must=conditions), with_vectors=True,
                                       with_payload=['object_id'], search_params=models.SearchParams(hnsw_ef=config.ef))
        return [(p.payload['object_id'], p.vector) for p in response.points]
    clauses = [{'entity': {'$eq': entity}}]
    clauses.extend({f"facet__{r['key']}": {'$eq': r['value']}} for r in rules)
    where = clauses[0] if len(clauses) == 1 else {'$and': clauses}
    response = client.query(query_embeddings=[query.tolist()], n_results=10, where=where, include=['embeddings'])
    return list(zip(response['ids'][0], response['embeddings'][0], strict=True))


def run(dataset, repeats, limit):
    previous, config, source, cache, directory = load(dataset)
    corpus = soft.Corpus(cache, directory)
    records = previous['rows'][:limit]
    bindings = soft.bench.choose_queries(previous['rows'], source, set(corpus.ids))
    out = ROOT / dataset
    out.mkdir(parents=True, exist_ok=True)
    plans = [{**r['resolved_plan'], 'required_entity': r['entity_hint'], 'entity_mode': 'hard',
              'search_scope': 'required_entity'} for r in records]
    workload = [{'id': r['id'], 'query_object': r['query_object'], 'intent': p} for r, p in zip(records, plans)]
    write(out / 'workload.json', workload)
    qdrant, qinfo = soft.setup_qdrant(config, corpus)
    milvus = MilvusClient(str(Path(f'cache/five_system/{dataset}_milvus_array.db').resolve()))
    results = []
    try:
        milvus.load_collection(f'{dataset}_array')
        index_info = milvus.describe_index(f'{dataset}_array', 'vector')
        if 'IVF_FLAT' not in json.dumps(index_info):
            raise ValueError(f'actual Milvus index is not IVF_FLAT: {index_info}')
        _, chroma = soft.backends.open_chroma(Path(f'cache/five_system/{dataset}_chroma'), f'{dataset}_chroma')
        clients = {'milvus': milvus, 'qdrant': qdrant, 'chroma': chroma}
        samples = [{name: [] for name in METHODS} for _ in records]
        checkpoint = out / 'checkpoint.json'
        if checkpoint.exists():
            results = read(checkpoint)['rows']
            for number, entry in enumerate(results):
                if entry['id'] != records[number]['id'] or entry['resolved_plan'] != plans[number]:
                    raise ValueError('checkpoint workload differs')
                for method in METHODS:
                    samples[number][method] = entry['methods'][method]['samples']
                    if len(samples[number][method]) > repeats:
                        raise ValueError('checkpoint has more rounds than requested')
        for repeat in range(repeats):
            for number, (record, plan) in enumerate(zip(records, plans)):
                if all(len(samples[number][method]) > repeat for method in METHODS):
                    continue
                bound = bindings[record['id']]
                if bound['object_id'] != record['query_object']['object_id']:
                    raise ValueError('frozen query changed')
                query = bound['normalized']
                raw = np.asarray(source['test_data'][record['entity_hint']]['vectors'][bound['test_index']],
                                 dtype=np.float32).reshape(-1)
                offset = (number + repeat) % len(METHODS)
                outputs = {}
                for method in METHODS[offset:] + METHODS[:offset]:
                    start = perf_counter_ns()
                    candidate_ids = None
                    if method == 'vg_hdmg':
                        ids, scores, stats = soft.group_search(corpus, query, plan, config, True)
                    elif method == 'vg_rep':
                        ids, scores, stats = soft.representative_scan(corpus, raw, plan, 8, online_cosine=True)
                    else:
                        mode, db = method.split('_')
                        if mode == 'pre':
                            fetched = fetch(db, clients[db], config, corpus, query, plan, True)
                        elif mode == 'index':
                            fetched = fetch_index(db, clients[db], config, query, plan)
                        else:
                            fetched = fetch_all(db, clients[db], config)
                            if len(fetched) != len(corpus.ids):
                                raise ValueError('full scan omitted objects')
                            matrix = np.asarray([v for _, v in fetched], dtype=np.float32)
                            distances = np.einsum('ij,j->i', matrix, query)
                            nearest = sorted(range(len(fetched)), key=lambda i: (-float(distances[i]), fetched[i][0]))[:10]
                            fetched = [fetched[i] for i in nearest]
                        fetch_stop = perf_counter_ns()
                        ids, scores, stats = select_fetched(fetched, corpus, query, plan, pre=mode == 'pre')
                        candidate_ids = [oid for oid, _ in fetched]
                    stop = perf_counter_ns()
                    ms = (stop - start) / 1e6
                    if candidate_ids is not None:
                        stats.update(candidate_ids=candidate_ids, fetch_ms=(fetch_stop-start)/1e6,
                                     select_ms=(stop-fetch_stop)/1e6)
                    outputs[method] = (ids, scores, stats, ms)
                eligible = [i for i in range(len(corpus.ids)) if corpus.eligible(i, plan)]
                oracle_scores = corpus.scores(eligible, query, plan, corpus.semantic_scores(plan))
                oracle_ids = soft.select(oracle_scores, corpus, plan)
                oracle_value = soft.core.objective(oracle_ids, oracle_scores, corpus.objects, plan['facet_dimension'],
                                                   plan['lambda'], plan['category_preferences'])
                if not oracle_ids:
                    raise ValueError('frozen workload has no feasible answer')
                methods = {}
                for method, (ids, scores, stats, ms) in outputs.items():
                    result = soft.result(ids, scores, stats, ms, oracle_ids, oracle_value, plan, corpus, len(eligible))
                    if result['metrics']['hard_condition_validity'] != 1:
                        raise ValueError(f'hard constraint failed: {dataset} {record["id"]} {method}')
                    if method.startswith('pre_'):
                        if scores.keys() != oracle_scores.keys() or abs(result['metrics']['objective_ratio']-1) > 2e-6:
                            raise ValueError(f'prefilter parity failed: {dataset} {record["id"]} {method}')
                        result['stats']['different_optimal_ids'] = set(ids) != set(oracle_ids)
                    samples[number][method].append({'ms': ms, 'metrics': result['metrics'], 'stats': stats})
                    result['samples'] = samples[number][method]
                    result['latency'] = soft.bench.latency([s['ms'] for s in result['samples']])
                    methods[method] = result
                entry = {k: record[k] for k in ('id', 'query_type', 'entity_hint', 'query_object')}
                entry.update(resolved_plan=plan, eligible_count=len(eligible), methods=methods,
                             oracle={'results': soft.core.result_rows(oracle_ids, oracle_scores, corpus.objects),
                                     'objective': oracle_value})
                if repeat == 0:
                    results.append(entry)
                else:
                    results[number] = entry
                if (number+1) % 10 == 0 or number+1 == len(records):
                    write(checkpoint, {'rounds':repeat, 'rows':results})
                    print(f'PROGRESS {dataset} round={repeat+1}/{repeats} query={number+1}/{len(records)}', flush=True)
            write(out / 'checkpoint.json', {'rounds': repeat+1, 'rows': results})
        aggregate = {}
        for method in METHODS:
            entries = [s for row in results for s in row['methods'][method]['samples']]
            aggregate[method] = {key: fmean(e['metrics'][key] for e in entries) for key in entries[0]['metrics']}
            aggregate[method]['latency'] = soft.bench.latency([s['ms'] for s in entries])
            aggregate[method]['scored_objects'] = fmean(s['stats']['unique_member_scores'] for s in entries)
        report = {'dataset': dataset, 'modality': previous['modality'], 'rows': results, 'aggregate': aggregate,
                  'config': {**vars(config), 'repeats': repeats, 'milvus_index': index_info, 'qdrant': qinfo,
                             'postfilter': 'exact full scan with vector materialization for each query',
                             'workload_sha256': hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()},
                  'data': {'objects': len(corpus.ids), 'entities': len(corpus.entities), 'dimensions': corpus.vectors.shape[1],
                           'clusters': len(corpus.members), 'queries': len(records), 'facet_manifest': cache.get('manifest', {})}}
        write(out / 'results.json', report)
        print(f'COMPLETE retrieval {dataset}', flush=True)
    finally:
        milvus.close()
        qdrant.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--output-root', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.output_root
    run(args.dataset, args.repeats, args.limit)
