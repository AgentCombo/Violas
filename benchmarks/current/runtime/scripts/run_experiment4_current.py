"""Replay legacy experiment 1 with current VG and all three key baselines."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter_ns
import numpy as np
from pymilvus import MilvusClient
from run_final_violas_bench import ROOT, DATASETS, load, read, write, soft
from adaptive_hdmg import AdaptiveHDMG

METHODS = ('vg_hdmg', 'vg_rep', 'milvus_key', 'qdrant_key', 'chroma_key')


def run(dataset):
    previous, config, source, old_cache, old_directory = load(dataset)
    config.candidate_k = 80
    config.key_budget = 3
    storage = read(ROOT/'maintenance_v6'/dataset/'vg_hdmg.json')
    saved = soft.bench.load_pickle(Path(storage['full_snapshot_path']), config.module_path)
    corpus = soft.Corpus(saved['cache'], saved['directory'])
    old = soft.Corpus(old_cache, old_directory)
    assert corpus.ids == old.ids
    assert corpus.entities == old.entities and corpus.facets == old.facets
    assert np.array_equal(corpus.entity_ids, old.entity_ids)
    assert np.allclose(corpus.keys, old.keys, atol=1e-7)
    assert np.allclose(corpus.vectors, old.vectors, atol=2e-6)
    engine = AdaptiveHDMG(corpus)
    bindings = soft.bench.choose_queries(previous['rows'], source, set(corpus.ids))
    records = previous['rows']
    assert len(records) == 100
    workload = [{'id': r['id'], 'query_object': r['query_object'], 'intent': r['resolved_plan']} for r in records]
    assert all(not r['intent'].get('required_entity') for r in workload)
    digest = hashlib.sha256(json.dumps(workload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    qdrant, qinfo = soft.setup_qdrant(config, corpus)
    milvus = MilvusClient(str(Path(f'cache/five_system/{dataset}_milvus_array.db').resolve()))
    rows = []
    try:
        milvus.load_collection(f'{dataset}_array')
        _, chroma = soft.backends.open_chroma(Path(f'cache/five_system/{dataset}_chroma'), f'{dataset}_chroma')
        clients = dict(milvus=milvus, qdrant=qdrant, chroma=chroma)
        for number, record in enumerate(records):
            bound = bindings[record['id']]
            assert bound['object_id'] == record['query_object']['object_id']
            query, plan = bound['normalized'], record['resolved_plan']
            raw = np.asarray(source['test_data'][record['entity_hint']]['vectors'][bound['test_index']], np.float32).reshape(-1)
            outputs = {}
            offset = number % len(METHODS)
            for name in METHODS[offset:]+METHODS[:offset]:
                start = perf_counter_ns()
                if name == 'vg_hdmg':
                    value = engine.search(query, plan, budget=32, score_batch=4)
                elif name == 'vg_rep':
                    value = soft.representative_scan(corpus, raw, plan, 32, online_cosine=True)
                else:
                    db = name.split('_')[0]
                    value = soft.db_search(db, 'key', clients[db], corpus, query, plan, config, False)
                outputs[name] = value, (perf_counter_ns()-start)/1e6
            # Compute the complete-domain reference outside timed execution.
            eligible = [i for i in range(len(corpus.ids)) if corpus.eligible(i, plan)]
            scores = corpus.scores(eligible, query, plan, corpus.semantic_scores(plan))
            oracle_ids = soft.select(scores, corpus, plan)
            objective = soft.core.objective(oracle_ids, scores, corpus.objects, plan['facet_dimension'], plan['lambda'], plan['category_preferences'])
            assert objective > 0
            assert np.isclose(objective, record['oracle']['objective'], rtol=2e-6, atol=2e-6), record['id']
            results = {}
            for name, ((ids, pool, stats), ms) in outputs.items():
                if name.startswith('vg_'):
                    assert all(corpus.eligible(corpus.locator[oid], plan) for oid in ids)
                else:
                    assert stats['candidate_count'] <= 80 and len(stats['routed_keys']) == 3
                    assert all(corpus.objects[oid]['description']['entity'] in stats['routed_keys'] for oid in ids)
                result = soft.result(ids, pool, stats, ms, oracle_ids, objective, plan, corpus, len(eligible))
                assert result['metrics']['objective_ratio'] <= 1+2e-6
                results[name] = result
            rows.append(dict(id=record['id'], query_object=record['query_object'], resolved_plan=plan,
                             oracle=dict(ids=oracle_ids, objective=objective, legacy_objective=record['oracle']['objective']), methods=results))
            if (number+1)%20 == 0:
                print(f'{dataset} {number+1}/100', flush=True)
        aggregate = {name: dict(objective_ratio=fmean(r['methods'][name]['metrics']['objective_ratio'] for r in rows),
                     latency=soft.bench.latency([r['methods'][name]['latency']['mean_ms'] for r in rows])) for name in METHODS}
        out = ROOT/'experiment4_current'/dataset
        out.mkdir(parents=True, exist_ok=True)
        write(out/'results.json', dict(dataset=dataset, rows=rows, aggregate=aggregate, config=dict(
            candidate_k=80, key_budget=3, vg_group_budget=32, score_batch=4, certify=False,
            repeats=1, queries=100, active_warmup=False, rotating_order=True,
            source_store=storage['full_snapshot_path'], source_workload=f'artifacts/six_dataset_soft_entity_online_cosine/{dataset}/results.json',
            workload_sha256=digest, source_config=vars(config), qdrant=qinfo,
            implementation_sha256={f:hashlib.sha256(Path('scripts',f).read_bytes()).hexdigest() for f in
                ('adaptive_hdmg.py','run_soft_entity_bench.py','run_experiment4_current.py')},
            validation='All gallery IDs, entity/facet metadata and vectors checked against legacy cache; all 100 Oracle objectives checked against frozen soft-entity reference.'
        )))
        print('COMPLETE '+dataset+' '+json.dumps(aggregate), flush=True)
    finally:
        milvus.close()
        qdrant.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS)
    args = parser.parse_args()
    for dataset in ([args.dataset] if args.dataset else DATASETS):
        run(dataset)
