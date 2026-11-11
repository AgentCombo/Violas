"""Matched single-object CRUD across VG and three database backends."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from time import perf_counter_ns
import numpy as np
from run_final_violas_bench import ROOT, load, read, write, soft
from run_maintenance_extended_v6 import Database
from incremental_vg_v6 import IncrementalVGStore
from test_storage_v6 import check


def run(dataset):
    previous, config, source, cache, _ = load(dataset)
    objects = soft.bench.retrieval.make_object_map(cache)
    rng = np.random.default_rng(42)
    groups = defaultdict(list)
    for oid in sorted(objects):
        groups[objects[oid]['description']['entity']].append(oid)
    base, held = [], []
    for entity, ids in sorted(groups.items()):
        ids = list(rng.permutation(ids))
        count = max(1, round(.1*len(ids)))
        held.extend(ids[:count])
        base.extend(ids[count:])
    operation_ids = list(np.random.default_rng(73).choice(held,60,replace=False))
    out = ROOT/'single_crud'/dataset
    out.mkdir(parents=True, exist_ok=True)
    names = ('vg_hdmg','milvus','qdrant','chroma')
    stores = {}
    samples = {name:{op:[] for op in ('C','R','U','D')} for name in names}
    frozen = read(ROOT/dataset/'results.json')['rows'][::5]
    bindings = soft.bench.choose_queries(previous['rows'],source,set(objects))
    active = set(base)

    def get(name, ids):
        store = stores[name]
        if name != 'vg_hdmg':
            return store.get(ids)
        return {oid:(store.active[oid]['normalized'].copy(),
                     [f'{k}={v}' for k,v in store.active[oid]['description']['facet_values'].items()])
                for oid in ids if oid in store.active}

    def verify_records(name, ids):
        rows = get(name, ids)
        assert set(rows) == set(ids)
        for oid,(vector,tokens) in rows.items():
            assert np.allclose(vector,objects[oid]['normalized'],atol=2e-6)
            assert set(tokens) == {f'{k}={v}' for k,v in objects[oid]['description']['facet_values'].items()}

    try:
        for name in names:
            if name == 'vg_hdmg':
                import time
                store = IncrementalVGStore(cache,objects,out/f'base_{time.time_ns()}.pkl',True)
            else:
                store = Database(name,config,objects,out)
                store.create()
            stores[name] = store
            store.insert(base)
            if name != 'vg_hdmg':
                store.ready()
            print(f'READY {dataset} {name}',flush=True)
        for trial in range(3):
            ids = operation_ids[trial*20:(trial+1)*20]
            for op in ('C','R','U','D'):
                for number,oid in enumerate(ids):
                    if op == 'U':
                        obj=objects[oid]
                        vector=np.asarray(obj['normalized'],np.float32)
                        vector=(.9*vector+.1*np.roll(vector,1)).astype(np.float32)
                        norm=float(np.linalg.norm(vector))
                        if norm==0:
                            vector[0]=1.
                        else:
                            vector/=norm
                        assert np.isfinite(vector).all()
                        obj['normalized']=vector
                        obj['vector']=vector.copy()
                        obj['description']=copy.deepcopy(obj['description'])
                        obj['description']['facet_values']['maintenance_revision']='updated'
                    shift = (number+trial)%len(names)
                    for name in names[shift:]+names[:shift]:
                        start = perf_counter_ns()
                        if op == 'C':
                            stores[name].insert([oid])
                        elif op == 'D':
                            stores[name].delete([oid])
                        elif op == 'U':
                            stores[name].update([oid])
                        else:
                            get(name,[oid])
                        samples[name][op].append(dict(trial=trial,id=oid,ms=(perf_counter_ns()-start)/1e6))
                if op == 'C':
                    active.update(ids)
                elif op == 'D':
                    active.difference_update(ids)
                for name in names:
                    stores[name].verify(active,ids,op=='D')
                    if op != 'D':
                        verify_records(name,ids)
                    if name != 'vg_hdmg':
                        stores[name].ready()
                check(stores['vg_hdmg'],frozen,bindings)
            print(f'COMPLETE {dataset} trial {trial+1}',flush=True)
        stores['vg_hdmg'].recover()
        stores['vg_hdmg'].verify(active,operation_ids,True)
        check(stores['vg_hdmg'],frozen,bindings)
        aggregate={name:{op:soft.bench.latency([r['ms'] for r in rows]) for op,rows in ops.items()}
                   for name,ops in samples.items()}
        ranks={op:sorted(names,key=lambda name:aggregate[name][op]['mean_ms']) for op in ('C','R','U','D')}
        write(out/'results.json',dict(dataset=dataset,aggregate=aggregate,ranks=ranks,samples=samples,
            config=dict(repeats=3,operations_per_repeat=20,base_count=len(base),held_count=len(held),
                        fresh_insert_ids=True,ack_timing=True,vg_fsync=True,verification_queries=260,
                        updates='normalized 0.9*v + 0.1*roll(v,1); metadata maintenance_revision=updated',
                        operation_ids_sha256=hashlib.sha256(json.dumps(operation_ids).encode()).hexdigest())))
        print(f'RESULT {dataset} '+json.dumps(dict(aggregate=aggregate,ranks=ranks)),flush=True)
    finally:
        for name,store in stores.items():
            if name != 'vg_hdmg':
                store.close()


if __name__ == '__main__':
    import test_hot_hdmg_v6
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    run(parser.parse_args().dataset)
