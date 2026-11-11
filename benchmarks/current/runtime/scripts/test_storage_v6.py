"""Validate actual persisted CRUD indexes against their active object records."""
import argparse
import copy
from collections import defaultdict
from pathlib import Path
import numpy as np
from run_final_violas_bench import load, soft, ROOT, read
from incremental_vg_v6 import IncrementalVGStore

def check(store, queries, bindings):
    c=store.corpus
    expected=defaultdict(set)
    entities=defaultdict(set)
    for oid,obj in store.active.items():
        i=c.locator[oid]
        entities[obj['description']['entity']].add(i)
        for token in obj['description']['facet_values'].items():expected[token].add(i)
    for token in expected.keys() | store.engine.postings.keys():
        assert set(store.engine.postings.get(token,[]))==expected[token],token
    for entity,values in entities.items():assert set(store.engine.entities[entity])==values
    for node,members in enumerate(c.members):
        assert all(c.active[i] for i in members)
        if len(members):
            distances=np.linalg.norm(c.vectors[members].astype(float)-c.centers[node].astype(float),axis=1)
            assert distances.max()<=store.engine.radii[node]+2e-6
        tokens={t for i in members for t in c.facets[i].items()}
        assert c.node_tokens[node]==tokens
    for token,nodes in c.token_nodes.items():
        assert set(nodes)=={i for i,tokens in enumerate(c.node_tokens) if token in tokens}
    for row in queries[:20]:
        plan=row['resolved_plan'];query=bindings[row['id']]['normalized']
        eligible=[i for i in range(len(c.ids)) if c.active[i] and c.eligible(i,plan)]
        scores=c.scores(eligible,query,plan,c.semantic_scores(plan))
        exact=soft.select(scores,c,plan)
        ids,found,_=store.engine.search(query,plan,budget=len(c.members),score_batch=4)
        objective=lambda result,sc:soft.core.objective(result,sc,c.objects,plan['facet_dimension'],plan['lambda'],plan['category_preferences'])
        assert abs(objective(ids,found)-objective(exact,scores))<2e-6

def run(dataset):
    previous,config,source,cache,directory=load(dataset)
    objects=soft.bench.retrieval.make_object_map(cache)
    out=ROOT/'maintenance_v6'/dataset
    saved=max(out.glob('updates_*.pkl'),key=lambda p:p.stat().st_mtime_ns)
    # Recovery reads and replays without modifying the saved files.
    store=IncrementalVGStore(cache,copy.deepcopy(objects),saved,True)
    store.recover()
    queries=read(ROOT/dataset/'results.json')['rows']
    bindings=soft.bench.choose_queries(previous['rows'],source,set(objects))
    check(store,queries,bindings)
    # Exercise insertion, facet-changing update, and deletion on the recovered
    # in-memory state, without appending to the published experiment WAL.
    held=sorted(set(objects)-set(store.active))[:5]
    store.mutate('insert',held,journal=False)
    check(store,queries,bindings)
    for oid in held:
        obj=copy.deepcopy(store.objects[oid])
        obj['description']['facet_values']['test_revision']='changed'
        store.objects[oid]=obj
    store.mutate('update',held,journal=False)
    check(store,queries,bindings)
    store.mutate('delete',held,journal=False)
    check(store,queries,bindings)
    print('PASS',dataset,'recovery, postings, graph memberships, bounds, and 80 exact-search comparisons',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True)
    run(p.parse_args().dataset)
