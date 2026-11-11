#!/usr/bin/env python3
"""Final quality and pure-embedding tests on the newly built optimized VG store."""
import argparse
from pathlib import Path
from statistics import fmean
from time import perf_counter_ns
import numpy as np
from pymilvus import MilvusClient
from run_final_violas_bench import ROOT,DATASETS,load,read,write,soft
from run_caltech_filter_order import fetch
from adaptive_hdmg import AdaptiveHDMG
from embedding_hdmg import EmbeddingHDMG

def run(dataset):
    previous,config,source,_,_=load(dataset)
    storage=read(ROOT/'maintenance_v6'/dataset/'vg_hdmg.json')
    saved=soft.bench.load_pickle(Path(storage['full_snapshot_path']),config.module_path)
    corpus=soft.Corpus(saved['cache'],saved['directory'])
    search=AdaptiveHDMG(corpus)
    embedding=EmbeddingHDMG(corpus,search.pivots)
    frozen=read(ROOT/dataset/'results.json')
    bindings=soft.bench.choose_queries(previous['rows'],source,set(corpus.ids))
    config.candidate_k=3
    qdrant,qinfo=soft.setup_qdrant(config,corpus)
    milvus=MilvusClient(str(Path(f'cache/five_system/{dataset}_milvus_array.db').resolve()))
    names=('vg_hdmg','vg_rep','embedding_vg','embedding_milvus','embedding_qdrant','embedding_chroma')
    rows=[]
    try:
        milvus.load_collection(f'{dataset}_array')
        _,chroma=soft.backends.open_chroma(Path(f'cache/five_system/{dataset}_chroma'),f'{dataset}_chroma')
        clients=dict(milvus=milvus,qdrant=qdrant,chroma=chroma)
        for number,original in enumerate(frozen['rows']):
            bound=bindings[original['id']]
            assert bound['object_id']==original['query_object']['object_id']
            query=bound['normalized'];plan=original['resolved_plan']
            raw=np.asarray(source['test_data'][original['entity_hint']]['vectors'][bound['test_index']],np.float32).reshape(-1)
            pure=dict(required_entity=None,entity_preference=corpus.entities[0],hard_filters=[],facet_dimension=None,
                      category_preferences={},semantic_weight=0.,embedding_weight=1.,top_k=3)
            pure['lambda']=0.
            methods={};offset=number%len(names)
            for name in names[offset:]+names[:offset]:
                start=perf_counter_ns()
                if name=='vg_hdmg':
                    ids,scores,stats=search.search(query,plan,budget=32,score_batch=4)
                elif name=='vg_rep':
                    ids,scores,stats=soft.representative_scan(corpus,raw,plan,32,online_cosine=True)
                elif name=='embedding_vg':
                    ids,scores,stats=embedding.search(query,group_budget=32,navigation_budget=64)
                else:
                    db=name.removeprefix('embedding_')
                    fetched=fetch(db,clients[db],config,corpus,query,pure,pre=False)
                    ids=[oid for oid,v in fetched]
                    vals=(1+np.einsum('ij,j->i',np.asarray([v for oid,v in fetched],np.float32),query))/2
                    scores=dict(zip(ids,map(float,vals)))
                    stats=dict(unique_member_scores=len(ids))
                ms=(perf_counter_ns()-start)/1e6
                methods[name]=dict(ids=ids,scores=scores,stats=stats,ms=ms)
            pure_values=(1+np.einsum('ij,j->i',corpus.vectors,query))/2
            best=np.sort(pure_values)[-3:]
            pure_oracle=float(best.sum())
            for name,r in methods.items():
                if name.startswith('embedding_'):
                    value=sum(float(pure_values[corpus.locator[oid]]) for oid in r['ids'])
                    ratio=value/pure_oracle
                else:
                    assert all(corpus.eligible(corpus.locator[oid],plan) for oid in r['ids'])
                    value=soft.core.objective(r['ids'],r['scores'],corpus.objects,plan['facet_dimension'],plan['lambda'],plan['category_preferences'])
                    ratio=value/original['oracle']['objective']
                assert ratio<=1+2e-6
                r.update(objective=value,objective_ratio=ratio)
            rows.append(dict(id=original['id'],query_object=original['query_object'],plan=plan,methods=methods,
                oracle=original['oracle'],pure_oracle=pure_oracle))
        aggregate={name:dict(objective_ratio=fmean(r['methods'][name]['objective_ratio'] for r in rows),
            latency=soft.bench.latency([r['methods'][name]['ms'] for r in rows]),
            scored_objects=fmean(r['methods'][name]['stats']['unique_member_scores'] for r in rows)) for name in names}
        out=ROOT/'integrated_v6'/dataset;out.mkdir(parents=True,exist_ok=True)
        write(out/'results.json',dict(dataset=dataset,rows=rows,aggregate=aggregate,data={**frozen['data'],'clusters':len(corpus.members)},
            config={**vars(config),'repeats':1,'vg_groups':32,'vg_score_batch':4,'embedding_navigation':64,
                'source_store':storage['full_snapshot_path'],'workload_sha256':frozen['config']['workload_sha256'],
                'vg_grouping':'normalized vectors; KMeans n_init=1, random_state=42; alpha=.5',
                'quality_oracle':'unchanged frozen original objective','active_warmup':False}))
        print(f'COMPLETE integrated {dataset} '+str(aggregate),flush=True)
    finally:
        milvus.close();qdrant.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=DATASETS,required=True)
    run(p.parse_args().dataset)
