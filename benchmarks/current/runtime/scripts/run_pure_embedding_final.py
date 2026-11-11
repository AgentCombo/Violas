"""Pure embedding recall/latency, native VG and minimal native VDB responses."""
import argparse
from pathlib import Path
from statistics import fmean
from time import perf_counter_ns
import numpy as np
import faiss
import os
from pymilvus import MilvusClient
from qdrant_client import models
from run_final_violas_bench import ROOT,load,read,write,soft
from adaptive_hdmg import AdaptiveHDMG
from embedding_hdmg_native import NativeEmbeddingHDMG

def run(dataset):
    previous,config,source,_,_=load(dataset)
    storage=read(ROOT/'maintenance_v6'/dataset/'vg_hdmg.json')
    saved=soft.bench.load_pickle(Path(storage['full_snapshot_path']),config.module_path)
    c=soft.Corpus(saved['cache'],saved['directory'])
    base=AdaptiveHDMG(c)
    start=perf_counter_ns();engine=NativeEmbeddingHDMG(c,base.pivots)
    layout_ms=(perf_counter_ns()-start)/1e6
    faiss.omp_set_num_threads(1)
    start=perf_counter_ns()
    object_index=faiss.IndexHNSWFlat(c.vectors.shape[1],32,faiss.METRIC_INNER_PRODUCT)
    object_index.hnsw.rng=faiss.RandomGenerator(42)
    object_index.hnsw.efConstruction=200
    object_index.hnsw.efSearch=256
    object_index.add(np.ascontiguousarray(c.vectors,np.float32))
    auxiliary=ROOT/'pure_embedding_final'/dataset
    auxiliary.mkdir(parents=True,exist_ok=True)
    index_path=auxiliary/'embedding_hnsw.faiss'
    faiss.write_index(object_index,str(index_path))
    with index_path.open('rb') as f:os.fsync(f.fileno())
    auxiliary_build_ms=(perf_counter_ns()-start)/1e6
    frozen=read(ROOT/dataset/'results.json')
    bindings=soft.bench.choose_queries(previous['rows'],source,set(c.ids))
    qdrant,_=soft.setup_qdrant(config,c)
    milvus=MilvusClient(str(Path(f'cache/five_system/{dataset}_milvus_array.db').resolve()))
    _,chroma=soft.backends.open_chroma(Path(f'cache/five_system/{dataset}_chroma'),f'{dataset}_chroma')
    milvus.load_collection(f'{dataset}_array')
    variants={'vg_32':(32,64),'vg_64':(64,128),'vg_128':(128,256),'vg_256':(256,512)}
    names=['vg_128','vg_embedding_stable','milvus','qdrant','chroma']
    rows=[]
    # Float64 cosine, deterministic object-ID tie break. Norms are shared gallery
    # preprocessing, not per-query answers; all query answers are recomputed.
    gallery=c.vectors.astype(np.float64)
    norms=np.linalg.norm(gallery,axis=1)
    for number,original in enumerate(frozen['rows']):
        query=bindings[original['id']]['normalized']
        assert bindings[original['id']]['object_id']==original['query_object']['object_id']
        methods={};offset=number%len(names)
        for name in names[offset:]+names[:offset]:
            start=perf_counter_ns();stats={}
            if name in variants:
                ids,scores,stats=engine.search(query,*variants[name])
            elif name=='vg_exact':ids,scores,stats=engine.search(query,exhaustive=True)
            elif name=='vg_embedding_stable':
                distance,positions=object_index.search(np.ascontiguousarray(query[None,:],np.float32),4)
                indices=positions[0]
                matrix=c.vectors[indices].astype(np.float64)
                denominator=engine.norms[indices]*np.linalg.norm(query.astype(np.float64))
                values=np.divide(matrix@query.astype(np.float64),denominator,out=np.zeros(4),where=denominator>0)
                order=np.lexsort((indices,-values))
                fallback=abs(values[order[2]]-values[order[3]])<=1e-12
                if fallback:
                    ids,scores,stats=engine.search(query,exhaustive=True)
                else:ids=[c.ids[int(indices[i])] for i in order[:3]]
                stats['tie_fallback']=bool(fallback)
            elif name=='vg_embedding_index':
                distance,positions=object_index.search(np.ascontiguousarray(query[None,:],np.float32),3)
                ids=[c.ids[int(i)] for i in positions[0]]
            elif name=='milvus':
                result=milvus.search(f'{dataset}_array',data=[query.tolist()],anns_field='vector',limit=3,
                    output_fields=['object_id'],search_params={'metric_type':'COSINE','params':{'nprobe':config.nprobe}})
                ids=[hit['entity']['object_id'] for hit in result[0]]
            elif name=='qdrant':
                result=qdrant.query_points(f'soft_{config.index_tag}',query=query.tolist(),limit=3,
                    search_params=models.SearchParams(hnsw_ef=config.ef),with_payload=['object_id'],with_vectors=False)
                ids=[point.payload['object_id'] for point in result.points]
            else:
                result=chroma.query(query_embeddings=[query.tolist()],n_results=3,include=['distances'])
                ids=result['ids'][0]
            elapsed=(perf_counter_ns()-start)/1e6
            methods[name]={'ids':ids,'ms':elapsed,'stats':stats}
        denom=norms*np.linalg.norm(query.astype(np.float64))
        sim=np.divide(gallery@query.astype(np.float64),denom,out=np.zeros(len(c.ids)),where=denom>0)
        order=np.lexsort((np.asarray(c.ids),-sim))[:3]
        gt=[c.ids[i] for i in order]
        for name,result in methods.items():
            assert len(set(result['ids']))==3
            result['recall_at_3']=len(set(result['ids'])&set(gt))/3
            # Diagnostic only: do not silently excuse missing strict IDs.
            result['ties_at_boundary']=int(np.count_nonzero(np.abs(sim-sim[order[-1]])<=1e-12))
            if name=='vg_exact':
                assert all(abs(sim[c.locator[a]]-sim[b])<1e-10 for a,b in zip(result['ids'],order))
        rows.append({'id':original['id'],'query_object':original['query_object'],'oracle_ids':gt,'query_norm':float(np.linalg.norm(query.astype(float))),'methods':methods})
    aggregate={name:{'tie_fallbacks':sum(r['methods'][name]['stats'].get('tie_fallback',False) for r in rows),'recall_at_3':fmean(r['methods'][name]['recall_at_3'] for r in rows),
        'latency':soft.bench.latency([r['methods'][name]['ms'] for r in rows])} for name in names}
    out=ROOT/'pure_embedding_final'/dataset;out.mkdir(parents=True,exist_ok=True)
    write(out/'results.json',dict(dataset=dataset,rows=rows,aggregate=aggregate,layout_build_ms=layout_ms,auxiliary_index_build_ms=auxiliary_build_ms,auxiliary_index_bytes=index_path.stat().st_size,
        layout_bytes=sum(a.nbytes for a in [engine.norms,engine.centers,engine.edges,engine.edge_offsets,engine.members,engine.member_offsets,engine.entries]),
        config=dict(workload_sha256=frozen['config']['workload_sha256'],variants=variants,repeats=1,active_warmup=False,
        auxiliary_index=dict(library='faiss',version=faiss.__version__,M=32,efConstruction=200,efSearch=256,threads=1,seed=42,route='object-level embedding HNSW; bypasses HDMG'),result_payload='top3 IDs/scores only; no vectors',oracle='full-gallery float64 cosine, object-ID tie break',
        source_store=storage['full_snapshot_path'],compiler='g++ -O3 -march=native -fopenmp-simd -shared -fPIC')))
    milvus.close();qdrant.close()
    print(dataset,aggregate,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);run(p.parse_args().dataset)
