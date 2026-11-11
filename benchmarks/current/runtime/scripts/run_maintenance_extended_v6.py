#!/usr/bin/env python3
"""Build, point reads, insert/update/delete, and independent index creation.

All stores use isolated names; writes are checked for vector and metadata
visibility. VG additionally reloads its snapshot and durable operation log.
"""
import argparse
import copy
import os
import pickle
from pathlib import Path
from statistics import fmean
import time
from collections import defaultdict
import numpy as np
from violas.retrieval import directory as core_directory
from pymilvus import MilvusClient,DataType
from qdrant_client import QdrantClient,models
from run_final_violas_bench import ROOT,DATASETS,load,write,soft
from run_final_violas_maintenance import DBStore,measure,stored_bytes
from incremental_vg_v6 import IncrementalVGStore
from fast_hdmg_v3 import build_fast_hdmg
from adaptive_hdmg import AdaptiveHDMG
from fast_summary import geometric_summary


class Database(DBStore):
    def update(self,ids):
        for start in range(0,len(ids),500):
            batch=ids[start:start+500]
            rows=[dict(id=self.numbers[oid],object_id=oid,entity=self.objects[oid]['description']['entity'],
                facet_tokens=[f'{k}={v}' for k,v in self.objects[oid]['description']['facet_values'].items()],
                vector=self.objects[oid]['normalized'].tolist()) for oid in batch]
            if self.db=='milvus':
                self.client.upsert(self.name,rows)
            elif self.db=='qdrant':
                self.client.upsert(self.name,[models.PointStruct(id=r['id'],vector=r['vector'],
                    payload={k:r[k] for k in ('object_id','entity','facet_tokens')}) for r in rows],wait=True)
            else:
                metadata=[{'object_id':oid,'entity':self.objects[oid]['description']['entity'],
                    **{f'facet__{k}':v for k,v in self.objects[oid]['description']['facet_values'].items()}} for oid in batch]
                self.collection.update(ids=batch,embeddings=[r['vector'] for r in rows],metadatas=metadata)
        if self.db=='milvus':
            self.client.flush(self.name)

    def get(self,ids):
        if self.db=='milvus':
            rows=self.client.get(self.name,ids=[self.numbers[oid] for oid in ids],output_fields=['object_id','vector','entity','facet_tokens'])
            return {r['object_id']:(r['vector'],r['facet_tokens']) for r in rows}
        if self.db=='qdrant':
            rows=self.client.retrieve(self.name,ids=[self.numbers[oid] for oid in ids],with_vectors=True,with_payload=True)
            return {r.payload['object_id']:(r.vector,r.payload['facet_tokens']) for r in rows}
        rows=self.collection.get(ids=ids,include=['embeddings','metadatas'])
        return {oid:(vector,[f'{k.removeprefix("facet__")}={v}' for k,v in meta.items() if k.startswith('facet__')])
            for oid,vector,meta in zip(rows['ids'],rows['embeddings'],rows['metadatas'])}

    def ready(self):
        if self.db!='qdrant':
            return super().ready()
        deadline=time.monotonic()+600
        while True:
            info=self.client.get_collection(self.name)
            self.index_state={'points':info.points_count,'indexed_vectors':info.indexed_vectors_count,'status':str(info.status)}
            if info.indexed_vectors_count>=info.points_count and getattr(info.status,'value',info.status)=='green':
                return
            if time.monotonic()>deadline:
                raise TimeoutError(str(self.index_state))
            time.sleep(.02)

    def create_unindexed(self):
        dimension=len(next(iter(self.objects.values()))['normalized'])
        if self.db=='milvus':
            self.path=self.path.with_suffix('.db')
            self.client=MilvusClient(str(self.path.resolve()))
            schema=self.client.create_schema(auto_id=False,enable_dynamic_field=False)
            schema.add_field('id',DataType.INT64,is_primary=True)
            schema.add_field('object_id',DataType.VARCHAR,max_length=256)
            schema.add_field('entity',DataType.VARCHAR,max_length=128)
            schema.add_field('facet_tokens',DataType.ARRAY,element_type=DataType.VARCHAR,max_capacity=32,max_length=160)
            schema.add_field('vector',DataType.FLOAT_VECTOR,dim=dimension)
            self.client.create_collection(self.name,schema=schema)
        elif self.db=='qdrant':
            self.client=QdrantClient(url=self.config.qdrant_url,timeout=180)
            self.client.create_collection(self.name,vectors_config=models.VectorParams(size=dimension,distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=0),optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0))
        else:
            raise NotImplementedError('Chroma builds its HNSW with ingestion; no separate create_index operation')

    def create_indexes(self):
        if self.db=='milvus':
            params=self.client.prepare_index_params()
            params.add_index('vector',index_type='IVF_FLAT',metric_type='COSINE',params={'nlist':128})
            params.add_index('entity',index_type='INVERTED',index_name='entity_inverted')
            self.client.create_index(self.name,params)
        elif self.db=='qdrant':
            for field in ('entity','facet_tokens'):
                self.client.create_payload_index(self.name,field,models.PayloadSchemaType.KEYWORD,wait=True)
            self.client.update_collection(self.name,hnsw_config=models.HnswConfigDiff(m=16,ef_construct=100,full_scan_threshold=10),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1))


def run(dataset,method):
    previous,config,source,cache,directory=load(dataset)
    objects=soft.bench.retrieval.make_object_map(cache)
    objects={oid:{**obj,'description':copy.deepcopy(obj['description'])} for oid,obj in objects.items()}
    all_ids=sorted(objects)
    out=ROOT/'maintenance_v6'/dataset
    out.mkdir(parents=True,exist_ok=True)
    is_vg=method=='vg_hdmg'
    def make_store(tag):
        if is_vg:
            return IncrementalVGStore(cache,objects,out/f'{tag}_{time.time_ns()}.pkl',True)
        return Database(method,config,objects,out)
    def get(store,ids):
        if not is_vg:
            return store.get(ids)
        return {oid:(store.active[oid]['normalized'].copy(),[f'{k}={v}' for k,v in store.active[oid]['description']['facet_values'].items()])
            for oid in ids if oid in store.active}
    def verify_fields(store,ids):
        rows=get(store,ids)
        assert set(rows)==set(ids)
        for oid,(vector,tokens) in rows.items():
            assert np.allclose(vector,objects[oid]['normalized'],atol=2e-6)
            assert set(tokens)=={f'{k}={v}' for k,v in objects[oid]['description']['facet_values'].items()}
    def revise(ids):
        for oid in ids:
            obj=objects[oid]
            v=np.asarray(obj['normalized'],np.float32)
            v=(.9*v+.1*np.roll(v,1)).astype(np.float32)
            norm=float(np.linalg.norm(v))
            if norm==0:
                v[0]=1.
            else:
                v/=norm
            assert np.isfinite(v).all()
            obj['vector']=v.copy()
            obj['normalized']=v.copy()
            obj['description']=copy.deepcopy(obj['description'])
            obj['description']['facet_values']['maintenance_revision']='updated'
    store=make_store('full')
    try:
        def full_build():
            if not is_vg: store.create()
            store.insert(all_ids)
        build=measure(full_build,None if is_vg else store.ready)
        store.verify(set(all_ids),all_ids[:10],False)
        verify_fields(store,all_ids[:10])
        build_bytes=stored_bytes(store.path)
        full_snapshot_path=str(store.path)
        full_cache=store.cache if is_vg else None
        quality_check=None
        if is_vg:
            frozen=__import__('run_final_violas_bench').read(ROOT/dataset/'results.json')
            bindings=soft.bench.choose_queries(previous['rows'],source,set(store.corpus.ids))
            checked=[]
            for row in frozen['rows']:
                query=bindings[row['id']]['normalized']
                t=time.perf_counter_ns()
                ids,scores,stats=store.engine.search(query,row['resolved_plan'],budget=32,score_batch=4)
                elapsed=(time.perf_counter_ns()-t)/1e6
                value=soft.core.objective(ids,scores,store.corpus.objects,row['resolved_plan']['facet_dimension'],row['resolved_plan']['lambda'],row['resolved_plan']['category_preferences'])
                ratio=value/row['oracle']['objective']
                assert ratio<=1+2e-6
                checked.append(dict(id=row['id'],objective_ratio=ratio,ms=elapsed,stats=stats,ids=ids))
            quality_check=dict(rows=checked,objective_ratio=fmean(r['objective_ratio'] for r in checked),latency=soft.bench.latency([r['ms'] for r in checked]),group_budget=32,score_batch=4)

        read_ids=list(np.random.default_rng(43).choice(all_ids,10,replace=False))
        reads=[measure(lambda oid=oid:get(store,[oid])) for oid in read_ids]
        if not is_vg: store.close()
        rng=np.random.default_rng(42)
        groups=defaultdict(list)
        for oid in all_ids: groups[objects[oid]['description']['entity']].append(oid)
        base,held=[],[]
        for entity,ids in sorted(groups.items()):
            ids=list(rng.permutation(ids)); count=max(1,round(.1*len(ids)))
            held.extend(ids[:count]);base.extend(ids[count:])
        store=make_store('updates')
        if not is_vg:store.create()
        store.insert(base)
        if not is_vg:store.ready()
        active=set(base)
        singles=list(np.random.default_rng(43).choice(held,10,replace=False))
        single_insert=[];single_delete=[];single_update=[]
        for oid in singles:
            single_insert.append(measure(lambda oid=oid:store.insert([oid])))
            active.add(oid);store.verify(active,[oid],False);verify_fields(store,[oid])
        for oid in singles:
            single_delete.append(measure(lambda oid=oid:store.delete([oid])))
            active.remove(oid);store.verify(active,[oid],True)
        update_ids=list(np.random.default_rng(44).choice(base,10,replace=False))
        for oid in update_ids:
            revise([oid])
            single_update.append(measure(lambda oid=oid:store.update([oid])))
            store.verify(active,[oid],False);verify_fields(store,[oid])
        batch_insert=measure(lambda:store.insert(held),None if is_vg else store.ready)
        active.update(held);store.verify(active,held[:10],False);verify_fields(store,held)
        revise(held)
        batch_update=measure(lambda:store.update(held),None if is_vg else store.ready)
        store.verify(active,held[:10],False);verify_fields(store,held)
        batch_delete=measure(lambda:store.delete(held),None if is_vg else store.ready)
        active.difference_update(held);store.verify(active,held[:10],True)
        recovery=None
        if is_vg:
            recovery=measure(store.recover)
            store.verify(active,held[:10],True);verify_fields(store,update_ids)
        if not is_vg:store.close()
        # A separate index-only experiment uses the original, unmodified input.
        objects=soft.bench.retrieval.make_object_map(cache)
        index_build=None
        if is_vg:
            vm=copy.deepcopy(full_cache['vectormap'])
            if hasattr(vm,'_fast_hdmg_state'):del vm._fast_hdmg_state
            def index_op():
                build_fast_hdmg(vm,embedding_k=16,semantic_intra_k=8)
                snap={**full_cache,'vectormap':vm}
                original_summary=core_directory._geometric_summary
                core_directory._geometric_summary=geometric_summary
                try:
                    idx=soft.core.build_semantic_directory(snap)
                finally:
                    core_directory._geometric_summary=original_summary
                engine=AdaptiveHDMG(soft.Corpus(snap,idx))
                with (out/'new_index.pkl').open('wb') as f:
                    pickle.dump({'directory':idx,'edges':(vm._hdmg_embedding_edges,vm._hdmg_semantic_edges),
                        'pivots':engine.pivots,'radii':engine.radii,'postings':engine.postings},f,pickle.HIGHEST_PROTOCOL)
                    f.flush();os.fsync(f.fileno())
            index_build=measure(index_op)
        elif method!='chroma':
            store=make_store('new_index')
            store.create_unindexed();store.insert(all_ids)
            index_build=measure(store.create_indexes,store.ready)
            store.verify(set(all_ids),all_ids[:10],False)
            store.close()
        report=dict(dataset=dataset,method=method,objects=len(objects),build=build,build_bytes=build_bytes,
            point_read=reads,single_insert=single_insert,single_delete=single_delete,single_update=single_update,
            batch_insert=batch_insert,batch_update=batch_update,batch_delete=batch_delete,batch_size=len(held),
            index_build=index_build,recovery=recovery,quality_after_build=quality_check,full_snapshot_path=full_snapshot_path,validation='IDs, updated vectors and metadata visible; VG WAL reload verified',
            config=dict(single_samples=10,seed=42,poll_seconds=.02,
                vg='unit embeddings; KMeans n_init=1, random_state=42, alpha=0.5; compact canonical metadata; batch graph and hot changed-node updates; logical deletion retains conservative caps/routing; physical compaction excluded; fsync snapshot + WAL',
                index_only='already loaded vectors/groups; construct graph and metadata indexes; Chroma not separately exposed'))
        write(out/f'{method}.json',report)
        print(f'COMPLETE maintenance v2 {dataset} {method} build={build["ready_ms"]:.2f}ms',flush=True)
    finally:
        if not is_vg:store.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=DATASETS,required=True)
    p.add_argument('--method',choices=('vg_hdmg','milvus','qdrant','chroma'),required=True)
    a=p.parse_args();run(a.dataset,a.method)
