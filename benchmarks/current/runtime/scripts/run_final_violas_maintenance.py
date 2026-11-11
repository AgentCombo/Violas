#!/usr/bin/env python3
"""Measure storage-ready build and updates on isolated, disposable benchmark stores."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
from pymilvus import DataType, MilvusClient
from qdrant_client import QdrantClient, models

import run_soft_entity_bench as soft
from run_final_violas_bench import ROOT, DATASETS, load, write
from storage.vectormap import VectorMap
from storage.vectorgroup import VectorGroup


from violas.retrieval.store_base import VGStore


class DBStore:
    def __init__(self, db, config, objects, out):
        self.db, self.config, self.objects = db, config, objects
        self.name = f'final_{config.dataset}_{db}_{time.time_ns()}'
        self.numbers = {oid: i for i, oid in enumerate(sorted(objects))}
        self.client = None
        self.collection = None
        self.path = out / self.name
        self.index_state = {}

    def create(self):
        dimension = len(next(iter(self.objects.values()))['normalized'])
        if self.db == 'milvus':
            self.path = self.path.with_suffix('.db')
            self.client = MilvusClient(str(self.path.resolve()))
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field('id', DataType.INT64, is_primary=True)
            schema.add_field('object_id', DataType.VARCHAR, max_length=256)
            schema.add_field('entity', DataType.VARCHAR, max_length=128)
            schema.add_field('facet_tokens', DataType.ARRAY, element_type=DataType.VARCHAR,
                             max_capacity=32, max_length=160)
            schema.add_field('vector', DataType.FLOAT_VECTOR, dim=dimension)
            indexes = self.client.prepare_index_params()
            indexes.add_index('vector', index_type='IVF_FLAT', metric_type='COSINE', params={'nlist':128})
            indexes.add_index('entity', index_type='INVERTED', index_name='entity_inverted')
            self.client.create_collection(self.name, schema=schema, index_params=indexes)
        elif self.db == 'qdrant':
            self.client = QdrantClient(url=self.config.qdrant_url, timeout=180)
            self.client.create_collection(self.name, vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=10),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1))
            for field in ('entity', 'facet_tokens'):
                self.client.create_payload_index(self.name, field, models.PayloadSchemaType.KEYWORD, wait=True)
            self.path = Path('cache/soft_entity/qdrant_server/collections') / self.name
        else:
            import chromadb
            from chromadb.config import Settings
            self.client = chromadb.PersistentClient(str(self.path.resolve()), settings=Settings(anonymized_telemetry=False))
            self.collection = self.client.create_collection(self.name, metadata={'hnsw:space':'cosine'})

    def insert(self, ids):
        for start in range(0, len(ids), 500):
            batch = ids[start:start+500]
            rows = []
            for oid in batch:
                obj = self.objects[oid]
                desc = obj['description']
                rows.append({'id': self.numbers[oid], 'object_id': oid, 'entity': desc['entity'],
                             'facet_tokens': [f'{k}={v}' for k,v in desc['facet_values'].items()],
                             'vector': obj['normalized'].tolist()})
            if self.db == 'milvus':
                self.client.insert(self.name, rows)
            elif self.db == 'qdrant':
                points = [models.PointStruct(id=r['id'], vector=r['vector'], payload={
                    key:r[key] for key in ('object_id','entity','facet_tokens')}) for r in rows]
                self.client.upsert(self.name, points, wait=True)
            else:
                metas = [{'entity': self.objects[oid]['description']['entity'], 'object_id':oid,
                          **{f'facet__{k}':v for k,v in self.objects[oid]['description']['facet_values'].items()}}
                         for oid in batch]
                self.collection.add(ids=batch, embeddings=[r['vector'] for r in rows], metadatas=metas)
        if self.db == 'milvus':
            self.client.flush(self.name)

    def delete(self, ids):
        numbers = [self.numbers[oid] for oid in ids]
        if self.db == 'milvus':
            self.client.delete(self.name, ids=numbers)
            self.client.flush(self.name)
        elif self.db == 'qdrant':
            self.client.delete(self.name, points_selector=models.PointIdsList(points=numbers), wait=True)
        else:
            self.collection.delete(ids=ids)

    def ready(self):
        if self.db == 'milvus':
            self.client.load_collection(self.name)
            self.index_state = self.client.describe_index(self.name, 'vector')
        elif self.db == 'qdrant':
            deadline = time.monotonic()+600
            while True:
                info = self.client.get_collection(self.name)
                self.index_state = {'points':info.points_count, 'indexed_vectors':info.indexed_vectors_count,
                                    'status':str(info.status), 'optimizer':str(info.optimizer_status)}
                if info.indexed_vectors_count >= info.points_count and getattr(info.status, 'value', info.status) == 'green':
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'Qdrant index not ready: {self.index_state}')
                time.sleep(1)

    def verify(self, expected, touched, deleted):
        if self.db == 'milvus':
            response = self.client.query(self.name, filter='', output_fields=['count(*)'])
            count = response[0]['count(*)']
            present = self.client.get(self.name, ids=[self.numbers[oid] for oid in touched], output_fields=['object_id'])
        elif self.db == 'qdrant':
            count = self.client.count(self.name, exact=True).count
            present = self.client.retrieve(self.name, ids=[self.numbers[oid] for oid in touched], with_vectors=False)
        else:
            count = self.collection.count()
            present = self.collection.get(ids=touched, include=[])['ids']
        if count != len(expected) or len(present) != (0 if deleted else len(touched)):
            raise ValueError(f'{self.db} update not visible: {count} / {len(expected)}')

    def close(self):
        if self.db in ('milvus','qdrant') and self.client is not None:
            self.client.close()


def measure(operation, ready=None):
    start = time.perf_counter_ns()
    operation()
    ack = time.perf_counter_ns()
    if ready:
        ready()
    stop = time.perf_counter_ns()
    return {'ack_ms':(ack-start)/1e6, 'ready_ms':(stop-start)/1e6}


def stored_bytes(path):
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
    return None


def run(dataset, method):
    previous, config, source, cache, directory = load(dataset)
    objects = soft.bench.retrieval.make_object_map(cache)
    out = ROOT / dataset / 'maintenance'
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    by_entity = defaultdict(list)
    for oid in sorted(objects):
        by_entity[objects[oid]['description']['entity']].append(oid)
    base, held = [], []
    for entity, ids in sorted(by_entity.items()):
        shuffled = list(rng.permutation(ids))
        split = max(1, round(len(ids)*0.1))
        held.extend(shuffled[:split])
        base.extend(shuffled[split:])
    all_ids = sorted(objects)
    graph = method == 'vg_hdmg'
    store = VGStore(cache, objects, out/f'{method}.pkl', graph) if method.startswith('vg_') else DBStore(method, config, objects, out)
    def build():
        if isinstance(store, DBStore):
            store.create()
        store.insert(all_ids)
    try:
        build_time = measure(build, store.ready if isinstance(store, DBStore) else None)
        build_bytes = stored_bytes(store.path)
        store.verify(set(all_ids), all_ids[:10], False)
        # Use a fresh 90%-train store, so update tests do not inherit build-test tombstones.
        if isinstance(store, DBStore):
            store.close()
            store = DBStore(method, config, objects, out)
            store.create()
        else:
            store = VGStore(cache, objects, out/f'{method}_updates.pkl', graph)
        store.insert(base)
        if isinstance(store, DBStore):
            store.ready()
        store.verify(set(base), held[:10], True)
        samples = {'single_insert':[], 'single_delete':[]}
        active = set(base)
        singles = list(np.random.default_rng(43).choice(held, min(10, len(held)), replace=False))
        for oid in singles:
            active.add(oid)
            samples['single_insert'].append(measure(lambda oid=oid:store.insert([oid]),
                lambda oid=oid:store.verify(active, [oid], False)))
        for oid in singles:
            active.remove(oid)
            samples['single_delete'].append(measure(lambda oid=oid:store.delete([oid]),
                lambda oid=oid:store.verify(active, [oid], True)))
        inserted = measure(lambda:store.insert(held), store.ready if isinstance(store, DBStore) else None)
        store.verify(set(all_ids), held[:10], False)
        deleted = measure(lambda:store.delete(held), store.ready if isinstance(store, DBStore) else None)
        store.verify(set(base), held[:10], True)
        report = {'dataset':dataset, 'method':method, 'objects':len(objects), 'base_count':len(base),
                  'batch_size':len(held), 'single_samples':len(singles), 'build':build_time,
                  'build_bytes':build_bytes,
                  'batch_insert':inserted, 'batch_delete':deleted, **samples,
                  'index_state':store.index_state if isinstance(store, DBStore) else {},
                  'vg_update_policy':'reconstruct groups, recompute summaries/directory, rebuild graph when enabled, fsync snapshot',
                  'input_grouping':'build uses original insert_with_auto_cluster alpha=0.5, random_state=42, n_init=10; updates assign nearest same-entity centroid',
                  'single_timing':'ack_ms is write completion; ready_ms also includes immediate visibility verification',
                  'batch_timing':'write plus index ready; physical compaction excluded'}
        write(out/f'{method}.json', report)
        print(f'COMPLETE maintenance {dataset} {method}', flush=True)
    finally:
        if isinstance(store, DBStore):
            store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--method', choices=('vg_hdmg','vg_rep','milvus','qdrant','chroma'), required=True)
    args = parser.parse_args()
    run(args.dataset, args.method)
