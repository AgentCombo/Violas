from __future__ import annotations
from collections import defaultdict, Counter
from typing import Any, Iterable
import heapq
import math
import numpy as np

import copy
import os
import pickle
from violas.storage import VectorMap, VectorGroup
from . import scoring as soft
from . import directory
from .objects import make_object_map


class VGStore:
    def __init__(self, cache, objects, path, graph):
        self.template = cache
        self.objects = objects
        self.path = path
        self.graph = graph
        self.active = {}
        self.routes = {}
        self.cache = None

    def rebuild(self):
        vm = VectorMap()
        vm.key_vectors = self.template['vectormap'].key_vectors.copy()
        groups = defaultdict(list)
        for oid, obj in self.active.items():
            route = (obj['description']['entity'], obj['description']['entity']) if self.cache is None else self.routes[oid]
            groups[route].append(oid)
        for (key, name), ids in sorted(groups.items()):
            ids.sort()
            vectors = [self.active[oid]['vector'] for oid in ids]
            descriptions = [copy.deepcopy(self.active[oid]['description']) for oid in ids]
            old = self.active[ids[0]]['group']
            group = VectorGroup(name, np.mean(vectors, axis=0), old.rep_description,
                                vectors, descriptions, old.vector_type, old.group_type)
            if self.cache is None:
                vm.insert_with_auto_cluster(key, group, {'entity': descriptions[0]['entity']}, alpha=0.5)
            else:
                vm.insert(key, group, {'entity': descriptions[0]['entity']})
        entity_objects, locator = defaultdict(list), {}
        for key, entry in vm.data.items():
            for group in entry['groups']:
                for pos, desc in enumerate(group.descriptions):
                    oid = desc['object_id']
                    locator[oid] = (key, group.group_name, pos)
                    self.routes[oid] = (key, group.group_name)
                    entity_objects[desc['entity']].append(oid)
        if self.graph:
            vm.build_hdmg(embedding_k=16, semantic_intra_k=8)
        else:
            # The shared directory enumerates group nodes independently of graph edges.
            vm._hdmg_nodes = [(key, group) for key, entry in vm.data.items() for group in entry['groups']]
        cache = {'vectormap': vm, 'entity_objects': dict(entity_objects), 'object_locator': locator,
                 'schemas': self.template['schemas']}
        directory = directory.build_semantic_directory(cache)
        with self.path.open('wb') as handle:
            pickle.dump({'cache': cache, 'directory': directory}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        self.cache = cache
        self.directory = directory

    def insert(self, ids):
        for oid in ids:
            obj = self.objects[oid]
            if self.cache is not None:
                entity = obj['description']['entity']
                candidates = [(key, group) for key, entry in self.cache['vectormap'].data.items()
                              if entry['metadata']['entity'] == entity for group in entry['groups']]
                if candidates:
                    key, group = max(candidates, key=lambda pair: soft.cosine_similarity(obj['vector'], pair[1].representative))
                    self.routes[oid] = (key, group.group_name)
                else:
                    self.routes[oid] = (f'{entity}-0001', entity)
            self.active[oid] = {**obj, 'vector': obj['vector'].copy(), 'description': copy.deepcopy(obj['description'])}
        self.rebuild()

    def delete(self, ids):
        for oid in ids:
            del self.active[oid]
        self.rebuild()

    def verify(self, expected, touched, deleted):
        if set(self.cache['object_locator']) != expected:
            raise ValueError('VG IDs inconsistent after maintenance')
        members = {oid for ids in self.directory['node_members'] for oid in ids}
        if members != expected:
            raise ValueError('VG directory inconsistent after maintenance')
        actual = make_object_map(self.cache)
        if set(actual) != expected:
            raise ValueError('VG object reconstruction failed')
