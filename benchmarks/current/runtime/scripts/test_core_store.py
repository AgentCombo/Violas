"""Check the integrated package's CRUD, recovery, and query visibility."""
from collections import defaultdict
from pathlib import Path
import copy
import tempfile
import numpy as np
from violas import VectorMap, VectorGroup, IntentIndex
from violas.retrieval import IncrementalVGStore, evaluate_objective
from violas.retrieval.objects import make_object_map
from violas.retrieval.scoring import select


def run():
    rng = np.random.default_rng(71)
    vm = VectorMap()
    entities, locator = defaultdict(list), {}
    for entity in ('A', 'B', 'C'):
        vm.key_vectors[entity] = rng.normal(size=8).astype(np.float32)
        matrix = rng.normal(size=(20, 8)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        descriptions = [dict(object_id=f'{entity}/{i:02d}', entity=entity,
                             facet_values={'color': 'red' if i % 2 else 'blue'})
                        for i in range(len(matrix))]
        vm.insert(entity, VectorGroup(entity, matrix.mean(axis=0), entity,
                                      list(matrix), descriptions), {'entity': entity})
        for i, desc in enumerate(descriptions):
            oid = desc['object_id']
            entities[entity].append(oid)
            locator[oid] = (entity, entity, i)
    cache = dict(vectormap=vm, entity_objects=dict(entities), object_locator=locator, schemas={})
    objects = make_object_map(cache)
    held = ['A/19', 'B/19', 'C/19']
    initial = sorted(set(objects) - set(held))
    plan = dict(entity_preference='A', required_entity=None, hard_filters=[],
                facet_dimension='color', category_preferences={
                    'red': {'lambda': .15, 'quota': 1},
                    'blue': {'lambda': .15, 'quota': 1}},
                embedding_weight=.65, semantic_weight=.35, top_k=3)
    plan['lambda'] = .15
    query = objects['A/01']['normalized']

    def check(store, expected):
        store.verify(set(expected), [], False)
        c = store.corpus
        assert {c.ids[i] for i in np.flatnonzero(c.active)} == set(expected)
        for hard in ([], [{'key': 'color', 'value': 'red'}]):
            p = {**plan, 'hard_filters': hard}
            eligible = [i for i in np.flatnonzero(c.active) if c.eligible(i, p)]
            scores = c.scores(eligible, query, p, c.semantic_scores(p))
            oracle = select(scores, c, p)
            result, found, _ = store.engine.search(query, p, budget=len(c.members))
            value = lambda ids, s: evaluate_objective(ids, s, c.objects,
                                                     p['facet_dimension'], p['lambda'], p['category_preferences'])
            assert abs(value(result, found) - value(oracle, scores)) < 2e-6
            assert set(result) <= set(expected)

    with tempfile.TemporaryDirectory(prefix='violas-core-test-') as folder:
        path = Path(folder)/'snapshot.pkl'
        store = IncrementalVGStore(cache, copy.deepcopy(objects), path, True)
        store.insert(initial)
        check(store, initial)
        store.insert(held)
        check(store, objects)
        for oid in held:
            store.objects[oid]['description']['facet_values']['color'] = 'red'
        store.update(held)
        check(store, objects)
        store.delete(['A/19'])
        expected = set(objects) - {'A/19'}
        check(store, expected)
        restored = IncrementalVGStore(cache, copy.deepcopy(objects), path, True)
        restored.recover()
        check(restored, expected)
        for oid in ['B/19', 'C/19']:
            assert restored.active[oid]['description']['facet_values']['color'] == 'red'
        assert store.engine.__class__.__module__ == 'violas.retrieval.hdmg'
    print('PASS integrated store: build, insert, facet update, delete, WAL recovery and exact objective checks')


if __name__ == '__main__':
    run()
