"""A complete small intent + embedding query, using only the public package."""
import numpy as np
from violas import VectorMap, VectorGroup, IntentIndex


def main():
    vm = VectorMap()
    vm.key_vectors = {'airplanes': np.array([1., 0., 0.], np.float32)}
    vectors = [np.array(v, np.float32) for v in [[1,0,0], [.98,.2,0], [.8,0,.6], [.7,0,.7]]]
    descriptions = [dict(object_id=f'airplanes/{i}', entity='airplanes',
                         facet_values={'propulsion': 'jet' if i < 2 else 'propeller'})
                    for i in range(len(vectors))]
    group = VectorGroup('airplanes', np.mean(vectors, axis=0), 'airplanes',
                        vectors, descriptions, 'image', 'entity')
    vm.insert('airplanes', group, {'entity': 'airplanes'})
    index = IntentIndex.from_vectormap(vm)
    plan = dict(entity_preference='airplanes', required_entity='airplanes',
                hard_filters=[], semantic_weight=0., embedding_weight=1.,
                facet_dimension='propulsion', top_k=3,
                category_preferences={'jet': {'lambda': .15, 'quota': 1},
                                      'propeller': {'lambda': .15, 'quota': 1}})
    plan['lambda'] = .15
    ids, scores, stats = index.search(np.array([1., 0., 0.]), plan)
    assert len(ids) == 3 and any(descriptions[int(oid.split('/')[-1])]['facet_values']['propulsion'] == 'propeller' for oid in ids)
    print('Selected IDs:', ids)
    print('Scored candidates:', len(scores))
    print('Search statistics:', stats)


if __name__ == '__main__':
    main()
