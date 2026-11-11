"""Public adapter for the measured conditional retrieval engine."""
from collections import defaultdict
import numpy as np
from .scoring import Corpus
from .hdmg import AdaptiveHDMG
from .directory import build_semantic_directory
from .hdmg_build import build_fast_hdmg


class IntentIndex(AdaptiveHDMG):
    """Query an immutable gallery snapshot with an already resolved intent.

    Existing VectorMap methods retain their legacy behavior. Rebuild this
    adapter after modifying the map, or use IncrementalVGStore for measured
    incremental maintenance. Search returns (IDs, scored candidates, stats).
    """

    @classmethod
    def from_cache(cls, cache, directory=None):
        if directory is None:
            build_fast_hdmg(cache['vectormap'], embedding_k=16, semantic_intra_k=8)
            directory = build_semantic_directory(cache)
        return cls(Corpus(cache, directory))

    @classmethod
    def from_vectormap(cls, vectormap, schemas=None):
        entities, locator = defaultdict(list), {}
        for key, entry in vectormap.data.items():
            for group in entry['groups']:
                if len(group.vectors) != len(group.descriptions):
                    raise ValueError('Each vector requires an object description')
                for pos, desc in enumerate(group.descriptions):
                    oid, entity = desc['object_id'], desc['entity']
                    if oid in locator:
                        raise ValueError(f'Duplicate object_id: {oid}')
                    locator[oid] = (key, group.group_name, pos)
                    entities[entity].append(oid)
        if not locator:
            raise ValueError('IntentIndex requires a nonempty gallery')
        cache = dict(vectormap=vectormap, entity_objects=dict(entities),
                     object_locator=locator, schemas=schemas or {})
        return cls.from_cache(cache)

    def search(self, query, plan, *, budget=32, score_batch=4, **kwargs):
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if not np.isfinite(query).all() or norm <= 0:
            raise ValueError('Query embedding must be finite and nonzero')
        if query.shape[0] != self.c.vectors.shape[1]:
            raise ValueError('Query embedding dimension differs from gallery')
        return super().search(query / norm, plan, budget=budget,
                              score_batch=score_batch, **kwargs)
