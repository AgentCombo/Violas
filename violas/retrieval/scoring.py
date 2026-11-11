from __future__ import annotations
from collections import defaultdict, Counter
from typing import Any, Iterable
import heapq
import math
import numpy as np

from .objects import make_object_map
from . import objective as core
from violas.storage.utils import cosine_similarity


class Corpus:
    def __init__(self, cache, directory):
        self.objects = make_object_map(cache)
        self.ids = sorted(self.objects)
        self.locator = {oid: i for i, oid in enumerate(self.ids)}
        self.vectors = np.stack([self.objects[oid]["normalized"] for oid in self.ids])
        self.entities = sorted(cache["entity_objects"])
        self.key_index = {key: i for i, key in enumerate(self.entities)}
        self.raw_keys = np.stack([np.asarray(cache["vectormap"].key_vectors[key]).reshape(-1) for key in self.entities])
        norms = np.linalg.norm(self.raw_keys, axis=1, keepdims=True)
        if not np.isfinite(self.raw_keys).all() or (norms <= 0).any():
            raise ValueError("invalid entity key vectors")
        self.keys = (self.raw_keys / norms).astype(np.float32)
        self.entity_ids = np.array([self.key_index[self.objects[oid]["description"]["entity"]] for oid in self.ids])
        self.facets = [self.objects[oid]["description"].get("facet_values", {}) for oid in self.ids]
        self.centers = directory["centers"]
        self.members = []
        self.node_keys = []
        self.node_tokens = []
        self.value_nodes = defaultdict(list)
        for node, ids in enumerate(directory["node_members"]):
            members = np.array([self.locator[oid] for oid in ids], dtype=np.int32)
            keys = set(self.entity_ids[members].tolist())
            if len(keys) != 1:
                raise ValueError("a micro-cluster spans multiple entity groups")
            self.node_keys.append(next(iter(keys)))
            self.members.append(members)
            counts = Counter((k, v) for i in members for k, v in self.facets[i].items())
            self.node_tokens.append(set(counts))
            for token, count in counts.items():
                self.value_nodes[token].append((count, node))
        self.node_keys = np.asarray(self.node_keys, dtype=np.int32)
        self.token_nodes = {
            token: np.asarray([node for _, node in rows], dtype=np.int32)
            for token, rows in self.value_nodes.items()
        }
        vm = cache["vectormap"]
        self.entries = dict(vm._hdmg_key_entries)
        self.global_entry = vm._hdmg_global_entry
        self.edges = [tuple(sorted(set(a) | set(b))) for a, b in zip(vm._hdmg_embedding_edges, vm._hdmg_semantic_edges)]

    def semantic_scores(self, plan):
        key = self.keys[self.key_index[plan["entity_preference"]]]
        return np.clip((1.0 + np.einsum("ij,j->i", self.keys, key)) / 2.0, 0.0, 1.0)

    def eligible(self, index, plan):
        required = plan.get("required_entity")
        if required is not None and self.entity_ids[index] != self.key_index[required]:
            return False
        values = self.facets[index]
        return all(values.get(row["key"]) == row["value"] for row in plan["hard_filters"])

    def scores(self, indices, query, plan, semantic):
        indices = np.asarray(indices, dtype=np.int32)
        values = self.score_values(indices, query, plan, semantic)
        return {self.ids[i]: float(value) for i, value in zip(indices, values)}

    def score_values(self, indices, query, plan, semantic):
        # Batched form of the normalized dot-product objective.
        vectors = self.vectors if indices is None else self.vectors[indices]
        keys = self.entity_ids if indices is None else self.entity_ids[indices]
        embeddings = np.einsum("ij,j->i", vectors, query)
        return self.mixed_values(embeddings, keys, plan, semantic)

    @staticmethod
    def mixed_values(embeddings, keys, plan, semantic):
        values = plan["embedding_weight"] * (1.0 + embeddings) / 2.0
        values += plan["semantic_weight"] * semantic[keys]
        return values


def select(scored, corpus, plan, facets=True):
    if not facets:
        return sorted(scored, key=lambda oid: (-scored[oid], oid))[:plan["top_k"]]
    return core.select_exact(scored, corpus.objects, plan["facet_dimension"], plan["top_k"], plan["lambda"], plan["category_preferences"])


def representative_scan(corpus, query, plan, group_budget, *, online_cosine=False):
    """Scan representatives and selected members using the requested scalar kernel."""
    keys = corpus.raw_keys if online_cosine else corpus.keys
    preferred = keys[corpus.key_index[plan["entity_preference"]]]

    def cosine_score(a, b):
        return (1.0 + float(cosine_similarity(a, b))) / 2.0

    member_score = cosine_score if online_cosine else core.score_object
    semantic = [
        min(1.0, max(0.0, member_score(preferred, key)))
        for key in keys
    ]
    embedding_weight = plan["embedding_weight"]
    semantic_weight = plan["semantic_weight"]

    hard = {(row["key"], row["value"]) for row in plan["hard_filters"]}
    required = plan.get("required_entity")
    required_index = corpus.key_index[required] if required is not None else None
    dimension = plan["facet_dimension"]
    preferences = plan["category_preferences"]
    remaining = {}
    for node, center in enumerate(corpus.centers):
        if required_index is not None and corpus.node_keys[node] != required_index:
            continue
        if online_cosine:
            # Centroid magnitude is part of the existing representative score.
            # Only the query is unit-length in this dot product.
            similarity = float(np.dot(query, center)) / float(np.linalg.norm(query))
            center_score = (1.0 + similarity) / 2.0
        else:
            center_score = core.score_object(query, center)
        score = embedding_weight * center_score + semantic_weight * semantic[corpus.node_keys[node]]
        if hard <= corpus.node_tokens[node]:
            remaining[node] = score

    pool = {}
    selected = []
    groups = []
    visited_members = 0
    while remaining and len(groups) < group_budget:
        counts = Counter(core.label_of(corpus.objects, oid, dimension) for oid in selected)
        rewards = {
            (dimension, value): pref["lambda"]
            for value, pref in preferences.items()
            if counts[value] < pref["quota"]
        }

        def priority(node):
            bonus = max(
                (weight for token, weight in rewards.items() if token in corpus.node_tokens[node]),
                default=0.0,
            )
            score = remaining[node]
            return (score + bonus, score, -node)

        node = max(remaining, key=priority if rewards else remaining.get)
        del remaining[node]
        groups.append(node)
        visited_members += len(corpus.members[node])
        for index in corpus.members[node]:
            oid = corpus.ids[index]
            if oid not in pool and corpus.eligible(index, plan):
                vector = corpus.objects[oid]["vector"] if online_cosine else corpus.vectors[index]
                pool[oid] = (
                    embedding_weight * member_score(query, vector)
                    + semantic_weight * semantic[corpus.entity_ids[index]]
                )
        if preferences:
            selected = select(pool, corpus, plan)

    return select(pool, corpus, plan), pool, {
        "center_scores": len(corpus.centers) if required_index is None else int((corpus.node_keys == required_index).sum()),
        "searched_groups": len(groups),
        "unique_member_scores": len(pool),
        "metadata_objects": visited_members,
        "nav_expansions": 0,
        "neighbor_reads": 0,
        "discovered_entities": corpus.entities,
        "selected_groups": groups,
        "representative_kernel": "online_cosine" if online_cosine else "scalar_dot",
    }
