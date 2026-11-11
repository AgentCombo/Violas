#!/usr/bin/env python3
"""Eight retrieval methods with soft entity relevance and a full-corpus oracle."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_multidataset_canonical_bench as bench
import run_guideline_pipeline as core
import run_qdrant_chroma_backends as backends
from storage.utils import cosine_similarity


METHODS = ["vg_hdmg", "vg_rep"] + [
    f"{db}_{route}" for db in ("milvus", "qdrant", "chroma") for route in ("ann", "key")
]
ENTITY_METHODS = ["vg_hdmg", "vg_rep", "milvus_entity", "qdrant_entity", "chroma_entity"]


from violas.retrieval.scoring import Corpus


from violas.retrieval.scoring import select


from violas.retrieval.scoring import representative_scan


def group_search(corpus, query, plan, args, graph):
    semantic = corpus.semantic_scores(plan)
    hard = {(r["key"], r["value"]) for r in plan["hard_filters"]}
    node_count = len(corpus.members)
    scores = np.full(node_count, -np.inf, dtype=np.float32)
    discovered = np.zeros(node_count, dtype=bool)
    frontier = np.zeros(node_count, dtype=bool)
    available = np.ones(node_count, dtype=bool)
    required = plan.get("required_entity")
    if required is not None:
        available &= corpus.node_keys == corpus.key_index[required]
    for token in hard:
        matches = np.zeros(node_count, dtype=bool)
        matches[corpus.token_nodes.get(token, [])] = True
        available &= matches
    consumed = []
    pool = {}
    nav_expansions = 0
    visited_members = 0
    neighbor_reads = 0
    dimension = plan["facet_dimension"]
    preferences = plan["category_preferences"]
    bonus = np.zeros(node_count, dtype=np.float32)

    def discover(nodes):
        nodes = np.asarray(nodes, dtype=np.int32)
        nodes = nodes[~discovered[nodes]]
        if not len(nodes):
            return
        similarities = np.einsum("ij,j->i", corpus.centers[nodes], query)
        scores[nodes] = (
            plan["embedding_weight"] * (1.0 + similarities) / 2.0
            + plan["semantic_weight"] * semantic[corpus.node_keys[nodes]]
        )
        discovered[nodes] = True
        frontier[nodes] = True

    def update_bonus(selected):
        bonus.fill(0)
        selected_counts = Counter(core.label_of(corpus.objects, oid, dimension) for oid in selected)
        for value, pref in preferences.items():
            nodes = corpus.token_nodes.get((dimension, value))
            if nodes is not None and selected_counts[value] < pref["quota"]:
                bonus[nodes] = np.maximum(bonus[nodes], pref["lambda"])

    def consume_best():
        nonlocal visited_members
        candidates = np.flatnonzero(discovered & available)
        if not len(candidates):
            return False
        priorities = scores[candidates] + bonus[candidates]
        tied = candidates[priorities == priorities.max()]
        node = int(tied[np.argmax(scores[tied])])
        available[node] = False
        consumed.append(node)
        members = corpus.members[node]
        visited_members += len(members)
        ids = [i for i in members if corpus.ids[i] not in pool and corpus.eligible(i, plan)]
        pool.update(corpus.scores(ids, query, plan, semantic))
        if preferences:
            update_bonus(select(pool, corpus, plan))
        return True

    update_bonus([])
    if graph:
        preferred = plan["entity_preference"]
        entries = {int(corpus.entries[preferred])}
        if corpus.global_entry is not None:
            entries.add(int(corpus.global_entry))
        tokens = set(hard)
        tokens.update((dimension, value) for value in preferences)
        for token in sorted(tokens):
            nodes = corpus.token_nodes.get(token)
            if nodes is not None:
                candidates = np.sort(nodes[available[nodes]])
                if len(candidates):
                    entries.add(int(candidates[np.argmax(semantic[corpus.node_keys[candidates]])]))
        discover(sorted(entries))
        for step in range(args.nav_budget):
            nodes = np.flatnonzero(frontier)
            if not len(nodes):
                break
            node = int(nodes[np.argmax(scores[nodes] + bonus[nodes])])
            frontier[node] = False
            nav_expansions += 1
            neighbor_reads += len(corpus.edges[node])
            discover(corpus.edges[node])
            target = (step + 1) * args.group_budget // args.nav_budget
            while len(consumed) < target and consume_best():
                pass
    else:
        discover(np.arange(node_count))
    while len(consumed) < args.group_budget and consume_best():
        pass
    selected = select(pool, corpus, plan)
    return selected, pool, {
        "center_scores": int(discovered.sum()), "searched_groups": len(consumed),
        "unique_member_scores": len(pool), "metadata_objects": visited_members,
        "nav_expansions": nav_expansions, "neighbor_reads": neighbor_reads,
        "discovered_entities": [corpus.entities[key] for key in np.unique(corpus.node_keys[discovered])],
        "selected_groups": consumed,
    }


class TunedGraphSearch:
    """Route hard facets using object postings and rank graph nodes with member pivots."""

    def __init__(self, corpus, max_pivots=8):
        self.corpus = corpus
        self.object_nodes = np.full(len(corpus.ids), -1, dtype=np.int32)
        self.pivots = []
        for node, members in enumerate(corpus.members):
            if (self.object_nodes[members] >= 0).any():
                raise ValueError("tuned graph expects disjoint group membership")
            self.object_nodes[members] = node
            vectors = corpus.vectors[members]
            # Pick real members far apart to cover directions missed by the mean.
            chosen = [int(np.argmax(vectors @ corpus.centers[node]))]
            nearest = vectors @ vectors[chosen[0]]
            for _ in range(1, min(max_pivots, len(members))):
                nearest[chosen] = np.inf
                chosen.append(int(np.argmin(nearest)))
                nearest = np.maximum(nearest, vectors @ vectors[chosen[-1]])
            self.pivots.append(vectors[(chosen * max_pivots)[:max_pivots]])
        if (self.object_nodes < 0).any():
            raise ValueError("tuned graph does not cover the entire gallery")
        self.pivots = np.stack(self.pivots)
        postings = defaultdict(list)
        for index, facets in enumerate(corpus.facets):
            for token in facets.items():
                postings[token].append(index)
        self.postings = {
            token: np.asarray(indices, dtype=np.int32) for token, indices in postings.items()
        }
        self.empty = np.empty(0, dtype=np.int32)
        self.entry_nodes = np.asarray(sorted(set(corpus.entries.values())), dtype=np.int32)

    def hard_candidates(self, filters):
        lists = sorted(
            (self.postings.get((row["key"], row["value"]), self.empty) for row in filters),
            key=len,
        )
        indices = lists[0]
        for posting in lists[1:]:
            indices = np.intersect1d(indices, posting, assume_unique=True)
        return indices

    def search(self, query, plan, config):
        corpus = self.corpus
        semantic = corpus.semantic_scores(plan)
        hard = plan["hard_filters"]
        eligible = self.hard_candidates(hard) if hard else None
        if eligible is not None and len(eligible) <= config["exact_limit"]:
            pool = corpus.scores(eligible, query, plan, semantic)
            return select(pool, corpus, plan), pool, {
                "route": "exact_facet_posting", "center_scores": 0,
                "searched_groups": 0, "unique_member_scores": len(pool),
                "metadata_objects": len(eligible), "nav_expansions": 0,
                "neighbor_reads": 0, "selected_groups": [],
            }

        node_count = len(corpus.members)
        scores = np.full(node_count, -np.inf, dtype=np.float32)
        discovered = np.zeros(node_count, dtype=bool)
        frontier = np.zeros(node_count, dtype=bool)
        available = np.ones(node_count, dtype=bool)
        seeds = set(self.entry_nodes.tolist())
        if corpus.global_entry is not None:
            seeds.add(int(corpus.global_entry))
        eligible_mask = None
        if eligible is not None:
            eligible_mask = np.zeros(len(corpus.ids), dtype=bool)
            eligible_mask[eligible] = True
            counts = np.bincount(self.object_nodes[eligible], minlength=node_count)
            available = counts > 0
            # Seed real intersections, including facet matches outside the preferred entity.
            seeded = np.argsort(-counts, kind="stable")[:config["facet_seeds"]]
            seeds.update(int(node) for node in seeded if available[node])

        pivot_count = config["pivots"]

        def discover(nodes):
            nodes = np.asarray(nodes, dtype=np.int32)
            nodes = nodes[~discovered[nodes]]
            if not len(nodes):
                return
            relevance = np.einsum("ij,j->i", corpus.centers[nodes], query)
            if pivot_count:
                pivot_scores = np.einsum("ipj,j->ip", self.pivots[nodes, :pivot_count], query)
                relevance = np.maximum(relevance, pivot_scores.max(axis=1))
            scores[nodes] = (
                plan["embedding_weight"] * (1.0 + relevance) / 2.0
                + plan["semantic_weight"] * semantic[corpus.node_keys[nodes]]
            )
            discovered[nodes] = True
            frontier[nodes] = True

        discover(sorted(seeds))
        nav_expansions = neighbor_reads = 0
        for _ in range(config["nav_budget"]):
            nodes = np.flatnonzero(frontier)
            if not len(nodes):
                break
            # Both eligible and transit nodes can lead to better candidates.
            node = int(nodes[np.argmax(scores[nodes])])
            frontier[node] = False
            nav_expansions += 1
            neighbor_reads += len(corpus.edges[node])
            discover(corpus.edges[node])

        pool = {}
        selected = []
        groups = []
        inspected = 0
        dimension = plan["facet_dimension"]
        preferences = plan["category_preferences"]
        best_by_value = defaultdict(list)
        while len(groups) < config["group_budget"]:
            nodes = np.flatnonzero(discovered & available)
            if not len(nodes):
                break
            priorities = scores.copy()
            if preferences:
                counts = Counter(core.label_of(corpus.objects, oid, dimension) for oid in selected)
                bonus = np.zeros(node_count, dtype=np.float32)
                for value, pref in preferences.items():
                    value_nodes = corpus.token_nodes.get((dimension, value))
                    if value_nodes is not None and counts[value] < pref["quota"]:
                        bonus[value_nodes] = np.maximum(bonus[value_nodes], pref["lambda"])
                priorities += bonus
            ordered = np.lexsort((nodes, -scores[nodes], -priorities[nodes]))
            batch_size = 8 if preferences else config["group_budget"]
            batch = nodes[ordered[:min(batch_size, config["group_budget"] - len(groups))]]
            available[batch] = False
            groups.extend(batch.tolist())
            members = np.concatenate([corpus.members[node] for node in batch])
            inspected += len(members)
            if eligible_mask is not None:
                members = members[eligible_mask[members]]
            batch_scores = corpus.scores(members, query, plan, semantic)
            pool.update(batch_scores)
            if preferences:
                touched = set()
                for oid, score in batch_scores.items():
                    value = core.label_of(corpus.objects, oid, dimension)
                    best_by_value[value].append((oid, score))
                    touched.add(value)
                for value in touched:
                    # A global top-k answer can use at most k members of any value.
                    best_by_value[value] = heapq.nsmallest(
                        plan["top_k"], best_by_value[value], key=lambda item: (-item[1], item[0])
                    )
                prefix = dict(item for items in best_by_value.values() for item in items)
                selected = select(prefix, corpus, plan)
            else:
                selected = heapq.nsmallest(
                    plan["top_k"], pool, key=lambda oid: (-pool[oid], oid)
                )
        return selected, pool, {
            "route": "pivot_graph", "center_scores": int(discovered.sum()),
            "pivot_scores": int(discovered.sum()) * pivot_count,
            "searched_groups": len(groups), "unique_member_scores": len(pool),
            "metadata_objects": inspected, "nav_expansions": nav_expansions,
            "neighbor_reads": neighbor_reads, "selected_groups": groups,
        }


class VerifiedGraphSearch:
    """Use HDMG to seed the answer and centroid bounds to certify the remaining groups."""

    def __init__(self, graph, *, sparse_verification=False):
        self.graph = graph
        self.corpus = graph.corpus
        self.centers = self.corpus.centers.astype(np.float64)
        self.radii = np.asarray([
            np.linalg.norm(self.corpus.vectors[members] - self.centers[node], axis=1).max()
            for node, members in enumerate(self.corpus.members)
        ])
        self.sparse_vectors = None
        if sparse_verification:
            from scipy.sparse import csr_matrix

            self.sparse_vectors = csr_matrix(self.corpus.vectors)

    def select_from_arrays(self, indices, values, plan):
        k = plan["top_k"]
        dimension = plan["facet_dimension"]
        preferences = plan["category_preferences"]

        def top_positions(positions):
            if len(positions) <= k:
                return positions
            scores = values[positions]
            threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
            higher = positions[scores > threshold]
            tied = positions[scores == threshold]
            ties_by_id = tied[np.argsort(indices[tied], kind="stable")]
            return np.concatenate((higher, ties_by_id[:k - len(higher)]))

        if preferences:
            # Unrewarded values can share one bucket. At most k items can win
            # from any rewarded value, even when its quota is larger than k.
            buckets = np.zeros(len(self.corpus.ids), dtype=np.int32)
            for bucket, value in enumerate(preferences, start=1):
                posting = self.graph.postings.get((dimension, value))
                if posting is not None:
                    buckets[posting] = bucket
            labels = buckets[indices]
            shortlist = np.concatenate([top_positions(np.flatnonzero(labels == v)) for v in np.unique(labels)]) if len(labels) else self.graph.empty
        else:
            shortlist = top_positions(np.arange(len(indices)))
        prefix = {self.corpus.ids[indices[i]]: float(values[i]) for i in shortlist}
        return select(prefix, self.corpus, plan), prefix

    def search(self, query, plan, config):
        corpus = self.corpus
        graph = self.graph
        semantic = corpus.semantic_scores(plan)
        hard = plan["hard_filters"]
        eligible = graph.hard_candidates(hard) if hard else None
        if eligible is not None and len(eligible) <= config["exact_limit"]:
            values = corpus.score_values(eligible, query, plan, semantic)
            selected, prefix = self.select_from_arrays(eligible, values, plan)
            return selected, prefix, {
                "route": "exact_facet_posting", "center_scores": 0, "searched_groups": 0,
                "unique_member_scores": len(eligible), "metadata_objects": len(eligible),
                "nav_expansions": 0, "neighbor_reads": 0, "selected_groups": [],
            }

        seed_config = {**config, "nav_budget": 12, "group_budget": 8, "pivots": min(config["pivots"], 4)}
        seeds, seed_scores, stats = graph.search(query, plan, seed_config)
        preferences = plan["category_preferences"]
        dimension = plan["facet_dimension"]
        threshold = -np.inf
        if len(seeds) == plan["top_k"]:
            last = seeds[-1]
            value = core.label_of(corpus.objects, last, dimension)
            weight, quota = core.category_preference(preferences, value, dimension, plan["lambda"])
            count = sum(core.label_of(corpus.objects, oid, dimension) == value for oid in seeds)
            threshold = seed_scores[last] + (weight if count <= quota else 0.0)

        # Cauchy–Schwarz gives dot(q,x) <= dot(q,center) + norm(q) * radius.
        query64 = query.astype(np.float64)
        similarity_bound = np.einsum("ij,j->i", self.centers, query64)
        similarity_bound += float(np.linalg.norm(query64)) * self.radii
        bound = plan["embedding_weight"] * (1.0 + similarity_bound) / 2.0
        bound += plan["semantic_weight"] * semantic[corpus.node_keys]
        bonuses = np.zeros(len(corpus.members))
        for value, pref in preferences.items():
            nodes = corpus.token_nodes.get((dimension, value))
            if nodes is not None and pref["quota"] > 0:
                bonuses[nodes] = np.maximum(bonuses[nodes], pref["lambda"])
        candidates = bound + bonuses >= threshold - 1e-5
        candidates[stats["selected_groups"]] = False
        if eligible is None:
            verification = np.flatnonzero(candidates[graph.object_nodes])
        else:
            verification = eligible[candidates[graph.object_nodes[eligible]]]
        seed_indices = np.fromiter((corpus.locator[oid] for oid in seed_scores), dtype=np.int32)
        seed_values = np.fromiter(seed_scores.values(), dtype=np.float32)
        # Near-full verification is cheaper as a contiguous pass than a gathered copy.
        dense_pass = len(verification) * 10 >= len(corpus.ids) * 9
        if self.sparse_vectors is not None:
            if dense_pass:
                embedding_scores = (self.sparse_vectors @ query)[verification]
            else:
                embedding_scores = self.sparse_vectors[verification] @ query
            values = corpus.mixed_values(embedding_scores, corpus.entity_ids[verification], plan, semantic)
        elif dense_pass:
            values = corpus.score_values(None, query, plan, semantic)[verification]
        else:
            values = corpus.score_values(verification, query, plan, semantic)
        selected, prefix = self.select_from_arrays(
            np.concatenate((seed_indices, verification)), np.concatenate((seed_values, values)), plan
        )
        groups = stats["selected_groups"] + np.unique(graph.object_nodes[verification]).tolist()
        stats.update({
            "route": "verified_graph", "bounded_groups": len(candidates),
            "verification_scores": len(corpus.ids) if dense_pass else len(verification),
            "verification_candidates": len(verification), "dense_pass": dense_pass,
            "verification_kernel": "csr" if self.sparse_vectors is not None else "dense",
            "searched_groups": len(groups),
            "unique_member_scores": len(corpus.ids) if dense_pass else len(seed_indices) + len(verification),
            "metadata_objects": stats["metadata_objects"] + len(verification),
            "selected_groups": groups,
        })
        return selected, prefix, stats


def route_keys(corpus, semantic, budget):
    order = sorted(range(len(corpus.entities)), key=lambda i: (-semantic[i], corpus.entities[i]))
    return [corpus.entities[i] for i in order[:budget]]


def db_search(db, route, client, corpus, query, plan, args, facets, *, push_hard_facets=True,
              all_matches=False, profile=False):
    start = time.perf_counter_ns() if profile else 0
    semantic = corpus.semantic_scores(plan)
    if route == "entity":
        keys = [plan["entity_preference"]]
    else:
        keys = route_keys(corpus, semantic, args.key_budget) if route == "key" else None
    hard = plan["hard_filters"] if facets and push_hard_facets else []
    index_tag = getattr(args, "index_tag", None) or args.dataset
    if all_matches and db not in {"qdrant", "chroma"}:
        raise ValueError("filter-only retrieval is implemented for Qdrant and Chroma")
    if db == "milvus":
        filters = []
        if keys:
            filters.append("entity in " + json.dumps(keys))
        if hard:
            tokens = [f"{r['key']}={r['value']}" for r in hard]
            filters.append("ARRAY_CONTAINS_ALL(facet_tokens, " + json.dumps(tokens) + ")")
        response = client.search(
            collection_name=f"{index_tag}_array", data=[query.tolist()], anns_field="vector",
            filter=" and ".join(filters), limit=args.candidate_k,
            output_fields=["object_id"] + (["facet_tokens"] if facets else []),
            search_params={"metric_type": "COSINE", "params": {"nprobe": args.nprobe}},
        )
        ids = [hit["entity"]["object_id"] for hit in response[0]]
    elif db == "qdrant":
        from qdrant_client import models
        conditions = []
        if keys:
            conditions.append(models.FieldCondition(key="entity", match=models.MatchAny(any=keys)))
        conditions.extend(models.FieldCondition(key="facet_tokens", match=models.MatchValue(value=f"{r['key']}={r['value']}")) for r in hard)
        query_filter = models.Filter(must=conditions) if conditions else None
        if all_matches:
            ids, offset = [], None
            while True:
                points, offset = client.scroll(
                    collection_name=f"soft_{index_tag}", scroll_filter=query_filter,
                    offset=offset, limit=len(corpus.ids), with_payload=["object_id"], with_vectors=False,
                )
                ids.extend(point.payload["object_id"] for point in points)
                if offset is None:
                    break
        else:
            response = client.query_points(
                collection_name=f"soft_{index_tag}", query=query.tolist(), query_filter=query_filter,
                limit=args.candidate_k, search_params=models.SearchParams(hnsw_ef=args.ef),
                with_payload=["object_id"] + (["facet_tokens"] if facets else []), with_vectors=False,
            )
            ids = [hit.payload["object_id"] for hit in response.points]
    else:
        predicates = []
        if keys:
            predicates.append({"entity": {"$in": keys}})
        predicates.extend({f"facet__{r['key']}": {"$eq": r["value"]}} for r in hard)
        where = (predicates[0] if len(predicates) == 1 else {"$and": predicates}) if predicates else None
        if all_matches:
            ids = client.get(where=where, include=[])["ids"]
        else:
            response = client.query(query_embeddings=[query.tolist()], n_results=args.candidate_k, where=where, include=["distances"])
            ids = response["ids"][0]
    fetched = time.perf_counter_ns() if profile else 0
    ids = list(dict.fromkeys(ids))
    indices = [corpus.locator[oid] for oid in ids]
    if facets:
        indices = [i for i in indices if corpus.eligible(i, plan)]
    if all_matches and len(indices) == len(corpus.ids):
        values = corpus.score_values(None, query, plan, semantic)
        pool = {oid: float(value) for oid, value in zip(corpus.ids, values)}
    else:
        pool = corpus.scores(indices, query, plan, semantic)
    selected = select(pool, corpus, plan, facets)
    stats = {
        "candidate_count": len(ids), "unique_member_scores": len(pool),
        "routed_keys": keys, "uses_facets": facets,
    }
    if route == "entity":
        stats["candidate_ids"] = ids
    if profile:
        stats["fetch_ms"] = (fetched - start) / 1e6
        stats["rerank_ms"] = (time.perf_counter_ns() - fetched) / 1e6
    return selected, pool, stats


def setup_qdrant(args, corpus):
    from qdrant_client import QdrantClient, models
    client = QdrantClient(url=args.qdrant_url, timeout=120)
    index_tag = getattr(args, "index_tag", None) or args.dataset
    collection = f"soft_{index_tag}"
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=corpus.vectors.shape[1], distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=10),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
        )
        for start in range(0, len(corpus.ids), 500):
            points = [models.PointStruct(id=i, vector=corpus.vectors[i].tolist(), payload={
                "object_id": corpus.ids[i], "entity": corpus.entities[corpus.entity_ids[i]],
                "facet_tokens": [f"{k}={v}" for k, v in corpus.facets[i].items()],
            }) for i in range(start, min(start + 500, len(corpus.ids)))]
            client.upsert(collection_name=collection, points=points, wait=True)
        for field in ("entity", "facet_tokens"):
            client.create_payload_index(collection, field, models.PayloadSchemaType.KEYWORD, wait=True)
    info = client.get_collection(collection)
    if info.points_count != len(corpus.ids):
        raise RuntimeError("Qdrant collection size differs from gallery")
    deadline = time.monotonic() + 120
    while info.indexed_vectors_count < len(corpus.ids) and time.monotonic() < deadline:
        print(f"INDEX {args.dataset} {info.indexed_vectors_count}/{len(corpus.ids)}", flush=True)
        time.sleep(2)
        info = client.get_collection(collection)
    if not info.indexed_vectors_count:
        raise RuntimeError("Qdrant has no indexed vectors")
    return client, {"deployment": "Qdrant server", "url": args.qdrant_url, "points": info.points_count, "indexed_vectors": info.indexed_vectors_count}


def make_plan(row, directory, number):
    intent = row["canonical_intent"]
    base = core.materialize_plan_preferences(core.resolve_plan(intent), directory)
    preferred = base.pop("entity_scope")
    weight = float(intent.get("semantic_weight", (0.15, 0.35, 0.55)[number % 3]))
    embedding = float(intent.get("embedding_weight", 1.0 - weight))
    if not 0 <= weight <= 1 or not 0 <= embedding <= 1 or not math.isclose(weight + embedding, 1):
        raise ValueError("semantic and embedding weights must be nonnegative and sum to one")
    return {**base, "entity_preference": preferred, "entity_mode": "soft", "search_scope": "all_entities",
            "semantic_weight": weight, "embedding_weight": embedding}


def result(ids, scores, stats, ms, oracle_ids, oracle_value, plan, corpus, eligible_count):
    valid = [oid for oid in ids if corpus.eligible(corpus.locator[oid], plan)]
    value = core.objective(valid, scores, corpus.objects, plan["facet_dimension"], plan["lambda"], plan["category_preferences"])
    return {
        "results": core.result_rows(ids, scores, corpus.objects), "latency": bench.latency([ms]), "stats": stats,
        "metrics": {
            "objective": value, "objective_ratio": value / oracle_value if oracle_value else None,
            "oracle_object_recall": len(set(valid) & set(oracle_ids)) / len(oracle_ids) if oracle_ids else None,
            "hard_condition_validity": len(valid) / len(ids) if ids else 1.0,
            "fill_rate": len(valid) / min(plan["top_k"], eligible_count) if eligible_count else None,
            "cross_entity_rate": sum(corpus.objects[oid]["description"]["entity"] != plan["entity_preference"] for oid in ids) / len(ids) if ids else 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--hdmg-cache", type=Path, required=True)
    parser.add_argument("--directory-cache", type=Path, required=True)
    parser.add_argument("--module-path", action="append", default=[])
    parser.add_argument("--candidate-k", type=int, default=80)
    parser.add_argument("--key-budget", type=int, default=3)
    parser.add_argument("--group-budget", type=int, default=8)
    parser.add_argument("--representative-kernel", choices=("cosine", "scalar", "batch"), default="cosine")
    parser.add_argument("--nav-budget", type=int, default=12)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--ef", type=int, default=80)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:16333")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--smoke", action="store_true", help="Run one query per intent type.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--frozen-report", type=Path, help="Replay query bindings and resolved plans from a prior result")
    parser.add_argument("--index-tag", help="Use separate indexes for a relabeled gallery")
    parser.add_argument("--experiment3-only", action="store_true",
                        help="Compare VG with preferred-entity vector recall and local facet selection")
    args = parser.parse_args()
    if args.output_root is None:
        suffix = {"cosine": "online_cosine", "scalar": "scalar_rep", "batch": "batch_rep"}
        args.output_root = Path("artifacts") / f"six_dataset_soft_entity_{suffix[args.representative_kernel]}"
    if min(args.candidate_k, args.key_budget, args.group_budget, args.nav_budget, args.limit) < 1:
        parser.error("limits and search budgets must be positive")
    from pymilvus import MilvusClient
    source = bench.load_pickle(args.source_cache, args.module_path)
    cache = bench.load_pickle(args.hdmg_cache, args.module_path)
    directory = bench.load_pickle(args.directory_cache, args.module_path)
    corpus = Corpus(cache, directory)
    workload_source = args.frozen_report or Path(f"artifacts/six_dataset_five_system/{args.dataset}/results.json")
    previous = json.loads(workload_source.read_text(encoding="utf-8"))
    if args.frozen_report and previous["dataset"] != args.dataset:
        raise ValueError("frozen report dataset differs from the gallery")
    numbered = list(enumerate(previous["rows"]))
    if args.smoke:
        first_by_type = {}
        for number, row in numbered:
            first_by_type.setdefault(row["query_type"], (number, row))
        numbered = list(first_by_type.values())
    numbered = numbered[:args.limit]
    records = [row for _, row in numbered]
    queries = bench.choose_queries(previous["rows"], source, set(corpus.ids))
    output = args.output_root / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    plans = ([row["resolved_plan"] for row in records] if args.frozen_report
             else [make_plan(row, directory, i) for i, row in numbered])
    workload = {"dataset": args.dataset, "weight_source": "intent weights if supplied, otherwise fixed 0.15/0.35/0.55 cycle by original workload position", "records": [
        {"id": row["id"], "query_object": row["query_object"], "intent": plan} for row, plan in zip(records, plans)
    ]}
    workload_text = json.dumps(workload, ensure_ascii=False, indent=2)
    (output / "workload.json").write_text(workload_text, encoding="utf-8")
    qdrant, qinfo = setup_qdrant(args, corpus)
    index_tag = args.index_tag or args.dataset
    milvus = MilvusClient(str(Path(f"cache/five_system/{index_tag}_milvus_array.db").resolve()))
    milvus.load_collection(f"{index_tag}_array")
    chroma_client, chroma = backends.open_chroma(Path(f"cache/five_system/{index_tag}_chroma"), f"{index_tag}_chroma")
    clients = {"milvus": milvus, "qdrant": qdrant, "chroma": chroma}
    rows = []
    if args.experiment3_only:
        methods = [(name, 3) for name in ENTITY_METHODS]
    else:
        methods = [(name, exp) for exp in (1, 2) for name in METHODS if exp == 1 or not name.startswith("vg_")]
    try:
        for number, (record, plan) in enumerate(zip(records, plans)):
            query_row = queries[record["id"]]
            if query_row["object_id"] != record["query_object"]["object_id"]:
                raise AssertionError("query binding changed")
            query = query_row["normalized"]
            if args.representative_kernel == "cosine":
                raw_vector = source["test_data"][record["entity_hint"]]["vectors"][query_row["test_index"]]
                rep_query = np.asarray(raw_vector, dtype=np.float32).reshape(-1)
            else:
                rep_query = query
            outputs = {}
            offset = number % len(methods)
            for name, exp in methods[offset:] + methods[:offset]:
                start = time.perf_counter_ns()
                if name == "vg_rep" and args.representative_kernel != "batch":
                    value = representative_scan(
                        corpus, rep_query, plan, args.group_budget,
                        online_cosine=args.representative_kernel == "cosine",
                    )
                elif name.startswith("vg_"):
                    value = group_search(corpus, query, plan, args, name == "vg_hdmg")
                else:
                    db, route = name.split("_")
                    value = db_search(db, route, clients[db], corpus, query, plan, args, exp != 1,
                                      push_hard_facets=exp != 3)
                outputs[f"exp{exp}_{name}"] = (value, (time.perf_counter_ns() - start) / 1e6)
            if not args.experiment3_only:
                for name in ("vg_hdmg", "vg_rep"):
                    outputs[f"exp2_{name}"] = outputs[f"exp1_{name}"]
            # Reference calculation occurs only after measured searches.
            eligible = [i for i in range(len(corpus.ids)) if corpus.eligible(i, plan)]
            semantic = corpus.semantic_scores(plan)
            oracle_scores = corpus.scores(eligible, query, plan, semantic)
            oracle_ids = select(oracle_scores, corpus, plan)
            oracle_value = core.objective(oracle_ids, oracle_scores, corpus.objects, plan["facet_dimension"], plan["lambda"], plan["category_preferences"])
            methods_out = {name: result(ids, scores, stats, ms, oracle_ids, oracle_value, plan, corpus, len(eligible)) for name, ((ids, scores, stats), ms) in outputs.items()}
            row = {"id": record["id"], "entity_hint": record["entity_hint"], "query_type": record["query_type"],
                   "canonical_intent": plan, "resolved_plan": plan, "query_object": record["query_object"],
                   "eligible_count": len(eligible), "eligible_entity_count": len(set(corpus.entity_ids[eligible].tolist())),
                   "oracle": {"results": core.result_rows(oracle_ids, oracle_scores, corpus.objects), "objective": oracle_value},
                   "methods": methods_out}
            if "english" in record:
                row["english"] = record["english"]
            if previous["modality"] == "text":
                text_roots = {
                    "20news": Path("dataset/20_newsgroups-full"),
                    "ohsumed": Path("dataset/ohsumed-all-docs/ohsumed-all-full"),
                    "yahoo": Path("dataset/Yahoo!-answer_processed"),
                }
                for method in list(methods_out.values()) + [row["oracle"]]:
                    for item in method["results"]:
                        desc = corpus.objects[item["object_id"]]["description"]
                        path = bench.text_path(text_roots[args.dataset], desc["entity"], desc["img_path"])
                        item["preview"] = bench.preview(path)
            row["oracle"]["cross_entity_rate"] = sum(
                corpus.objects[oid]["description"]["entity"] != plan["entity_preference"]
                for oid in oracle_ids
            ) / len(oracle_ids) if oracle_ids else 0.0
            rows.append(row)
            prefix = "exp3" if args.experiment3_only else "exp1"
            ratios = [methods_out[f"{prefix}_{method}"]["metrics"]["objective_ratio"] for method in ("vg_hdmg", "vg_rep")]
            hdmg_ratio, rep_ratio = (f"{ratio:.4f}" if ratio is not None else "n/a" for ratio in ratios)
            print(f"QUERY {args.dataset} {number+1}/{len(records)} HDMG={hdmg_ratio} rep={rep_ratio} "
                  f"ms={outputs[f'{prefix}_vg_hdmg'][1]:.3f}/{outputs[f'{prefix}_vg_rep'][1]:.3f}", flush=True)
        names = list(rows[0]["methods"])
        aggregate = {}
        for name in names:
            metrics = rows[0]["methods"][name]["metrics"]
            aggregate[name] = {"mean_" + field: bench.mean_metric(rows, name, field) for field in metrics}
            aggregate[name]["latency"] = bench.latency([row["methods"][name]["latency"]["p50_ms"] for row in rows])
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update({"measurement": "no warmup; one measured execution per query/method; rotating order", "qdrant": qinfo,
                       "milvus": "Milvus Lite IVF_FLAT", "chroma": "Chroma persistent HNSW",
                       "workload_sha256": hashlib.sha256(workload_text.encode()).hexdigest(),
                       "objective": "sum(w_emb * (1+cos(q,x))/2 + w_sem * (1+cos(key_q,key_x))/2) + soft facet quotas"})
        report_format = "soft-entity-preferred-entity-v1" if args.experiment3_only else "soft-entity-eight-methods-v1"
        report = {"format": report_format, "dataset": args.dataset, "modality": previous["modality"],
                  "config": config, "aggregate": {"overall": aggregate}, "rows": rows}
        pending = output / "results.pending.json"
        pending.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(output / "results.json")
        print(f"COMPLETE {args.dataset} {output}", flush=True)
    finally:
        milvus.close()
        qdrant.close()


if __name__ == "__main__":
    main()
