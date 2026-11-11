#!/usr/bin/env python3
"""Build the KV-aware Caltech train corpus and compare three retrieval paths.

Methods:
  * ground_truth: exhaustive cosine scan after exact facet filtering
  * vectorgroup: entity/facet filtering, representative-group pruning, exact rerank
  * milvus: entity/facet scalar filtering, IVF_FLAT ANN, exact candidate rerank

All methods share the same deterministic diversity selector.  This isolates the
candidate-generation/index approximation from intent parsing and selection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
import os
import pickle
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path("cache/caltech_101_ObjectCategories_ts0.1_rs42_a0.500_folder.pkl"),
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("diversity/v2_caltech99_kv_schemas/assignments.jsonl"),
    )
    parser.add_argument(
        "--schemas",
        type=Path,
        default=Path("diversity/v2_caltech99_kv_schemas/schemas.jsonl"),
    )
    parser.add_argument(
        "--query-gt",
        type=Path,
        default=Path("queries/caltech101_kv_query_gt.json"),
    )
    parser.add_argument(
        "--kv-cache",
        type=Path,
        default=Path("cache/caltech101_kv_train_v1.pkl"),
    )
    parser.add_argument(
        "--milvus-uri",
        type=Path,
        default=Path("cache/caltech101_kv_milvus_ivf.db"),
    )
    parser.add_argument("--collection", default="caltech101_kv_train")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/caltech10_kv_retrieval"),
    )
    parser.add_argument(
        "--local-dataset-root",
        default=r"dataset/caltech-101/101_ObjectCategories",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--vg-group-budget", type=int, default=8)
    parser.add_argument("--vg-diversity-weight", type=float, default=1.0)
    parser.add_argument("--milvus-nlist", type=int, default=128)
    parser.add_argument("--milvus-nprobe", type=int, default=16)
    parser.add_argument("--force-build", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_id_from_path(path: str) -> str:
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    if len(parts) < 2:
        raise ValueError(f"cannot derive object_id from {path!r}")
    return f"{parts[-2]}/{parts[-1]}"


def entity_for_physical_key(key: str, entities: set[str]) -> str | None:
    matches = [entity for entity in entities if key == entity or key.startswith(entity + "-")]
    return max(matches, key=len) if matches else None


def compact_quality(attributes: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {
            "confidence": round(float(value.get("confidence", 0.0)), 6),
            "margin": round(float(value.get("margin", 0.0)), 6),
            "uncertain": bool(value.get("uncertain", False)),
        }
        for key, value in attributes.items()
    }


def build_kv_cache(args: argparse.Namespace) -> dict[str, Any]:
    print(f"BUILD VectorGroup KV cache from {args.source_cache}", flush=True)
    schema_rows = read_jsonl(args.schemas)
    schemas = {row["entity"]: row for row in schema_rows}
    entities = set(schemas)
    assignments_all = read_jsonl(args.assignments)
    assignments = {
        row["object_id"]: row
        for row in assignments_all
        if row.get("split") == "train" and row.get("entity") in entities
    }

    with args.source_cache.open("rb") as handle:
        source = pickle.load(handle)
    vectormap = source["vectormap"]
    vectormap.data = {
        key: entry
        for key, entry in vectormap.data.items()
        if entity_for_physical_key(key, entities) is not None
    }

    entity_objects: dict[str, set[str]] = defaultdict(set)
    postings: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    object_locator: dict[str, tuple[str, str, int]] = {}
    missing: list[str] = []
    duplicate: list[str] = []

    for physical_key, entry in vectormap.data.items():
        entity = entity_for_physical_key(physical_key, entities)
        assert entity is not None
        entry["metadata"]["entity"] = entity
        entry["metadata"]["schema_version"] = schemas[entity].get(
            "schema_version", "diversity-kv-v2"
        )
        for group in entry["groups"]:
            for vector_idx, description in enumerate(group.descriptions):
                object_id = object_id_from_path(description["img_path"])
                assignment = assignments.get(object_id)
                if assignment is None:
                    missing.append(object_id)
                    continue
                if object_id in object_locator:
                    duplicate.append(object_id)
                    continue
                attributes = assignment["attributes"]
                facet_values = {
                    key: value["value"] for key, value in attributes.items()
                }
                description.update(
                    {
                        "object_id": object_id,
                        "entity": entity,
                        "facet_values": facet_values,
                        "facet_quality": compact_quality(attributes),
                        "uncertain": bool(assignment.get("uncertain", False)),
                        "facet_source": assignment.get("source", "prototype"),
                        "schema_version": assignment.get(
                            "schema_version", "diversity-kv-v2"
                        ),
                    }
                )
                object_locator[object_id] = (
                    physical_key,
                    group.group_name,
                    vector_idx,
                )
                entity_objects[entity].add(object_id)
                for facet, value in facet_values.items():
                    postings[entity][facet][value].add(object_id)

    if missing or duplicate:
        raise RuntimeError(
            f"cache annotation mismatch: missing={len(missing)} duplicate={len(duplicate)}"
        )
    if len(object_locator) != len(assignments):
        assignment_only = set(assignments) - set(object_locator)
        raise RuntimeError(
            f"train count mismatch: vectormap={len(object_locator)} "
            f"assignments={len(assignments)} assignment_only={len(assignment_only)}"
        )

    # Convert nested defaultdict/set structures to stable plain dictionaries.
    postings_plain = {
        entity: {
            facet: {
                value: tuple(sorted(ids)) for value, ids in values.items()
            }
            for facet, values in facets.items()
        }
        for entity, facets in postings.items()
    }
    entity_objects_plain = {
        entity: tuple(sorted(ids)) for entity, ids in entity_objects.items()
    }
    # Persistent group-level routing metadata.  This is derived entirely from
    # the existing train assignments: no VLM/API call is required.  The query
    # path can inspect which facet values a representative group can cover
    # without walking the whole VectorMap first.
    group_routes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for physical_key, entry in vectormap.data.items():
        entity = entry["metadata"]["entity"]
        for group in entry["groups"]:
            object_ids: list[str] = []
            value_counts: dict[str, dict[str, int]] = defaultdict(
                lambda: defaultdict(int)
            )
            for description in group.descriptions:
                object_id = description.get("object_id")
                if not object_id:
                    continue
                object_ids.append(object_id)
                for facet, value in description["facet_values"].items():
                    value_counts[facet][value] += 1
            if not object_ids:
                continue
            group_routes[entity].append(
                {
                    "physical_key": physical_key,
                    "group_name": group.group_name,
                    "object_ids": tuple(sorted(object_ids)),
                    "facet_value_counts": {
                        facet: dict(sorted(values.items()))
                        for facet, values in sorted(value_counts.items())
                    },
                }
            )
    group_routes_plain = {
        entity: sorted(
            rows, key=lambda row: (row["physical_key"], row["group_name"])
        )
        for entity, rows in group_routes.items()
    }
    manifest = {
        "format": "caltech101-kv-train-cache-v2",
        "train_object_count": len(object_locator),
        "entity_count": len(entity_objects_plain),
        "physical_key_count": len(vectormap.data),
        "group_route_count": sum(len(rows) for rows in group_routes_plain.values()),
        "group_route_fields": [
            "physical_key",
            "group_name",
            "object_ids",
            "facet_value_counts",
        ],
        "source_cache": str(args.source_cache),
        "source_cache_sha256": sha256(args.source_cache),
        "assignments": str(args.assignments),
        "assignments_sha256": sha256(args.assignments),
        "schemas": str(args.schemas),
        "schemas_sha256": sha256(args.schemas),
    }
    cache = {
        "manifest": manifest,
        "vectormap": vectormap,
        "schemas": schemas,
        "entity_objects": entity_objects_plain,
        "postings": postings_plain,
        "object_locator": object_locator,
        "group_routes": group_routes_plain,
    }
    args.kv_cache.parent.mkdir(parents=True, exist_ok=True)
    with args.kv_cache.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"DONE VectorGroup cache: objects={len(object_locator)} "
        f"entities={len(entity_objects_plain)} physical_keys={len(vectormap.data)} "
        f"path={args.kv_cache}",
        flush=True,
    )
    return cache


def load_or_build_kv_cache(args: argparse.Namespace) -> dict[str, Any]:
    if args.kv_cache.exists() and not args.force_build:
        print(f"LOAD VectorGroup KV cache: {args.kv_cache}", flush=True)
        with args.kv_cache.open("rb") as handle:
            return pickle.load(handle)
    return build_kv_cache(args)


from violas.retrieval.objects import iter_objects


from violas.retrieval.objects import make_object_map


def build_milvus(args: argparse.Namespace, cache: dict[str, Any], objects: dict[str, Any]) -> None:
    from pymilvus import DataType, MilvusClient

    uri = args.milvus_uri.resolve()
    manifest_path = args.milvus_uri.with_name(args.milvus_uri.name + ".manifest.json")
    if uri.exists() and manifest_path.exists() and not args.force_build:
        print(f"LOAD Milvus Lite cache: {uri}", flush=True)
        return
    if uri.exists():
        resolved = uri.resolve()
        cache_root = (PROJECT_ROOT / "cache").resolve()
        if cache_root not in resolved.parents:
            raise RuntimeError(f"refusing to remove Milvus path outside cache: {resolved}")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
    manifest_path.unlink(missing_ok=True)
    uri.parent.mkdir(parents=True, exist_ok=True)

    print(f"BUILD Milvus Lite IVF_FLAT: {uri}", flush=True)
    client = MilvusClient(str(uri))
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="object_id", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="entity", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(
        field_name="facet_tokens",
        datatype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=16,
        max_length=160,
    )
    dimension = len(next(iter(objects.values()))["vector"])
    schema.add_field(
        field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension
    )
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": args.milvus_nlist},
    )
    client.create_collection(
        collection_name=args.collection,
        schema=schema,
        index_params=index_params,
    )
    scalar_index_params = client.prepare_index_params()
    scalar_index_params.add_index(
        field_name="entity", index_type="INVERTED", index_name="entity_inverted"
    )
    client.create_index(
        collection_name=args.collection, index_params=scalar_index_params
    )

    rows: list[dict[str, Any]] = []
    inserted = 0
    for numeric_id, object_id in enumerate(sorted(objects)):
        obj = objects[object_id]
        desc = obj["description"]
        tokens = [
            f"{key}={value}" for key, value in sorted(desc["facet_values"].items())
        ]
        row = {
            "id": numeric_id,
            "object_id": object_id,
            "entity": desc["entity"],
            "facet_tokens": tokens,
            "vector": obj["vector"].astype(np.float32).tolist(),
        }
        rows.append(row)
        if len(rows) >= 500:
            client.insert(args.collection, rows)
            inserted += len(rows)
            rows = []
            print(f"  inserted={inserted}", flush=True)
    if rows:
        client.insert(args.collection, rows)
        inserted += len(rows)
    client.load_collection(args.collection)
    client.close()
    manifest = {
        "format": "caltech101-kv-milvus-v1",
        "collection": args.collection,
        "object_count": inserted,
        "dimension": dimension,
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "nlist": args.milvus_nlist,
        "scalar_indexes": ["entity"],
        "scalar_index_type": "INVERTED",
        "facet_filter_storage": "facet_tokens ARRAY",
        "facet_filter_operator": "ARRAY_CONTAINS_ALL",
        "kv_cache_manifest": cache["manifest"],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"DONE Milvus cache: objects={inserted} path={uri}", flush=True)


def validate_intent(intent: dict[str, Any], schemas: dict[str, Any]) -> None:
    entity = intent["entity"]
    if entity not in schemas:
        raise ValueError(f"unknown entity {entity}")
    facets = {row["id"]: row for row in schemas[entity]["facets"]}
    for condition in intent["must"]:
        key = condition["key"]
        if condition["op"] != "eq" or key not in facets:
            raise ValueError(f"invalid condition {condition}")
        values = {row["id"] for row in facets[key]["values"]}
        if condition["value"] not in values:
            raise ValueError(f"invalid condition value {condition}")
    for key in intent["diversify_by"]:
        if key not in facets:
            raise ValueError(f"invalid diversify facet {key}")


def eligible_ids(cache: dict[str, Any], intent: dict[str, Any]) -> set[str]:
    entity = intent["entity"]
    current = set(cache["entity_objects"][entity])
    for condition in intent["must"]:
        current &= set(
            cache["postings"][entity][condition["key"]][condition["value"]]
        )
    return current


def cosine_scores(
    object_ids: Iterable[str], query_vector: np.ndarray, objects: dict[str, Any]
) -> list[dict[str, Any]]:
    q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    q_norm = float(np.linalg.norm(q))
    q = q / q_norm if q_norm else q
    rows = [
        {
            "object_id": object_id,
            "score": float(np.dot(q, objects[object_id]["normalized"])),
        }
        for object_id in object_ids
    ]
    rows.sort(key=lambda row: (-row["score"], row["object_id"]))
    return rows


def marginal_coverage(
    object_id: str,
    covered: dict[str, set[str]],
    diversify_by: list[str],
    objects: dict[str, Any],
) -> int:
    values = objects[object_id]["description"]["facet_values"]
    return sum(
        1 for facet in diversify_by if values.get(facet) not in covered[facet]
    )


def select_diverse(
    ranked_candidates: list[dict[str, Any]],
    diversify_by: list[str],
    top_k: int,
    objects: dict[str, Any],
) -> list[dict[str, Any]]:
    remaining = list(ranked_candidates)
    selected: list[dict[str, Any]] = []
    covered = {facet: set() for facet in diversify_by}
    while remaining and len(selected) < top_k:
        best_index, best = min(
            enumerate(remaining),
            key=lambda pair: (
                -marginal_coverage(
                    pair[1]["object_id"], covered, diversify_by, objects
                ),
                -pair[1]["score"],
                pair[1]["object_id"],
            ),
        )
        row = remaining.pop(best_index)
        object_id = row["object_id"]
        desc = objects[object_id]["description"]
        values = desc["facet_values"]
        for facet in diversify_by:
            if facet in values:
                covered[facet].add(values[facet])
        selected.append(
            {
                "object_id": object_id,
                "score": round(float(row["score"]), 6),
                "facet_values": values,
                "facet_quality": desc["facet_quality"],
                "uncertain": bool(desc.get("uncertain", False)),
            }
        )
    return selected


def exhaustive_candidates(
    eligible: set[str],
    query_vector: np.ndarray,
    objects: dict[str, Any],
) -> list[dict[str, Any]]:
    # This is the oracle: diversity is optimized over every object that passes
    # the hard predicates, not over a similarity-truncated candidate list.
    return cosine_scores(eligible, query_vector, objects)


def diversity_shortlist(
    ranked_candidates: list[dict[str, Any]],
    diversify_by: list[str],
    limit: int,
    objects: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build a bounded pool while retaining candidates for uncovered values."""
    if limit >= len(ranked_candidates):
        return ranked_candidates
    if not diversify_by:
        return ranked_candidates[:limit]
    selected = select_diverse(ranked_candidates, diversify_by, limit, objects)
    return [
        {"object_id": row["object_id"], "score": float(row["score"])}
        for row in selected
    ]


def vectorgroup_candidates(
    eligible: set[str],
    query_vector: np.ndarray,
    cache: dict[str, Any],
    objects: dict[str, Any],
    diversify_by: list[str],
    group_budget: int,
    candidate_k: int,
    diversity_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not eligible:
        return [], {
            "eligible_groups": 0,
            "searched_groups": 0,
            "scored_vectors": 0,
            "selected_group_routes": [],
        }
    entity = objects[next(iter(eligible))]["description"]["entity"]
    route_rows = cache.get("group_routes", {}).get(entity, [])
    if not route_rows:
        # Backward-compatible fallback for old caches.
        by_group: dict[tuple[str, str], list[str]] = defaultdict(list)
        for object_id in eligible:
            obj = objects[object_id]
            by_group[(obj["physical_key"], obj["group_name"])].append(object_id)
        route_rows = [
            {
                "physical_key": key[0],
                "group_name": key[1],
                "object_ids": tuple(ids),
            }
            for key, ids in by_group.items()
        ]

    q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    q_norm = float(np.linalg.norm(q))
    q = q / q_norm if q_norm else q
    groups = []
    for route in route_rows:
        ids = [object_id for object_id in route["object_ids"] if object_id in eligible]
        if not ids:
            continue
        first = objects[ids[0]]
        group_key = (route["physical_key"], route["group_name"])
        representative = np.asarray(
            first["group"].representative, dtype=np.float32
        ).reshape(-1)
        norm = float(np.linalg.norm(representative))
        representative = representative / norm if norm else representative
        facet_pairs = {
            (facet, objects[object_id]["description"]["facet_values"].get(facet))
            for object_id in ids
            for facet in diversify_by
            if facet in objects[object_id]["description"]["facet_values"]
        }
        groups.append(
            {
                "relevance": float(np.dot(q, representative)),
                "key": group_key,
                "ids": ids,
                "facet_pairs": facet_pairs,
            }
        )

    relevances = [row["relevance"] for row in groups]
    low = min(relevances, default=0.0)
    high = max(relevances, default=0.0)
    for row in groups:
        row["relevance_norm"] = (
            (row["relevance"] - low) / (high - low) if high > low else 1.0
        )
    pair_frequency: dict[tuple[str, str], int] = defaultdict(int)
    for row in groups:
        for pair in row["facet_pairs"]:
            pair_frequency[pair] += 1

    selected_groups: list[dict[str, Any]] = []
    remaining = list(groups)
    covered: set[tuple[str, str]] = set()
    while remaining and len(selected_groups) < group_budget:
        def group_key(row: dict[str, Any]) -> tuple[float, float, float, tuple[str, str]]:
            new_pairs = row["facet_pairs"] - covered
            rarity_gain = sum(1.0 / pair_frequency[pair] for pair in new_pairs)
            utility = row["relevance_norm"] + diversity_weight * rarity_gain
            return (utility, rarity_gain, row["relevance"], tuple(reversed(row["key"])))

        best = max(remaining, key=group_key)
        remaining.remove(best)
        selected_groups.append(best)
        covered.update(best["facet_pairs"])

    candidate_ids = [object_id for row in selected_groups for object_id in row["ids"]]
    all_ranked = cosine_scores(candidate_ids, query_vector, objects)
    ranked = diversity_shortlist(
        all_ranked, diversify_by, candidate_k, objects
    )
    stats = {
        "eligible_groups": len(groups),
        "searched_groups": len(selected_groups),
        "scored_vectors": len(set(candidate_ids)),
        "covered_group_facet_values": len(covered),
        "selected_group_routes": [
            {
                "physical_key": row["key"][0],
                "group_name": row["key"][1],
                "representative_score": round(row["relevance"], 6),
                "eligible_objects": len(row["ids"]),
                "facet_values": [f"{facet}={value}" for facet, value in sorted(row["facet_pairs"])],
            }
            for row in selected_groups
        ],
    }
    return ranked, stats


def milvus_filter(
    intent: dict[str, Any], schemas: dict[str, Any] | None = None
) -> str:
    """Compile facet predicates without rewriting the logical ARRAY layout."""
    entity = json.dumps(intent["entity"], ensure_ascii=False)
    expression = f"entity == {entity}"
    tokens = [
        f"{condition['key']}={condition['value']}" for condition in intent["must"]
    ]
    if tokens:
        token_json = json.dumps(tokens, ensure_ascii=False)
        expression += f" and ARRAY_CONTAINS_ALL(facet_tokens, {token_json})"
    return expression


def milvus_candidates(
    client: Any,
    collection: str,
    intent: dict[str, Any],
    query_vector: np.ndarray,
    objects: dict[str, Any],
    schemas: dict[str, Any],
    candidate_k: int,
    nprobe: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expression = milvus_filter(intent, schemas)
    response = client.search(
        collection_name=collection,
        data=[np.asarray(query_vector, dtype=np.float32).tolist()],
        anns_field="vector",
        filter=expression,
        limit=candidate_k,
        output_fields=["object_id", "facet_tokens"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": nprobe}},
    )
    hits = response[0] if response else []
    object_ids = []
    ann_distances: dict[str, float] = {}
    for hit in hits:
        entity = hit.get("entity", {})
        object_id = entity.get("object_id") or hit.get("object_id")
        if object_id in objects:
            object_ids.append(object_id)
            ann_distances[object_id] = float(hit.get("distance", 0.0))
    # Exact rerank only within the ANN candidates; candidate recall remains Milvus-dependent.
    ranked = cosine_scores(object_ids, query_vector, objects)
    return ranked, {
        "filter": expression,
        "ann_hit_count": len(object_ids),
        "ann_distances": ann_distances,
    }


def encode_queries(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    import clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"LOAD CLIP ViT-B/32 on {device}", flush=True)
    model, _ = clip.load("ViT-B/32", device=device)
    texts = [record["semantic_text_en"] for record in records]
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    array = features.cpu().numpy().astype(np.float32)
    return {record["id"]: array[index] for index, record in enumerate(records)}


def recall_at_k(gt: list[dict[str, Any]], result: list[dict[str, Any]]) -> float:
    gt_ids = {row["object_id"] for row in gt}
    if not gt_ids:
        return 0.0
    return len(gt_ids & {row["object_id"] for row in result}) / len(gt_ids)


def ndcg_at_k(gt: list[dict[str, Any]], result: list[dict[str, Any]]) -> float:
    # Graded relevance is defined by the deterministic oracle rank: k, k-1, ..., 1.
    relevance = {
        row["object_id"]: len(gt) - index for index, row in enumerate(gt)
    }

    def dcg(rows: list[dict[str, Any]]) -> float:
        return sum(
            (2 ** relevance.get(row["object_id"], 0) - 1) / math.log2(rank + 2)
            for rank, row in enumerate(rows)
        )

    ideal = dcg(gt)
    return dcg(result) / ideal if ideal else 0.0


def coverage(results: list[dict[str, Any]], diversify_by: list[str]) -> dict[str, list[str]]:
    return {
        facet: sorted(
            {
                row["facet_values"][facet]
                for row in results
                if facet in row["facet_values"]
            }
        )
        for facet in diversify_by
    }


def diversity_coverage_recall(
    gt: list[dict[str, Any]],
    result: list[dict[str, Any]],
    diversify_by: list[str],
) -> float | None:
    if not diversify_by:
        return None
    gt_pairs = {
        (facet, row["facet_values"].get(facet))
        for row in gt
        for facet in diversify_by
        if facet in row["facet_values"]
    }
    result_pairs = {
        (facet, row["facet_values"].get(facet))
        for row in result
        for facet in diversify_by
        if facet in row["facet_values"]
    }
    return len(gt_pairs & result_pairs) / len(gt_pairs) if gt_pairs else 1.0


def run_benchmark(
    args: argparse.Namespace,
    cache: dict[str, Any],
    objects: dict[str, Any],
) -> dict[str, Any]:
    from pymilvus import MilvusClient

    query_doc = json.loads(args.query_gt.read_text(encoding="utf-8"))
    records = query_doc["records"]
    for record in records:
        validate_intent(record["intent"], cache["schemas"])
    query_vectors = encode_queries(records)
    client = MilvusClient(str(args.milvus_uri.resolve()))
    client.load_collection(args.collection)

    output_records = []
    method_times: dict[str, list[float]] = defaultdict(list)
    for position, record in enumerate(records, start=1):
        intent = record["intent"]
        eligible = eligible_ids(cache, intent)
        query_vector = query_vectors[record["id"]]

        started = time.perf_counter()
        gt_candidates = exhaustive_candidates(eligible, query_vector, objects)
        gt_results = select_diverse(
            gt_candidates,
            intent["diversify_by"],
            args.top_k,
            objects,
        )
        gt_ms = (time.perf_counter() - started) * 1000
        method_times["ground_truth"].append(gt_ms)

        started = time.perf_counter()
        vg_candidates, vg_stats = vectorgroup_candidates(
            eligible,
            query_vector,
            cache,
            objects,
            intent["diversify_by"],
            args.vg_group_budget,
            args.candidate_k,
            args.vg_diversity_weight,
        )
        vg_results = select_diverse(
            vg_candidates,
            intent["diversify_by"],
            args.top_k,
            objects,
        )
        vg_ms = (time.perf_counter() - started) * 1000
        method_times["vectorgroup"].append(vg_ms)

        started = time.perf_counter()
        milvus_ranked, milvus_stats = milvus_candidates(
            client,
            args.collection,
            intent,
            query_vector,
            objects,
            cache["schemas"],
            args.candidate_k,
            args.milvus_nprobe,
        )
        milvus_results = select_diverse(
            milvus_ranked,
            intent["diversify_by"],
            args.top_k,
            objects,
        )
        milvus_ms = (time.perf_counter() - started) * 1000
        method_times["milvus"].append(milvus_ms)

        output_records.append(
            {
                **record,
                "eligible_count": len(eligible),
                "methods": {
                    "ground_truth": {
                        "latency_ms": round(gt_ms, 3),
                        "candidate_count": len(gt_candidates),
                        "results": gt_results,
                        "coverage": coverage(gt_results, intent["diversify_by"]),
                    },
                    "vectorgroup": {
                        "latency_ms": round(vg_ms, 3),
                        "candidate_count": len(vg_candidates),
                        "recall_at_k": round(recall_at_k(gt_results, vg_results), 6),
                        "ndcg_at_k": round(ndcg_at_k(gt_results, vg_results), 6),
                        "diversity_coverage_recall": (
                            None
                            if diversity_coverage_recall(
                                gt_results, vg_results, intent["diversify_by"]
                            ) is None
                            else round(
                                diversity_coverage_recall(
                                    gt_results, vg_results, intent["diversify_by"]
                                ),
                                6,
                            )
                        ),
                        "coverage": coverage(vg_results, intent["diversify_by"]),
                        "stats": vg_stats,
                        "results": vg_results,
                    },
                    "milvus": {
                        "latency_ms": round(milvus_ms, 3),
                        "candidate_count": len(milvus_ranked),
                        "recall_at_k": round(recall_at_k(gt_results, milvus_results), 6),
                        "ndcg_at_k": round(ndcg_at_k(gt_results, milvus_results), 6),
                        "diversity_coverage_recall": (
                            None
                            if diversity_coverage_recall(
                                gt_results, milvus_results, intent["diversify_by"]
                            ) is None
                            else round(
                                diversity_coverage_recall(
                                    gt_results, milvus_results, intent["diversify_by"]
                                ),
                                6,
                            )
                        ),
                        "coverage": coverage(milvus_results, intent["diversify_by"]),
                        "stats": {
                            "filter": milvus_stats["filter"],
                            "ann_hit_count": milvus_stats["ann_hit_count"],
                        },
                        "results": milvus_results,
                    },
                },
            }
        )
        print(
            f"OK {position:02d}/{len(records)} {record['id']} eligible={len(eligible)} "
            f"VG R={recall_at_k(gt_results, vg_results):.3f} "
            f"M R={recall_at_k(gt_results, milvus_results):.3f}",
            flush=True,
        )

    client.close()
    summaries = {}
    for method in ("vectorgroup", "milvus"):
        recalls = [row["methods"][method]["recall_at_k"] for row in output_records]
        ndcgs = [row["methods"][method]["ndcg_at_k"] for row in output_records]
        coverage_recalls = [
            row["methods"][method]["diversity_coverage_recall"]
            for row in output_records
            if row["methods"][method]["diversity_coverage_recall"] is not None
        ]
        summaries[method] = {
            "mean_recall_at_k": round(float(np.mean(recalls)), 6),
            "mean_ndcg_at_k": round(float(np.mean(ndcgs)), 6),
            "mean_diversity_coverage_recall": (
                round(float(np.mean(coverage_recalls)), 6)
                if coverage_recalls
                else None
            ),
            "diversity_query_count": len(coverage_recalls),
            "mean_latency_ms": round(float(np.mean(method_times[method])), 3),
            "p95_latency_ms": round(float(np.percentile(method_times[method], 95)), 3),
        }
    summaries["ground_truth"] = {
        "mean_latency_ms": round(float(np.mean(method_times["ground_truth"])), 3),
        "p95_latency_ms": round(
            float(np.percentile(method_times["ground_truth"], 95)), 3
        ),
    }
    return {
        "format": "caltech101-kv-retrieval-comparison-v2",
        "config": {
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "vectorgroup_group_budget": args.vg_group_budget,
            "vectorgroup_group_selection": "relevance_plus_rarity_weighted_facet_coverage",
            "vectorgroup_diversity_weight": args.vg_diversity_weight,
            "milvus_index": "IVF_FLAT",
            "milvus_nlist": args.milvus_nlist,
            "milvus_nprobe": args.milvus_nprobe,
            "diversity_selector": "lexicographic(marginal_coverage, cosine_similarity, object_id)",
            "ndcg_relevance": "oracle rank graded relevance: k,k-1,...,1",
            "query_gt": str(args.query_gt),
            "query_gt_sha256": sha256(args.query_gt),
            "execution_policy": query_doc.get("execution_policy", "best_effort"),
        },
        "corpus": cache["manifest"],
        "summary": summaries,
        "records": output_records,
    }


def image_uri(local_root: str, object_id: str) -> str:
    from artifact_paths import image_uri as resolve_image_uri
    return resolve_image_uri(local_root, object_id)


def render_result_card(
    row: dict[str, Any],
    rank: int,
    local_root: str,
    diversify_by: list[str],
) -> str:
    uri = image_uri(local_root, row["object_id"])
    filename = row["object_id"].split("/")[-1]
    facets = []
    for key, value in row["facet_values"].items():
        active = " active" if key in diversify_by else ""
        uncertain = row["facet_quality"].get(key, {}).get("uncertain", False)
        marker = " ?" if uncertain else ""
        facets.append(
            f'<span class="facet{active}">{html.escape(key)}={html.escape(str(value))}{marker}</span>'
        )
    return f"""
      <article class="result">
        <div class="rank">#{rank}</div>
        <img src="{uri}" alt="{html.escape(row['object_id'])}" loading="lazy">
        <div class="result-body">
          <div class="filename">{html.escape(filename)}</div>
          <div class="score">cosine {row['score']:.4f}</div>
          <div class="facets">{''.join(facets)}</div>
        </div>
      </article>"""


def render_method(
    label: str,
    method: dict[str, Any],
    local_root: str,
    diversify_by: list[str],
    top_k: int,
    is_gt: bool = False,
) -> str:
    metric = (
        '<span class="baseline">oracle</span>'
        if is_gt
        else f'<span>Recall@{top_k} <b>{method["recall_at_k"]:.3f}</b></span>'
        f'<span>nDCG@{top_k} <b>{method["ndcg_at_k"]:.3f}</b></span>'
    )
    cards = "".join(
        render_result_card(row, index, local_root, diversify_by)
        for index, row in enumerate(method["results"], start=1)
    )
    coverage_text = " · ".join(
        f"{key}: {', '.join(values)}" for key, values in method["coverage"].items()
    ) or "无 diversity facet"
    return f"""
      <section class="method">
        <div class="method-head">
          <h3>{html.escape(label)}</h3>
          <div class="metrics">{metric}</div>
        </div>
        <div class="method-meta">候选 {method['candidate_count']} · {method['latency_ms']:.2f} ms · {html.escape(coverage_text)}</div>
        <div class="results">{cards}</div>
      </section>"""


def render_html(data: dict[str, Any], local_root: str) -> str:
    summary = data["summary"]
    top_k = data["config"]["top_k"]
    query_sections = []
    for record in data["records"]:
        intent_json = json.dumps(record["intent"], ensure_ascii=False, indent=2)
        unresolved = record["intent"].get("unresolved", [])
        unresolved_html = (
            '<div class="unresolved"><b>Best-effort：</b>'
            + "；".join(
                f'{html.escape(item["text"])} ({html.escape(item["reason"])})'
                for item in unresolved
            )
            + "</div>"
            if unresolved
            else ""
        )
        methods = record["methods"]
        query_sections.append(
            f"""
    <section class="query-block" id="{html.escape(record['id'])}">
      <div class="query-head">
        <div><span class="query-id">{html.escape(record['id'])}</span><h2>{html.escape(record['natural_query'])}</h2></div>
        <div class="eligible">eligible corpus <b>{record['eligible_count']}</b></div>
      </div>
      <div class="query-grid">
        <div>
          <div class="label">Fixed CLIP text</div>
          <code>{html.escape(record['semantic_text_en'])}</code>
          {unresolved_html}
        </div>
        <details>
          <summary>Qwen intent · actually executed</summary>
          <pre>{html.escape(intent_json)}</pre>
        </details>
      </div>
      <div class="method-grid">
        {render_method('Ground Truth · exhaustive', methods['ground_truth'], local_root, record['intent']['diversify_by'], top_k, True)}
        {render_method('VectorGroup · diversity-aware', methods['vectorgroup'], local_root, record['intent']['diversify_by'], top_k)}
        {render_method('Milvus · IVF_FLAT', methods['milvus'], local_root, record['intent']['diversify_by'], top_k)}
      </div>
    </section>"""
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Caltech-101 KV Retrieval Comparison</title>
<style>
:root{{--bg:#f5f7fb;--panel:#fff;--text:#172033;--muted:#667085;--line:#dfe4ec;--blue:#315efb;--blue-soft:#eaf0ff;--green:#138a5b;--orange:#b85c00;--shadow:0 8px 28px rgba(25,35,55,.08)}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif}} .wrap{{max-width:1500px;margin:auto;padding:28px}}
h1{{margin:0 0 6px;font-size:28px}} h2{{display:inline;margin:0 0 0 10px;font-size:19px}} h3{{margin:0;font-size:15px}} .subtitle,.method-meta,.label,.score{{color:var(--muted)}}
.summary{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:22px 0}} .summary-card,.query-block,.method{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}} .summary-card{{padding:16px}} .summary-card .big{{font-size:26px;font-weight:700;margin:6px 0}} .pair{{display:flex;justify-content:space-between;gap:10px;margin-top:5px}}
.protocol{{background:var(--blue-soft);border-radius:10px;padding:12px 14px;color:#22335d;margin-bottom:20px}} .query-block{{padding:18px;margin:18px 0}} .query-head{{display:flex;justify-content:space-between;gap:12px;align-items:center}} .query-id{{display:inline-block;background:var(--text);color:#fff;border-radius:6px;padding:3px 7px;font-size:12px}} .eligible{{color:var(--muted);white-space:nowrap}}
.query-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:14px 0}} .label{{font-size:12px;margin-bottom:5px;text-transform:uppercase;letter-spacing:.04em}} code,pre{{font-family:"SFMono-Regular",Consolas,monospace}} pre{{white-space:pre-wrap;background:#f3f5f8;border-radius:8px;padding:12px;margin:8px 0 0;font-size:12px}} summary{{cursor:pointer;color:var(--blue)}} .unresolved{{margin-top:10px;color:var(--orange);font-size:13px}}
.method-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}} .method{{overflow:hidden;box-shadow:none}} .method-head{{display:flex;justify-content:space-between;align-items:center;padding:12px;border-bottom:1px solid var(--line)}} .metrics{{display:flex;gap:8px;font-size:12px;color:var(--muted)}} .baseline{{color:var(--green);font-weight:700}} .method-meta{{padding:8px 12px;font-size:12px;border-bottom:1px solid var(--line)}} .results{{padding:8px}}
.result{{display:grid;grid-template-columns:26px 108px 1fr;gap:8px;align-items:center;padding:7px 4px;border-bottom:1px solid var(--line)}} .result:last-child{{border-bottom:0}} .result img{{width:108px;height:82px;object-fit:contain;background:#eef1f5;border-radius:7px}} .rank{{color:var(--muted);font-weight:700}} .filename{{font-weight:600;font-size:13px;overflow-wrap:anywhere}} .score{{font-size:12px;margin:3px 0}} .facets{{display:flex;flex-wrap:wrap;gap:4px}} .facet{{font-size:10px;padding:2px 5px;border-radius:999px;background:#eef1f5;color:#4b5565;overflow-wrap:anywhere}} .facet.active{{background:var(--blue-soft);color:var(--blue)}}
.footer{{color:var(--muted);font-size:12px;margin-top:18px}} @media(max-width:1100px){{.method-grid{{grid-template-columns:1fr}}.result{{grid-template-columns:30px 140px 1fr}}.result img{{width:140px}}}} @media(max-width:700px){{.wrap{{padding:14px}}.summary,.query-grid{{grid-template-columns:1fr}}.query-head{{align-items:flex-start;flex-direction:column}}.result{{grid-template-columns:24px 90px 1fr}}.result img{{width:90px;height:72px}}}}
</style>
</head>
<body><main class="wrap">
<h1>Caltech-101 KV Retrieval：GT vs VectorGroup vs Milvus</h1>
<div class="subtitle">{len(data['records'])} 条 Qwen intent · 6,984 train images · top-{top_k}</div>
<div class="summary">
  <div class="summary-card"><div>VectorGroup</div><div class="big">R@{top_k} {summary['vectorgroup']['mean_recall_at_k']:.3f}</div><div class="pair"><span>mean nDCG@{top_k}</span><b>{summary['vectorgroup']['mean_ndcg_at_k']:.3f}</b></div><div class="pair"><span>diversity coverage</span><b>{summary['vectorgroup']['mean_diversity_coverage_recall']:.3f}</b></div><div class="pair"><span>mean latency</span><b>{summary['vectorgroup']['mean_latency_ms']:.2f} ms</b></div></div>
  <div class="summary-card"><div>Milvus IVF_FLAT</div><div class="big">R@{top_k} {summary['milvus']['mean_recall_at_k']:.3f}</div><div class="pair"><span>mean nDCG@{top_k}</span><b>{summary['milvus']['mean_ndcg_at_k']:.3f}</b></div><div class="pair"><span>diversity coverage</span><b>{summary['milvus']['mean_diversity_coverage_recall']:.3f}</b></div><div class="pair"><span>mean latency</span><b>{summary['milvus']['mean_latency_ms']:.2f} ms</b></div></div>
  <div class="summary-card"><div>Exhaustive GT</div><div class="big">{data['corpus']['train_object_count']:,}</div><div class="pair"><span>mean latency</span><b>{summary['ground_truth']['mean_latency_ms']:.2f} ms</b></div><div class="pair"><span>candidate pool</span><b>{data['config']['candidate_k']}</b></div></div>
</div>
<div class="protocol"><b>Protocol.</b> must 先做精确 KV 过滤；GT 在全部 eligible corpus 上执行 diversity oracle；VectorGroup 用相关性与 facet-value 覆盖联合选择 {data['config']['vectorgroup_group_budget']} 个组；Milvus 使用 IVF_FLAT nprobe={data['config']['milvus_nprobe']} 返回纯相似度 Top-{data['config']['candidate_k']}；最终输出 top-{top_k}。</div>
{''.join(query_sections)}
<div class="footer">“?” 表示该 facet assignment 的 margin 低于 0.03。Best-effort 查询只执行 schema 能表达的部分。</div>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    cache = load_or_build_kv_cache(args)
    objects = make_object_map(cache)
    if len(objects) != cache["manifest"]["train_object_count"]:
        raise RuntimeError("object map count mismatch")
    build_milvus(args, cache, objects)
    result = run_benchmark(args, cache, objects)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    html_path = args.output_dir / "caltech10_kv_retrieval.html"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    html_path.write_text(
        render_html(result, args.local_dataset_root), encoding="utf-8"
    )
    print(f"RESULT JSON {result_path}", flush=True)
    print(f"REPORT HTML {html_path}", flush=True)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
