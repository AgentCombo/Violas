#!/usr/bin/env python3
"""Persistent local Qdrant and Chroma baselines for canonical intent queries."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

import run_guideline_pipeline as core


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _manifest_path(path: Path) -> Path:
    return path.with_name(path.name + ".manifest.json")


def _replace_cache(path: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    cache_root = (PROJECT_ROOT / "cache").resolve()
    if cache_root not in resolved.parents:
        raise RuntimeError(f"refusing to replace backend path outside cache: {resolved}")
    shutil.rmtree(resolved) if resolved.is_dir() else resolved.unlink()


def _tokens(obj: dict[str, Any]) -> list[str]:
    return [
        f"{key}={value}"
        for key, value in sorted(
            obj["description"].get("facet_values", {}).items()
        )
    ]


def _manifest_matches(
    manifest: dict[str, Any], fmt: str, collection: str, objects: dict[str, Any]
) -> bool:
    return (
        manifest.get("format") == fmt
        and manifest.get("collection") == collection
        and manifest.get("object_count") == len(objects)
        and manifest.get("dimension")
        == len(next(iter(objects.values()))["normalized"])
    )


def build_qdrant(
    path: Path, collection: str, objects: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    from qdrant_client import QdrantClient, models

    fmt = "violas-qdrant-local-v1"
    mp = _manifest_path(path)
    if path.exists() and mp.exists():
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        if _manifest_matches(manifest, fmt, collection, objects):
            print(f"LOAD Qdrant cache: {path}", flush=True)
            return manifest
    _replace_cache(path)
    mp.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    dimension = len(next(iter(objects.values()))["normalized"])
    print(f"BUILD Qdrant local: {path}", flush=True)
    client = QdrantClient(path=str(path.resolve()))
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=dimension, distance=models.Distance.COSINE
        ),
    )
    points = []
    inserted = 0
    for numeric_id, oid in enumerate(sorted(objects)):
        obj = objects[oid]
        points.append(
            models.PointStruct(
                id=numeric_id,
                vector=obj["normalized"].astype(np.float32).tolist(),
                payload={
                    "object_id": oid,
                    "entity": obj["description"]["entity"],
                    "facet_tokens": _tokens(obj),
                },
            )
        )
        if len(points) >= 500:
            client.upsert(collection_name=collection, points=points, wait=True)
            inserted += len(points)
            points = []
    if points:
        client.upsert(collection_name=collection, points=points, wait=True)
        inserted += len(points)
    client.create_payload_index(
        collection_name=collection,
        field_name="entity",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )
    client.create_payload_index(
        collection_name=collection,
        field_name="facet_tokens",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )
    client.close()
    manifest = {
        "format": fmt,
        "collection": collection,
        "object_count": inserted,
        "dimension": dimension,
        "vector_index": "Qdrant default HNSW/COSINE",
        "payload_indexes": ["entity", "facet_tokens"],
    }
    mp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def open_qdrant(path: Path):
    from qdrant_client import QdrantClient

    return QdrantClient(path=str(path.resolve()))


def qdrant_search(
    client: Any,
    collection: str,
    query: np.ndarray,
    plan: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    candidate_k: int,
    use_facets: bool,
) -> tuple[list[str], dict[str, float], dict[str, Any]]:
    from qdrant_client import models

    conditions = [
        models.FieldCondition(
            key="entity", match=models.MatchValue(value=plan["entity_scope"])
        )
    ]
    if use_facets:
        conditions.extend(
            models.FieldCondition(
                key="facet_tokens",
                match=models.MatchValue(value=f"{row['key']}={row['value']}"),
            )
            for row in plan["hard_filters"]
        )
    limit = candidate_k if use_facets else plan["top_k"]
    response = client.query_points(
        collection_name=collection,
        query=np.asarray(query, dtype=np.float32).tolist(),
        query_filter=models.Filter(must=conditions),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    hits = response.points
    ids = []
    for hit in hits:
        oid = (hit.payload or {}).get("object_id")
        if oid in objects and oid not in ids:
            ids.append(oid)
    scored = {
        oid: core.score_object(query, objects[oid]["normalized"]) for oid in ids
    }
    if use_facets:
        ids = core.select_exact(
            scored,
            objects,
            plan["facet_dimension"],
            plan["top_k"],
            plan["lambda"],
            plan["category_preferences"],
        )
    else:
        ids = ids[: plan["top_k"]]
    return ids, scored, {
        "filter": "entity+facet_tokens" if use_facets else "entity-only",
        "candidate_count": len(hits),
        "post_ann_select": use_facets,
    }


def _chroma_facet_key(key: str) -> str:
    return f"facet__{key}"


def build_chroma(
    path: Path, collection: str, objects: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    import chromadb
    from chromadb.config import Settings

    fmt = "violas-chroma-local-v1"
    mp = _manifest_path(path)
    if path.exists() and mp.exists():
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        if _manifest_matches(manifest, fmt, collection, objects):
            print(f"LOAD Chroma cache: {path}", flush=True)
            return manifest
    _replace_cache(path)
    mp.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    dimension = len(next(iter(objects.values()))["normalized"])
    print(f"BUILD Chroma local: {path}", flush=True)
    client = chromadb.PersistentClient(
        path=str(path.resolve()),
        settings=Settings(anonymized_telemetry=False),
    )
    db = client.create_collection(
        name=collection,
        metadata={"hnsw:space": "cosine"},
    )
    ids: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, Any]] = []
    inserted = 0
    for oid in sorted(objects):
        obj = objects[oid]
        metadata: dict[str, Any] = {
            "object_id": oid,
            "entity": obj["description"]["entity"],
        }
        metadata.update(
            {
                _chroma_facet_key(key): value
                for key, value in obj["description"]
                .get("facet_values", {})
                .items()
            }
        )
        ids.append(oid)
        embeddings.append(obj["normalized"].astype(np.float32).tolist())
        metadatas.append(metadata)
        if len(ids) >= 500:
            db.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            inserted += len(ids)
            ids, embeddings, metadatas = [], [], []
    if ids:
        db.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        inserted += len(ids)
    manifest = {
        "format": fmt,
        "collection": collection,
        "object_count": inserted,
        "dimension": dimension,
        "vector_index": "Chroma default HNSW/COSINE",
        "metadata_filter": "entity + facet__<key>",
    }
    mp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def open_chroma(path: Path, collection: str):
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(path.resolve()),
        settings=Settings(anonymized_telemetry=False),
    )
    return client, client.get_collection(name=collection)


def chroma_search(
    collection: Any,
    query: np.ndarray,
    plan: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    candidate_k: int,
    use_facets: bool,
) -> tuple[list[str], dict[str, float], dict[str, Any]]:
    predicates: list[dict[str, Any]] = [
        {"entity": {"$eq": plan["entity_scope"]}}
    ]
    if use_facets:
        predicates.extend(
            {
                _chroma_facet_key(row["key"]): {"$eq": row["value"]}
            }
            for row in plan["hard_filters"]
        )
    where = predicates[0] if len(predicates) == 1 else {"$and": predicates}
    limit = candidate_k if use_facets else plan["top_k"]
    response = collection.query(
        query_embeddings=[np.asarray(query, dtype=np.float32).tolist()],
        n_results=limit,
        where=where,
        include=["metadatas", "distances"],
    )
    ids = []
    for oid in (response.get("ids") or [[]])[0]:
        if oid in objects and oid not in ids:
            ids.append(oid)
    scored = {
        oid: core.score_object(query, objects[oid]["normalized"]) for oid in ids
    }
    if use_facets:
        ids = core.select_exact(
            scored,
            objects,
            plan["facet_dimension"],
            plan["top_k"],
            plan["lambda"],
            plan["category_preferences"],
        )
    else:
        ids = ids[: plan["top_k"]]
    return ids, scored, {
        "filter": "entity+facet metadata" if use_facets else "entity-only",
        "candidate_count": len((response.get("ids") or [[]])[0]),
        "post_ann_select": use_facets,
    }
