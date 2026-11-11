from __future__ import annotations
from collections import defaultdict, Counter
from typing import Any, Iterable
import heapq
import math
import numpy as np

from .objects import make_object_map


def _geometric_summary(
    ids: Iterable[str],
    objects: dict[str, dict[str, Any]],
    representative_count: int = 4,
) -> dict[str, Any]:
    """Build a deterministic spherical cap and farthest-first representatives."""
    ordered_ids = tuple(sorted(ids))
    matrix = np.stack([objects[oid]["normalized"] for oid in ordered_ids]).astype(np.float32)
    raw_center = matrix.mean(axis=0)
    center_norm = float(np.linalg.norm(raw_center))
    if center_norm > 0:
        unit_center = (raw_center / center_norm).astype(np.float32)
        center_cosines = np.clip(matrix @ unit_center, -1.0, 1.0)
        angular_radius = float(np.arccos(center_cosines).max())
        first = min(
            range(len(ordered_ids)),
            key=lambda index: (-float(center_cosines[index]), ordered_ids[index]),
        )
    else:
        unit_center = np.zeros_like(raw_center, dtype=np.float32)
        angular_radius = math.pi
        first = 0

    selected = [first]
    min_sq_distance = np.sum((matrix - matrix[first]) ** 2, axis=1)
    while len(selected) < min(representative_count, len(ordered_ids)):
        candidates = [index for index in range(len(ordered_ids)) if index not in selected]
        next_index = min(
            candidates,
            key=lambda index: (-float(min_sq_distance[index]), ordered_ids[index]),
        )
        selected.append(next_index)
        min_sq_distance = np.minimum(
            min_sq_distance,
            np.sum((matrix - matrix[next_index]) ** 2, axis=1),
        )
    return {
        "unit_center": unit_center,
        "angular_radius": angular_radius,
        "representative_ids": tuple(ordered_ids[index] for index in selected),
    }


def _angular_similarity_upper(query: np.ndarray, summary: dict[str, Any]) -> float:
    """Upper-bound score_object() for unit vectors inside a spherical cap."""
    center = summary["unit_center"]
    if not np.any(center):
        return 1.0
    cosine = float(np.clip(np.dot(query, center), -1.0, 1.0))
    theta = math.acos(cosine)
    best_cosine = math.cos(max(0.0, theta - float(summary["angular_radius"])))
    return float((1.0 + best_cosine) / 2.0)


def build_semantic_directory(cache: dict[str, Any]) -> dict[str, Any]:
    vm = cache["vectormap"]
    objects = make_object_map(cache)
    node_members: list[tuple[str, ...]] = []
    centers: list[np.ndarray] = []
    radii: list[float] = []
    node_summaries: list[dict[str, Any]] = []
    subset_summaries: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    postings: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for node_id, (physical_key, group) in enumerate(vm._hdmg_nodes):
        members = tuple(
            desc["object_id"] for desc in group.descriptions if desc.get("object_id") in objects
        )
        node_members.append(members)
        matrix = np.stack([objects[oid]["normalized"] for oid in members]).astype(np.float32)
        center = matrix.mean(axis=0)
        radius = float(np.linalg.norm(matrix - center, axis=1).max()) if len(matrix) else 0.0
        centers.append(center)
        radii.append(radius)
        node_summaries.append(_geometric_summary(members, objects))
        by_value: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for oid in members:
            desc = objects[oid]["description"]
            entity = desc["entity"]
            for dimension, value in desc.get("facet_values", {}).items():
                by_value[(entity, dimension, value)].append(oid)
        for (entity, dimension, value), ids in by_value.items():
            summary = _geometric_summary(ids, objects)
            subset_summaries[(node_id, entity, dimension, value)] = summary
            postings[entity][dimension][value].append(
                {
                    "node_id": node_id,
                    "physical_key": physical_key,
                    "group_name": group.group_name,
                    "count": len(ids),
                    "object_ids": tuple(sorted(ids)),
                    "summary": summary,
                }
            )
    for entity in postings:
        for dimension in postings[entity]:
            for value, entries in postings[entity][dimension].items():
                entries.sort(key=lambda row: (-row["count"], row["node_id"]))
    return {
        "format": "violas-semantic-entry-directory-v2",
        "node_count": len(node_members),
        "object_count": len(objects),
        "node_members": node_members,
        "centers": np.stack(centers).astype(np.float32),
        "radii": np.asarray(radii, dtype=np.float32),
        "node_summaries": node_summaries,
        "subset_summaries": subset_summaries,
        "postings": {
            entity: {
                dimension: dict(values) for dimension, values in dimensions.items()
            }
            for entity, dimensions in postings.items()
        },
    }
