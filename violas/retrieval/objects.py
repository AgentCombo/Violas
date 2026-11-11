from __future__ import annotations
from collections import defaultdict, Counter
from typing import Any, Iterable
import heapq
import math
import numpy as np


def iter_objects(cache: dict[str, Any]) -> Iterable[tuple[str, np.ndarray, dict[str, Any], str, Any]]:
    vectormap = cache["vectormap"]
    for physical_key, entry in vectormap.data.items():
        for group in entry["groups"]:
            for idx, (vector, description) in enumerate(
                zip(group.vectors, group.descriptions)
            ):
                if "object_id" not in description:
                    continue
                yield description["object_id"], np.asarray(vector), description, physical_key, group


def make_object_map(cache: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for object_id, vector, description, physical_key, group in iter_objects(cache):
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        normalized = vec / norm if norm else vec
        result[object_id] = {
            "vector": vec,
            "normalized": normalized,
            "description": description,
            "physical_key": physical_key,
            "group_name": group.group_name,
            "group": group,
        }
    return result
