from __future__ import annotations
from collections import defaultdict, Counter
from typing import Any, Iterable
import heapq
import math
import numpy as np

UNKNOWN="__unknown__"
NO_DIMENSION="__relevance__"
LAMBDA_BY_LEVEL={"off":0.0,"light":0.05,"balanced":0.10,"strong":0.15}


def resolve_plan(intent: dict[str, Any]) -> dict[str, Any]:
    level = intent["diversity_level"]
    return {
        "entity_scope": intent["entity_scope"],
        "hard_filters": intent["hard_filters"],
        "facet_dimension": intent["facet_dimension"],
        "diversity_level": level,
        "lambda": LAMBDA_BY_LEVEL[level],
        "category_preferences": intent.get("category_preferences"),
        "top_k": intent["top_k"],
        "embedding_text_en": intent["embedding_text_en"],
        "unsupported": intent["unsupported"],
    }


def materialize_plan_preferences(
    plan: dict[str, Any], directory: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the paper's per-category (lambda_v, r_v) query preferences.

    The current language adapter supplies a common preference level.  When it
    does not name per-category quotas, the paper's default coverage request is
    used: every known value of the selected dimension receives the common
    weight and quota one.  Structured callers may instead provide explicit
    ``category_preferences`` rows with value, quota, and either lambda or a
    configured weight_level.
    """
    dimension = plan["facet_dimension"]
    common_lambda = float(plan["lambda"])
    raw = plan.get("category_preferences")
    resolved: dict[str, dict[str, float | int]] = {}
    if dimension is None or common_lambda == 0.0:
        plan["category_preferences"] = resolved
        return plan

    known_values = set(
        directory["postings"].get(plan["entity_scope"], {}).get(dimension, {})
    )
    if raw:
        if not isinstance(raw, list):
            raise ValueError("category_preferences must be a list")
        for row in raw:
            if not isinstance(row, dict) or "value" not in row or "quota" not in row:
                raise ValueError(f"invalid category preference: {row!r}")
            value = str(row["value"])
            if value not in known_values:
                raise ValueError(f"unknown category preference value {value!r}")
            quota = int(row["quota"])
            if quota < 0:
                raise ValueError("category quota must be non-negative")
            if "lambda" in row:
                weight = float(row["lambda"])
            else:
                level = row.get("weight_level", plan["diversity_level"])
                if level not in LAMBDA_BY_LEVEL:
                    raise ValueError(f"invalid category weight level {level!r}")
                weight = float(LAMBDA_BY_LEVEL[level])
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("category lambda must be finite and non-negative")
            resolved[value] = {"lambda": weight, "quota": quota}
    else:
        resolved = {
            value: {"lambda": common_lambda, "quota": 1}
            for value in sorted(known_values)
        }
    plan["category_preferences"] = resolved
    return plan


def category_preference(
    preferences: dict[str, dict[str, float | int]] | None,
    value: str,
    dimension: str | None,
    fallback_lambda: float,
) -> tuple[float, int]:
    if dimension is None or value == UNKNOWN:
        return 0.0, 0
    if preferences is None:
        return float(fallback_lambda), 1
    row = preferences.get(value)
    if row is None:
        return 0.0, 0
    return float(row["lambda"]), int(row["quota"])


def satisfies(desc: dict[str, Any], plan: dict[str, Any]) -> bool:
    if desc.get("entity") != plan["entity_scope"]:
        return False
    values = desc.get("facet_values", {})
    return all(values.get(c["key"]) == c["value"] for c in plan["hard_filters"])


def eligible_ids(objects: dict[str, dict[str, Any]], plan: dict[str, Any]) -> set[str]:
    return {oid for oid, obj in objects.items() if satisfies(obj["description"], plan)}


def score_object(query: np.ndarray, vector: np.ndarray) -> float:
    return float((1.0 + float(np.dot(query, vector))) / 2.0)


def label_of(objects: dict[str, dict[str, Any]], oid: str, dimension: str | None) -> str:
    if dimension is None:
        return NO_DIMENSION
    return objects[oid]["description"].get("facet_values", {}).get(dimension, UNKNOWN)


def select_exact(
    scored: dict[str, float],
    objects: dict[str, dict[str, Any]],
    dimension: str | None,
    top_k: int,
    lambda_value: float,
    category_preferences: dict[str, dict[str, float | int]] | None = None,
) -> list[str]:
    """Exactly optimize the paper's one-dimension soft-quota objective."""
    if top_k <= 0 or not scored:
        return []
    lists: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for oid, score in scored.items():
        lists[label_of(objects, oid, dimension)].append((oid, float(score)))
    for rows in lists.values():
        rows.sort(key=lambda row: (-row[1], row[0]))
    heap: list[tuple[float, str, str, int]] = []
    for value, rows in lists.items():
        oid, score = rows[0]
        weight, quota = category_preference(
            category_preferences, value, dimension, lambda_value
        )
        bonus = weight if quota >= 1 else 0.0
        heapq.heappush(heap, (-(score + bonus), oid, value, 0))
    selected: list[str] = []
    while heap and len(selected) < min(top_k, len(scored)):
        _, oid, value, index = heapq.heappop(heap)
        selected.append(oid)
        next_index = index + 1
        if next_index < len(lists[value]):
            next_oid, next_score = lists[value][next_index]
            weight, quota = category_preference(
                category_preferences, value, dimension, lambda_value
            )
            bonus = weight if next_index < quota else 0.0
            heapq.heappush(
                heap, (-(next_score + bonus), next_oid, value, next_index)
            )
    return selected


def objective(
    selected: Iterable[str],
    scored: dict[str, float],
    objects: dict[str, dict[str, Any]],
    dimension: str | None,
    lambda_value: float,
    category_preferences: dict[str, dict[str, float | int]] | None = None,
) -> float:
    ids = list(selected)
    counts: dict[str, int] = defaultdict(int)
    for oid in ids:
        counts[label_of(objects, oid, dimension)] += 1
    reward = 0.0
    for value, count in counts.items():
        weight, quota = category_preference(
            category_preferences, value, dimension, lambda_value
        )
        reward += weight * min(count, quota)
    return float(sum(scored[oid] for oid in ids) + reward)
