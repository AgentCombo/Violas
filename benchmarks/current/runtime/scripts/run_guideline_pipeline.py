#!/usr/bin/env python3
"""Run the intent-guided, interleaved HDMG pipeline from the ICDE guides.

This is a P0 executable reference implementation for the contracts in:
  * 01-paper-blueprint-and-terra-guide.md
  * 02-retrieval-implementation-guide.md
  * 03-evaluation-and-evidence-plan.md

It deliberately keeps the existing geometric micro-clusters and HDMG graph.
Facet assignments are materialized as a semantic entry directory.  Query-time
navigation, member scoring, result selection, and semantic entry injection are
then interleaved under explicit navigation and member-score budgets.
"""

from __future__ import annotations

import argparse
import heapq
import html
import itertools
import json
import math
import os
import pickle
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_caltech_kv_retrieval as retrieval


MODEL = "qwen3.5"
BASE_URL = "https://uni-api.cstcloud.cn/v1"
LAMBDA_BY_LEVEL = {
    "off": 0.0,
    "light": 0.05,
    "balanced": 0.10,
    "strong": 0.15,
}
UNKNOWN = "__unknown__"
NO_DIMENSION = "__relevance__"
_thread_local = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("queries/caltech101_guideline_10_input.json"),
    )
    parser.add_argument(
        "--intent-output",
        type=Path,
        default=Path("queries/caltech101_guideline_10_qwen.json"),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("queries/caltech101_guideline_10_raw.jsonl"),
    )
    parser.add_argument(
        "--hdmg-cache",
        type=Path,
        default=Path("cache/caltech101_kv_train_v3_hdmg.pkl"),
    )
    parser.add_argument(
        "--directory-cache",
        type=Path,
        default=Path("cache/caltech101_semantic_directory_v1.pkl"),
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
        default=Path("artifacts/caltech10_guideline_interleaved"),
    )
    parser.add_argument("--nav-budget", type=int, default=12)
    parser.add_argument("--score-budget", type=int, default=80)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--execution-profile",
        choices=("paper", "subset-v2"),
        default="paper",
        help="paper follows the SIGMOD draft; subset-v2 retains the experimental refinement",
    )
    parser.add_argument("--milvus-candidate-k", type=int, default=80)
    parser.add_argument("--milvus-nprobe", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--force-intents", action="store_true")
    parser.add_argument("--compile-intents-only", action="store_true")
    parser.add_argument("--force-directory", action="store_true")
    parser.add_argument(
        "--local-dataset-root",
        default=r"dataset/caltech-101/101_ObjectCategories",
    )
    return parser.parse_args()


def api_session():
    import requests

    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def resolve_api_key() -> str:
    for name in ("VIOLAS_VLM_API_KEY", "ARK_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    raise RuntimeError("Set VIOLAS_VLM_API_KEY or ARK_API_KEY")


def call_llm(prompt: str) -> tuple[str, dict[str, Any]]:
    response = api_session().post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {resolve_api_key()}"},
        json={
            "model": MODEL,
            "temperature": 0.0,
            "seed": 42,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"].get("content", ""), {
        "response_id": payload.get("id"),
        "model": payload.get("model"),
        "usage": payload.get("usage"),
    }


def extract_json(text: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response does not contain a JSON object")
    return json.loads(stripped[start : end + 1])


def compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity": schema["entity"],
        "facets": [
            {
                "id": facet["id"],
                "values": [value["id"] for value in facet["values"]],
            }
            for facet in schema["facets"]
        ],
    }


def prompt_for_query(row: dict[str, Any], schema: dict[str, Any], error: str | None) -> str:
    repair = f"\nPrevious output failed validation: {error}\nReturn a corrected object." if error else ""
    return f"""You compile one Chinese natural-language retrieval request into a strict JSON intent.

The authoritative entity scope is already resolved as {row['entity_hint']!r}; copy it exactly.
Supported schema:
{json.dumps(compact_schema(schema), ensure_ascii=False, indent=2)}

Rules:
1. Return JSON only, with exactly the fields shown below.
2. hard_filters contains only explicit mandatory facet=value conditions. Use only schema ids and values.
3. facet_dimension is null unless the user explicitly requests variety/coverage. If enabled it must be exactly one supported facet id.
4. diversity_level is the default common preference level. It is off when facet_dimension is null; otherwise choose light, balanced, or strong. Never output a numeric lambda.
5. category_preferences is [] for a broad request to cover all known values once. If the user explicitly names values or quotas, output one row per named value with a non-negative quota and a configured weight_level.
6. The first version supports exactly one diversity dimension. If the request explicitly requires more than one, put a reason in unsupported instead of silently relaxing it.
7. embedding_text_en is a concise English CLIP text query that preserves the requested visual content.
8. top_k defaults to 3.

Required shape:
{{
  "entity_scope": "{row['entity_hint']}",
  "hard_filters": [{{"key": "facet_id", "op": "eq", "value": "value_id"}}],
  "facet_dimension": "facet_id or null",
  "diversity_level": "off|light|balanced|strong",
  "category_preferences": [{{"value": "facet_value", "quota": 1, "weight_level": "light|balanced|strong"}}],
  "top_k": 3,
  "embedding_text_en": "English visual query",
  "unsupported": []
}}

Natural-language request:
{row['natural_query']}
{repair}"""


def validate_intent(intent: dict[str, Any], row: dict[str, Any], schema: dict[str, Any]) -> None:
    required = {
        "entity_scope",
        "hard_filters",
        "facet_dimension",
        "diversity_level",
        "category_preferences",
        "top_k",
        "embedding_text_en",
        "unsupported",
    }
    if set(intent) != required:
        raise ValueError(f"fields must be exactly {sorted(required)}")
    if intent["entity_scope"] != row["entity_hint"]:
        raise ValueError("entity_scope must equal the authoritative hint")
    facets = {f["id"]: {v["id"] for v in f["values"]} for f in schema["facets"]}
    if not isinstance(intent["hard_filters"], list):
        raise ValueError("hard_filters must be a list")
    for condition in intent["hard_filters"]:
        if set(condition) != {"key", "op", "value"} or condition["op"] != "eq":
            raise ValueError("each hard filter must be an eq condition")
        if condition["key"] not in facets or condition["value"] not in facets[condition["key"]]:
            raise ValueError(f"unsupported hard filter {condition}")
    dimension = intent["facet_dimension"]
    if dimension is not None and dimension not in facets:
        raise ValueError(f"unsupported facet dimension {dimension!r}")
    if intent["diversity_level"] not in LAMBDA_BY_LEVEL:
        raise ValueError("invalid diversity_level")
    if dimension is None and intent["diversity_level"] != "off":
        raise ValueError("null dimension requires diversity_level=off")
    if dimension is not None and intent["diversity_level"] == "off":
        raise ValueError("enabled dimension requires a non-off level")
    preferences = intent["category_preferences"]
    if not isinstance(preferences, list):
        raise ValueError("category_preferences must be a list")
    if dimension is None and preferences:
        raise ValueError("category_preferences requires a facet dimension")
    seen_values: set[str] = set()
    for preference in preferences:
        if set(preference) != {"value", "quota", "weight_level"}:
            raise ValueError("invalid category preference fields")
        value = preference["value"]
        if value in seen_values:
            raise ValueError(f"duplicate category preference {value!r}")
        seen_values.add(value)
        if value not in facets[dimension]:
            raise ValueError(f"unsupported category preference value {value!r}")
        if not isinstance(preference["quota"], int) or preference["quota"] < 0:
            raise ValueError("category preference quota must be a non-negative integer")
        if preference["weight_level"] not in {"light", "balanced", "strong"}:
            raise ValueError("invalid category preference weight_level")
    if not isinstance(intent["top_k"], int) or intent["top_k"] < 0:
        raise ValueError("top_k must be a non-negative integer")
    if not isinstance(intent["embedding_text_en"], str) or not intent["embedding_text_en"].strip():
        raise ValueError("embedding_text_en must be nonempty")
    if not isinstance(intent["unsupported"], list):
        raise ValueError("unsupported must be a list")


def parse_one(row: dict[str, Any], schema: dict[str, Any], max_attempts: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    error: str | None = None
    for attempt in range(1, max_attempts + 1):
        prompt = prompt_for_query(row, schema, error)
        text, meta = call_llm(prompt)
        record = {"attempt": attempt, "prompt": prompt, "response": text, **meta}
        try:
            intent = extract_json(text)
            validate_intent(intent, row, schema)
            record["valid"] = True
            attempts.append(record)
            return intent, attempts
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            record.update({"valid": False, "error": error})
            attempts.append(record)
    raise RuntimeError(f"{row['id']} failed intent validation after {max_attempts} attempts: {error}")


def load_or_parse_intents(args: argparse.Namespace, cache: dict[str, Any]) -> dict[str, Any]:
    if args.intent_output.exists() and not args.force_intents:
        return json.loads(args.intent_output.read_text(encoding="utf-8"))
    source = json.loads(args.input.read_text(encoding="utf-8"))
    rows = source["queries"]
    records: dict[str, dict[str, Any]] = {}
    raw: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(parse_one, row, cache["schemas"][row["entity_hint"]], args.max_attempts): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            intent, attempts = future.result()
            records[row["id"]] = {
                **row,
                "intent": intent,
                "semantic_text_en": intent["embedding_text_en"],
            }
            raw[row["id"]] = attempts
            print(f"INTENT {row['id']}: {intent}", flush=True)
    document = {
        "format": "caltech101-guideline-intent-qwen-soft-quota-v2",
        "model": MODEL,
        "temperature": 0.0,
        "seed": 42,
        "lambda_mapping": LAMBDA_BY_LEVEL,
        "records": [records[row["id"]] for row in rows],
    }
    args.intent_output.parent.mkdir(parents=True, exist_ok=True)
    args.intent_output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.raw_output.open("w", encoding="utf-8") as handle:
        for row in rows:
            for attempt in raw[row["id"]]:
                handle.write(json.dumps({"id": row["id"], **attempt}, ensure_ascii=False) + "\n")
    return document


from violas.retrieval.directory import _geometric_summary


from violas.retrieval.directory import _angular_similarity_upper


from violas.retrieval.directory import build_semantic_directory


def load_or_build_directory(args: argparse.Namespace, cache: dict[str, Any]) -> dict[str, Any]:
    if args.directory_cache.exists() and not args.force_directory:
        print(f"LOAD semantic directory: {args.directory_cache}", flush=True)
        with args.directory_cache.open("rb") as handle:
            return pickle.load(handle)
    started = time.perf_counter()
    directory = build_semantic_directory(cache)
    args.directory_cache.parent.mkdir(parents=True, exist_ok=True)
    with args.directory_cache.open("wb") as handle:
        pickle.dump(directory, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"BUILT semantic directory: {directory['node_count']} nodes, "
        f"{directory['object_count']} objects in {(time.perf_counter()-started)*1000:.1f} ms",
        flush=True,
    )
    return directory


from violas.retrieval.objective import resolve_plan


from violas.retrieval.objective import materialize_plan_preferences


from violas.retrieval.objective import category_preference


from violas.retrieval.objective import satisfies


from violas.retrieval.objective import eligible_ids


from violas.retrieval.objective import score_object


from violas.retrieval.objective import label_of


from violas.retrieval.objective import select_exact


from violas.retrieval.objective import objective


def self_test_select() -> None:
    rng = np.random.default_rng(42)
    fake_objects: dict[str, dict[str, Any]] = {}
    for n in range(1, 9):
        for trial in range(40):
            scored: dict[str, float] = {}
            for i in range(n):
                oid = f"o{i}"
                scored[oid] = float(rng.random())
                fake_objects[oid] = {"description": {"facet_values": {"d": ["a", "b", UNKNOWN][i % 3]}}}
            k = min(4, n)
            lam = [0.0, 0.05, 0.15][trial % 3]
            preferences = {
                "a": {"lambda": lam, "quota": 1 + trial % 2},
                "b": {"lambda": lam / 2.0, "quota": 1 + trial % 3},
            }
            actual = select_exact(scored, fake_objects, "d", k, lam, preferences)
            actual_value = objective(actual, scored, fake_objects, "d", lam, preferences)
            best = max(
                objective(combo, scored, fake_objects, "d", lam, preferences)
                for combo in itertools.combinations(scored, k)
            )
            if not math.isclose(actual_value, best, rel_tol=1e-9, abs_tol=1e-9):
                raise AssertionError((n, trial, actual_value, best, actual))


@dataclass
class SearchState:
    H: dict[int, float] = field(default_factory=dict)
    D: set[int] = field(default_factory=set)
    X: set[int] = field(default_factory=set)
    R: dict[tuple[int, str], list[str]] = field(default_factory=dict)
    offsets: dict[tuple[int, str], int] = field(default_factory=dict)
    E: dict[str, int] = field(default_factory=dict)
    P: dict[str, float] = field(default_factory=dict)
    S: list[str] = field(default_factory=list)
    cache: set[str] = field(default_factory=set)
    nav_used: int = 0
    score_used: int = 0
    center_scores: int = 0
    neighbor_reads: int = 0
    metadata_objects: int = 0
    semantic_injections: int = 0
    t_scores: dict[int, float] = field(default_factory=dict)
    u_scores: dict[int, float] = field(default_factory=dict)
    subset_u_scores: dict[tuple[int, str], float] = field(default_factory=dict)
    subset_t_scores: dict[tuple[int, str], float] = field(default_factory=dict)
    representative_scores: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)


def interleaved_search(
    query: np.ndarray,
    plan: dict[str, Any],
    cache: dict[str, Any],
    directory: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    nav_budget: int,
    score_budget: int,
    beam_width: int,
    batch_size: int,
    execution_profile: str = "paper",
) -> tuple[list[str], dict[str, float], dict[str, Any]]:
    vm = cache["vectormap"]
    eligible = eligible_ids(objects, plan)
    dimension = plan["facet_dimension"]
    routing_dimension = dimension or next(
        (condition["key"] for condition in plan["hard_filters"]), None
    )
    lam = plan["lambda"]
    preferences = plan["category_preferences"]
    state = SearchState()

    if execution_profile not in {"paper", "subset-v2"}:
        raise ValueError(f"unknown execution profile {execution_profile!r}")
    if execution_profile == "subset-v2" and directory.get("format") != "violas-semantic-entry-directory-v2":
        raise ValueError("interleaved_search requires semantic directory v2")

    def summary_for(node_id: int, value: str) -> dict[str, Any]:
        if dimension is None:
            return directory["node_summaries"][node_id]
        return directory["subset_summaries"].get(
            (node_id, plan["entity_scope"], dimension, value),
            directory["node_summaries"][node_id],
        )

    def ordered_subset(node_id: int, value: str, ids: list[str]) -> list[str]:
        if execution_profile == "paper":
            return sorted(ids)
        summary = summary_for(node_id, value)
        allowed = set(ids)
        representatives = [oid for oid in summary["representative_ids"] if oid in allowed]
        representative_set = set(representatives)
        return representatives + sorted(oid for oid in ids if oid not in representative_set)

    def covered_values() -> set[str]:
        return {
            label_of(objects, oid, dimension)
            for oid in state.S
            if dimension is not None and label_of(objects, oid, dimension) != UNKNOWN
        }

    def refresh_selection() -> None:
        state.S = select_exact(
            state.P,
            objects,
            dimension,
            plan["top_k"],
            lam,
            preferences,
        )

    def remaining_category_bonus(value: str) -> float:
        weight, quota = category_preference(preferences, value, dimension, lam)
        if weight == 0.0 or quota <= 0:
            return 0.0
        selected_count = sum(
            label_of(objects, oid, dimension) == value for oid in state.S
        )
        return weight if selected_count < quota else 0.0

    def discover(node_id: int) -> None:
        if node_id in state.D:
            return
        center = directory["centers"][node_id]
        t_score = float((1.0 + float(np.dot(query, center))) / 2.0)
        if execution_profile == "paper":
            u_score = min(
                1.0,
                float(
                    (
                        1.0
                        + float(np.dot(query, center))
                        + float(directory["radii"][node_id])
                    )
                    / 2.0
                ),
            )
        else:
            u_score = _angular_similarity_upper(query, directory["node_summaries"][node_id])
        state.t_scores[node_id] = t_score
        state.u_scores[node_id] = u_score
        state.center_scores += 1
        state.D.add(node_id)
        subsets: dict[str, list[str]] = defaultdict(list)
        for oid in directory["node_members"][node_id]:
            state.metadata_objects += 1
            if oid in eligible and oid not in state.cache:
                subsets[label_of(objects, oid, routing_dimension)].append(oid)
        for value, ids in subsets.items():
            key = (node_id, value)
            state.R[key] = ordered_subset(node_id, value, ids)
            state.offsets[key] = 0
            if execution_profile == "paper":
                state.subset_u_scores[key] = u_score
                state.subset_t_scores[key] = t_score
            else:
                summary = summary_for(node_id, value)
                state.subset_u_scores[key] = _angular_similarity_upper(query, summary)
                state.subset_t_scores[key] = score_object(query, summary["unit_center"])
        if node_id not in state.X:
            state.H[node_id] = t_score
            if len(state.H) > beam_width:
                evicted = min(state.H, key=lambda nid: (state.H[nid], -nid))
                state.H.pop(evicted, None)
        state.trace.append({
            "event": "discover",
            "node": node_id,
            "t": round(t_score, 4),
            "u": round(u_score, 4),
            "subset_u": {
                value: round(state.subset_u_scores[(node_id, value)], 4)
                for value in sorted(subsets)
            },
            "subset_t": {
                value: round(state.subset_t_scores[(node_id, value)], 4)
                for value in sorted(subsets)
            },
            "representatives": {} if execution_profile == "paper" else {
                value: [
                    oid
                    for oid in summary_for(node_id, value)["representative_ids"]
                    if oid in set(ids)
                ]
                for value, ids in sorted(subsets.items())
            },
            "eligible_members": sum(len(ids) for ids in subsets.values()),
        })

    def navigate(node_id: int, source: str) -> bool:
        if node_id in state.X or state.nav_used >= nav_budget:
            return False
        discover(node_id)
        state.X.add(node_id)
        state.H.pop(node_id, None)
        state.nav_used += 1
        if source == "semantic_entry":
            state.semantic_injections += 1
        neighbors = list(dict.fromkeys(
            list(vm._hdmg_embedding_edges[node_id]) + list(vm._hdmg_semantic_edges[node_id])
        ))
        state.neighbor_reads += len(neighbors)
        for neighbor in neighbors:
            discover(int(neighbor))
        state.trace.append({
            "event": "navigate",
            "node": node_id,
            "source": source,
            "nav_used": state.nav_used,
            "neighbors": len(neighbors),
        })
        return True

    def clean_iterators() -> None:
        for key in list(state.R):
            ids = state.R[key]
            offset = state.offsets[key]
            while offset < len(ids) and ids[offset] in state.cache:
                offset += 1
            state.offsets[key] = offset
            if offset >= len(ids):
                state.R.pop(key, None)
                state.offsets.pop(key, None)
                state.subset_u_scores.pop(key, None)
                state.subset_t_scores.pop(key, None)

    def member_key(key: tuple[int, str], covered: set[str]) -> tuple[float, float, int, str]:
        node_id, value = key
        bonus = remaining_category_bonus(value)
        if execution_profile == "paper":
            return (state.u_scores[node_id] + bonus, 0.0, -node_id, value)
        return (state.subset_u_scores[key] + bonus, state.subset_t_scores[key], -node_id, value)

    def entry_key(value: str, node_id: int, covered: set[str]) -> tuple[float, float, int, str]:
        if execution_profile == "paper":
            center = directory["centers"][node_id]
            u = min(
                1.0,
                float(
                    (
                        1.0
                        + float(np.dot(query, center))
                        + float(directory["radii"][node_id])
                    )
                    / 2.0
                ),
            )
            t = 0.0
        else:
            summary = directory["subset_summaries"].get(
                (node_id, plan["entity_scope"], routing_dimension, value),
                directory["node_summaries"][node_id],
            )
            u = _angular_similarity_upper(query, summary)
            t = score_object(query, summary["unit_center"])
        bonus = remaining_category_bonus(value)
        return (u + bonus, t, -node_id, value)

    def score_batch(key: tuple[int, str]) -> bool:
        ids = state.R.get(key, [])
        offset = state.offsets.get(key, 0)
        newly_scored: list[str] = []
        while offset < len(ids) and len(newly_scored) < batch_size and state.score_used < score_budget:
            oid = ids[offset]
            offset += 1
            if oid in state.cache:
                continue
            state.cache.add(oid)
            state.P[oid] = score_object(query, objects[oid]["normalized"])
            state.score_used += 1
            if execution_profile == "subset-v2" and oid in summary_for(key[0], key[1])["representative_ids"]:
                state.representative_scores += 1
            newly_scored.append(oid)
        state.offsets[key] = offset
        clean_iterators()
        if not newly_scored:
            return False
        refresh_selection()
        state.trace.append({
            "event": "score_batch",
            "node": key[0],
            "value": key[1],
            "new_scores": len(newly_scored),
            "scored_ids": newly_scored,
            "scored_items": [
                {
                    "object_id": oid,
                    "score": round(state.P[oid], 6),
                    "value": label_of(objects, oid, routing_dimension),
                }
                for oid in newly_scored
            ],
            "score_used": state.score_used,
            "selected_ids": list(state.S),
            "selected_items": [
                {
                    "object_id": oid,
                    "score": round(state.P[oid], 6),
                    "value": label_of(objects, oid, routing_dimension),
                }
                for oid in state.S
            ],
            "selected_values": sorted(covered_values()),
            "objective": round(
                objective(state.S, state.P, objects, dimension, lam, preferences), 5
            ),
        })
        return True

    # Existing group/entity entry remains the geometric starting point.
    initial = vm._hdmg_key_entries.get(plan["entity_scope"], vm._hdmg_global_entry)
    discover(int(initial))

    # Prepare one eligible semantic entry for every rewarded or hard-required value.
    relevant_values = {
        value
        for value, preference in preferences.items()
        if float(preference["lambda"]) > 0 and int(preference["quota"]) > 0
    }
    if routing_dimension is not None:
        relevant_values.update(
            condition["value"]
            for condition in plan["hard_filters"]
            if condition["key"] == routing_dimension
        )
    posting_values = (
        directory["postings"]
        .get(plan["entity_scope"], {})
        .get(routing_dimension, {})
        if routing_dimension is not None
        else {}
    )
    for value in sorted(relevant_values):
        entries = posting_values.get(value, [])
        if entries:
            best: tuple[float, int, int] | tuple[int, int] | None = None
            for row in entries:
                count = sum(oid in eligible for oid in row["object_ids"])
                if not count:
                    continue
                if execution_profile == "paper":
                    candidate = (count, -row["node_id"])
                else:
                    candidate = (
                        _angular_similarity_upper(query, row["summary"]),
                        count,
                        -row["node_id"],
                    )
                if best is None or candidate > best:
                    best = candidate
            if best is not None:
                state.E[value] = -best[-1]

    rounds = 0
    while state.score_used < score_budget:
        rounds += 1
        if rounds > 10000:
            raise RuntimeError("interleaved search did not make bounded progress")
        progressed = False
        if state.nav_used < nav_budget and state.H:
            node_id = max(state.H, key=lambda nid: (state.H[nid], -nid))
            progressed |= navigate(node_id, "geometry")

        clean_iterators()
        for value, node_id in list(state.E.items()):
            if node_id in state.X:
                state.E.pop(value, None)
        covered = covered_values()
        best_r_key = max(state.R, key=lambda key: member_key(key, covered)) if state.R else None
        best_e_value = max(state.E, key=lambda value: entry_key(value, state.E[value], covered)) if state.E else None
        entry_outranks_member = False
        if best_e_value is not None:
            if best_r_key is None:
                entry_outranks_member = True
            elif execution_profile == "paper":
                entry_outranks_member = (
                    entry_key(best_e_value, state.E[best_e_value], covered)[0]
                    > member_key(best_r_key, covered)[0]
                )
            else:
                entry_outranks_member = (
                    entry_key(best_e_value, state.E[best_e_value], covered)
                    > member_key(best_r_key, covered)
                )
        if (
            best_e_value is not None
            and state.nav_used < nav_budget
            and entry_outranks_member
        ):
            node_id = state.E.pop(best_e_value)
            progressed |= navigate(node_id, "semantic_entry")

        clean_iterators()
        covered = covered_values()
        if state.R and state.score_used < score_budget:
            best_r_key = max(state.R, key=lambda key: member_key(key, covered))
            progressed |= score_batch(best_r_key)
        if not progressed:
            break

    if state.score_used >= score_budget:
        stop_reason = "score_budget_exhausted"
    elif not state.H and not state.R and not state.E:
        stop_reason = "frontier_exhausted"
    elif state.nav_used >= nav_budget and not state.R:
        stop_reason = "navigation_budget_exhausted_and_no_scoring_work"
    else:
        stop_reason = "no_progress"
    refresh_selection()
    stats = {
        "guarantee": "candidate_pool_optimal; global_search_approximate",
        "execution_profile": execution_profile,
        "stop_reason": stop_reason,
        "eligible_count": len(eligible),
        "candidate_pool_size": len(state.P),
        "nav_expansions": state.nav_used,
        "unique_member_scores": state.score_used,
        "representative_scores": state.representative_scores,
        "center_scores": state.center_scores,
        "neighbor_reads": state.neighbor_reads,
        "metadata_objects": state.metadata_objects,
        "semantic_entry_injections": state.semantic_injections,
        "discovered_clusters": len(state.D),
        "trace": state.trace,
    }
    return state.S, state.P, stats


def group_scan_search(
    query: np.ndarray,
    plan: dict[str, Any],
    groups: dict[str, tuple[str, ...]],
    objects: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, float], dict[str, Any]]:
    """Resolve the entity key and score every hard-valid member of that group."""
    member_ids = groups.get(plan["entity_scope"], ())
    scored = {
        oid: score_object(query, objects[oid]["normalized"])
        for oid in member_ids
        if satisfies(objects[oid]["description"], plan)
    }
    selected = select_exact(
        scored,
        objects,
        plan["facet_dimension"],
        plan["top_k"],
        plan["lambda"],
        plan["category_preferences"],
    )
    return selected, scored, {
        "execution": "entity_key_group_scan",
        "group_key": plan["entity_scope"],
        "group_size": len(member_ids),
        "eligible_count": len(scored),
        "unique_member_scores": len(scored),
        "candidate_pool_size": len(scored),
    }


def exhaustive_oracle(
    query: np.ndarray,
    plan: dict[str, Any],
    objects: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, float]]:
    ids = eligible_ids(objects, plan)
    scored = {oid: score_object(query, objects[oid]["normalized"]) for oid in ids}
    selected = select_exact(
        scored,
        objects,
        plan["facet_dimension"],
        plan["top_k"],
        plan["lambda"],
        plan["category_preferences"],
    )
    return selected, scored


def milvus_search(
    client: Any,
    collection: str,
    query: np.ndarray,
    plan: dict[str, Any],
    cache: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    candidate_k: int,
    nprobe: int,
) -> tuple[list[str], dict[str, float], dict[str, Any]]:
    legacy_intent = {"entity": plan["entity_scope"], "must": plan["hard_filters"]}
    # Keep the logical ARRAY predicate used by the Milvus baseline.
    expression = retrieval.milvus_filter(legacy_intent, None)
    response = client.search(
        collection_name=collection,
        data=[np.asarray(query, dtype=np.float32).tolist()],
        anns_field="vector",
        filter=expression,
        limit=candidate_k,
        output_fields=["object_id", "facet_tokens"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": nprobe}},
    )
    hits = response[0] if response else []
    ids: list[str] = []
    for hit in hits:
        entity = hit.get("entity", {})
        oid = entity.get("object_id") or hit.get("object_id")
        if oid in objects and satisfies(objects[oid]["description"], plan):
            ids.append(oid)
    scored = {oid: score_object(query, objects[oid]["normalized"]) for oid in dict.fromkeys(ids)}
    selected = select_exact(
        scored,
        objects,
        plan["facet_dimension"],
        plan["top_k"],
        plan["lambda"],
        plan["category_preferences"],
    )
    return selected, scored, {"filter": expression, "ann_hit_count": len(scored), "nprobe": nprobe}


def result_rows(
    ids: list[str], scored: dict[str, float], objects: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "object_id": oid,
            "score": round(float(scored[oid]), 6),
            "facet_values": objects[oid]["description"].get("facet_values", {}),
            "uncertain": bool(objects[oid]["description"].get("uncertain", False)),
        }
        for oid in sorted(ids, key=lambda value: (-scored[value], value))
    ]


def method_metrics(
    selected: list[str],
    scored: dict[str, float],
    oracle_selected: list[str],
    oracle_value: float,
    plan: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    eligible_count: int,
) -> dict[str, Any]:
    value = objective(
        selected,
        scored,
        objects,
        plan["facet_dimension"],
        plan["lambda"],
        plan["category_preferences"],
    )
    labels = sorted({
        label_of(objects, oid, plan["facet_dimension"])
        for oid in selected
        if plan["facet_dimension"] is not None and label_of(objects, oid, plan["facet_dimension"]) != UNKNOWN
    })
    return {
        "objective": round(value, 6),
        "objective_ratio": round(value / oracle_value, 6) if oracle_value > 0 else None,
        "objective_gap": round(oracle_value - value, 6),
        "mean_embedding_similarity": round(sum(scored[oid] for oid in selected) / len(selected), 6) if selected else None,
        "known_category_coverage": len(labels),
        "covered_values": labels,
        "fill_rate": round(len(selected) / min(plan["top_k"], eligible_count), 6) if eligible_count else None,
        "hard_condition_validity": 1.0 if all(satisfies(objects[oid]["description"], plan) for oid in selected) else 0.0,
        "oracle_object_recall": round(len(set(selected) & set(oracle_selected)) / len(oracle_selected), 6) if oracle_selected else None,
    }


def run_benchmark(
    args: argparse.Namespace,
    query_doc: dict[str, Any],
    cache: dict[str, Any],
    directory: dict[str, Any],
) -> dict[str, Any]:
    from pymilvus import MilvusClient

    objects = retrieval.make_object_map(cache)
    vectors = retrieval.encode_queries(query_doc["records"])
    client = MilvusClient(str(args.milvus_uri.resolve()))
    client.load_collection(args.collection)
    rows: list[dict[str, Any]] = []
    for record in query_doc["records"]:
        plan = materialize_plan_preferences(resolve_plan(record["intent"]), directory)
        if plan["unsupported"]:
            rows.append({**record, "resolved_plan": plan, "status": "unsupported", "methods": {}})
            continue
        query = vectors[record["id"]]
        eligible_count = len(eligible_ids(objects, plan))

        started = time.perf_counter()
        oracle_ids, oracle_scored = exhaustive_oracle(query, plan, objects)
        oracle_ms = (time.perf_counter() - started) * 1000
        oracle_value = objective(
            oracle_ids,
            oracle_scored,
            objects,
            plan["facet_dimension"],
            plan["lambda"],
            plan["category_preferences"],
        )

        started = time.perf_counter()
        violas_ids, violas_scored, violas_stats = interleaved_search(
            query,
            plan,
            cache,
            directory,
            objects,
            args.nav_budget,
            args.score_budget,
            args.beam_width,
            args.batch_size,
            args.execution_profile,
        )
        violas_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        milvus_ids, milvus_scored, milvus_stats = milvus_search(
            client,
            args.collection,
            query,
            plan,
            cache,
            objects,
            args.milvus_candidate_k,
            args.milvus_nprobe,
        )
        milvus_ms = (time.perf_counter() - started) * 1000

        methods: dict[str, Any] = {}
        for name, ids, scored, elapsed, stats in (
            ("oracle", oracle_ids, oracle_scored, oracle_ms, {"scored_objects": len(oracle_scored), "guarantee": "full-corpus exact"}),
            ("violas", violas_ids, violas_scored, violas_ms, violas_stats),
            ("milvus", milvus_ids, milvus_scored, milvus_ms, milvus_stats),
        ):
            methods[name] = {
                "latency_ms": round(elapsed, 3),
                "results": result_rows(ids, scored, objects),
                "metrics": method_metrics(ids, scored, oracle_ids, oracle_value, plan, objects, eligible_count),
                "stats": stats,
            }
        rows.append({
            **record,
            "resolved_plan": plan,
            "status": "ok",
            "eligible_count": eligible_count,
            "methods": methods,
        })
        print(
            f"DONE {record['id']}: d={plan['facet_dimension']} lambda={plan['lambda']:.2f} "
            f"oracle={oracle_value:.4f} Violas={methods['violas']['metrics']['objective_ratio']:.3f} "
            f"Milvus={methods['milvus']['metrics']['objective_ratio']:.3f}",
            flush=True,
        )
    ok_rows = [row for row in rows if row["status"] == "ok"]
    aggregate: dict[str, Any] = {}
    for name in ("violas", "milvus"):
        aggregate[name] = {
            "mean_objective_ratio": round(float(np.mean([r["methods"][name]["metrics"]["objective_ratio"] for r in ok_rows])), 6),
            "mean_oracle_object_recall": round(float(np.mean([r["methods"][name]["metrics"]["oracle_object_recall"] for r in ok_rows])), 6),
            "mean_latency_ms": round(float(np.mean([r["methods"][name]["latency_ms"] for r in ok_rows])), 3),
        }
    return {
        "format": "violas-guideline-interleaved-report-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "intent_model": MODEL,
            "temperature": 0.0,
            "seed": 42,
            "lambda_mapping": LAMBDA_BY_LEVEL,
            "nav_budget": args.nav_budget,
            "score_budget": args.score_budget,
            "beam_width": args.beam_width,
            "batch_size": args.batch_size,
            "execution_profile": args.execution_profile,
            "milvus_candidate_k": args.milvus_candidate_k,
            "milvus_nprobe": args.milvus_nprobe,
            "geometric_clusters_preserved": True,
            "facet_role": "semantic entry directory and dynamic member scheduling",
        },
        "directory": {
            "format": directory["format"],
            "node_count": directory["node_count"],
            "object_count": directory["object_count"],
        },
        "aggregate": aggregate,
        "queries": rows,
    }


def image_uri(local_root: str, object_id: str) -> str:
    from artifact_paths import image_uri as resolve_image_uri
    return resolve_image_uri(local_root, object_id)


def result_cards(method: dict[str, Any], local_root: str, active_dimension: str | None) -> str:
    cards = []
    for rank, row in enumerate(method["results"], 1):
        facet_chips = []
        for key, value in sorted(row["facet_values"].items()):
            active = " active" if key == active_dimension else ""
            facet_chips.append(f'<span class="chip{active}">{html.escape(key)}={html.escape(str(value))}</span>')
        cards.append(f"""
        <article class="result">
          <div class="rank">{rank}</div>
          <img src="{image_uri(local_root, row['object_id'])}" loading="lazy">
          <div><b>{html.escape(row['object_id'])}</b><div class="score">embedding similarity {row['score']:.4f}</div>
          <div class="chips">{''.join(facet_chips)}</div></div>
        </article>""")
    return "".join(cards)


def metric_badges(method: dict[str, Any]) -> str:
    m = method["metrics"]
    return (
        f'<span>objective ratio <b>{m["objective_ratio"]:.3f}</b></span>'
        f'<span>gap <b>{m["objective_gap"]:.3f}</b></span>'
        f'<span>similarity <b>{m["mean_embedding_similarity"]:.3f}</b></span>'
        f'<span>coverage <b>{m["known_category_coverage"]}</b></span>'
        f'<span>oracle recall <b>{m["oracle_object_recall"]:.3f}</b></span>'
        f'<span>latency <b>{method["latency_ms"]:.2f} ms</b></span>'
    )


def render_html(report: dict[str, Any], local_root: str) -> str:
    rows = []
    for row in report["queries"]:
        plan = row["resolved_plan"]
        if row["status"] != "ok":
            rows.append(f'<section class="query"><h2>{row["id"]} · unsupported</h2><p>{html.escape(row["natural_query"])}</p><pre>{html.escape(json.dumps(plan, ensure_ascii=False, indent=2))}</pre></section>')
            continue
        methods = row["methods"]
        trace = methods["violas"]["stats"]["trace"]
        trace_rows = "".join(
            f"<tr><td>{i+1}</td><td>{html.escape(event['event'])}</td><td><code>{html.escape(json.dumps(event, ensure_ascii=False))}</code></td></tr>"
            for i, event in enumerate(trace)
        )
        method_sections = []
        for key, title in (("oracle", "Full-corpus Oracle"), ("violas", "Interleaved Violas"), ("milvus", "Milvus ANN + same SELECT")):
            method = methods[key]
            stats = html.escape(json.dumps(method["stats"], ensure_ascii=False, indent=2))
            method_sections.append(f"""
            <div class="method">
              <h3>{title}</h3><div class="badges">{metric_badges(method)}</div>
              {result_cards(method, local_root, plan['facet_dimension'])}
              <details><summary>execution details</summary><pre>{stats}</pre></details>
            </div>""")
        rows.append(f"""
        <section class="query">
          <h2>{row['id']} · {html.escape(row['natural_query'])}</h2>
          <div class="plan"><b>scope</b> {html.escape(plan['entity_scope'])} · <b>facet</b> {html.escape(str(plan['facet_dimension']))} · <b>level</b> {plan['diversity_level']} · <b>λ</b> {plan['lambda']:.2f} · <b>eligible</b> {row['eligible_count']} · <b>text</b> {html.escape(plan['embedding_text_en'])}</div>
          <details><summary>Qwen intent and resolved plan</summary><pre>{html.escape(json.dumps({'intent': row['intent'], 'plan': plan}, ensure_ascii=False, indent=2))}</pre></details>
          <div class="methods">{''.join(method_sections)}</div>
          <details><summary>Violas search trace ({len(trace)} events)</summary><table><thead><tr><th>#</th><th>event</th><th>state change</th></tr></thead><tbody>{trace_rows}</tbody></table></details>
        </section>""")
    aggregate = report["aggregate"]
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Violas guideline pipeline</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--text:#172033;--muted:#62708a;--line:#dce3ef;--blue:#315fe8;--soft:#edf2ff;--green:#147d64}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Microsoft YaHei",sans-serif}}main{{max-width:1540px;margin:auto;padding:26px}}h1{{margin:0 0 8px}}.subtitle{{color:var(--muted);margin-bottom:18px}}.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}}.summary div{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}}.summary b{{display:block;font-size:22px;color:var(--blue)}}.query{{background:var(--card);border:1px solid var(--line);border-radius:15px;padding:18px;margin:18px 0;box-shadow:0 5px 18px #20304a0a}}.query h2{{font-size:18px;margin:0 0 10px}}.plan{{background:var(--soft);padding:9px 12px;border-radius:8px;margin-bottom:12px;font-size:13px}}.methods{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}.method{{border:1px solid var(--line);border-radius:11px;padding:11px;min-width:0}}.method h3{{margin:0 0 8px;font-size:15px}}.badges{{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px}}.badges span{{font-size:11px;background:#f1f4f9;border-radius:999px;padding:4px 7px}}.result{{display:grid;grid-template-columns:22px 100px minmax(0,1fr);gap:8px;align-items:center;border-top:1px solid var(--line);padding:8px 0;font-size:12px}}.result img{{width:100px;height:76px;object-fit:contain;background:#eef1f5;border-radius:7px}}.rank{{font-weight:800;color:var(--muted)}}.score{{color:var(--muted);margin:3px 0}}.chips{{display:flex;gap:3px;flex-wrap:wrap}}.chip{{font-size:9px;background:#eef1f5;border-radius:999px;padding:2px 5px}}.chip.active{{background:#dce7ff;color:var(--blue);font-weight:700}}details{{margin-top:9px}}summary{{cursor:pointer;color:var(--blue);font-size:12px}}pre{{background:#111827;color:#dbeafe;padding:11px;border-radius:8px;white-space:pre-wrap;overflow:auto;font-size:11px}}table{{width:100%;border-collapse:collapse;font-size:11px;margin-top:8px}}th,td{{border:1px solid var(--line);padding:5px;text-align:left;vertical-align:top}}td code{{white-space:pre-wrap;overflow-wrap:anywhere}}@media(max-width:1100px){{.methods,.summary{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Violas ICDE Guideline Pipeline · 10-query P0</h1>
<div class="subtitle">纯几何 micro-clusters 保持不变；Facet 构建语义入口目录；HDMG 导航、真实成员评分和结果反馈交错执行。</div>
<div class="summary">
  <div><b>{report['directory']['node_count']}</b>geometric HDMG nodes</div>
  <div><b>{report['directory']['object_count']}</b>indexed objects</div>
  <div><b>{aggregate['violas']['mean_objective_ratio']:.3f}</b>Violas mean objective ratio</div>
  <div><b>{aggregate['milvus']['mean_objective_ratio']:.3f}</b>Milvus mean objective ratio</div>
</div>
<details open><summary>Protocol</summary><pre>{html.escape(json.dumps(report['config'], ensure_ascii=False, indent=2))}</pre></details>
{''.join(rows)}
</main></body></html>"""


def main() -> int:
    args = parse_args()
    self_test_select()
    with args.hdmg_cache.open("rb") as handle:
        cache = pickle.load(handle)
    intents = load_or_parse_intents(args, cache)
    if args.compile_intents_only:
        print(f"WROTE {args.intent_output}", flush=True)
        print(f"WROTE {args.raw_output}", flush=True)
        return 0
    directory = load_or_build_directory(args, cache)
    report = run_benchmark(args, intents, cache, directory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    html_path = args.output_dir / "guideline_pipeline_10.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(report, args.local_dataset_root), encoding="utf-8")
    print(f"WROTE {json_path}", flush=True)
    print(f"WROTE {html_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
