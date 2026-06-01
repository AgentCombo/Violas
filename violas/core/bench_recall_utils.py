"""
Shared recall helpers for vector-group benchmarks.

When beta == 1, mixed search is purely key/semantic; recall is computed on
normalized key identity (not per-document img_path / doc id).

For beta == 1, per-method recall is **hits@k / k**: among the top-k retrieved
items, count how many have a normalized key that appears in the oracle GT
key set, then divide by ``top_k``. This avoids the degenerate case where GT
contains only one distinct key (e.g. three mandolin images) and set-intersection
with retrieved keys would use a denominator of 1.
"""

from __future__ import annotations

import math
import re

# Sub-cluster keys like "accordion-0003" -> "accordion" (matches VectorMap grouping)
_KEY_SUFFIX_RE = re.compile(r"^(.+)-(\d{4})$")


def norm_key_for_recall(k) -> str:
    """Map raw index keys to a comparable label (strip -NNNN sub-key suffix if present)."""
    if k is None:
        return ""
    s = str(k).strip()
    m = _KEY_SUFFIX_RE.match(s)
    return m.group(1) if m else s


def lookup_key_vector(key, key_vectors: dict):
    """Resolve key embedding for mixed score: prefer exact key, else class key (strip -NNNN)."""
    if key is None or not key_vectors:
        return None
    k = str(key).strip()
    if not k:
        return None
    if k in key_vectors:
        return key_vectors[k]
    nk = norm_key_for_recall(k)
    if nk and nk in key_vectors:
        return key_vectors[nk]
    return None


def is_beta_pure_key(beta: float) -> bool:
    """True when mixed score uses only key distance (beta == 1)."""
    return math.isclose(float(beta), 1.0, rel_tol=0.0, abs_tol=1e-9)


def padded_norm_keys_from_tuples(items, norm_key, top_k: int) -> list[str]:
    """(path, key, ...) rows -> length-``top_k`` list of norm keys; ``\"\"`` if missing."""
    out: list[str] = []
    for i in range(min(len(items), top_k)):
        it = items[i]
        if isinstance(it, (list, tuple)) and len(it) > 1 and it[1] is not None:
            out.append(norm_key(it[1]))
        else:
            out.append("")
    while len(out) < top_k:
        out.append("")
    return out[:top_k]


def padded_norm_keys_from_search_results(results, norm_key, top_k: int) -> list[str]:
    """SearchResult-like rows with ``.key`` -> length-``top_k`` norm keys."""
    out: list[str] = []
    for i in range(min(len(results), top_k)):
        r = results[i]
        k = getattr(r, "key", None)
        out.append(norm_key(k) if k is not None else "")
    while len(out) < top_k:
        out.append("")
    return out[:top_k]


def recall_key_hit_rate_at_k(gt_key_set: set, ordered_norm_keys: list[str], top_k: int) -> float:
    """Hits among the first ``top_k`` ranks whose norm key is in ``gt_key_set``, divided by ``top_k``."""
    if top_k <= 0:
        return 0.0
    if not gt_key_set:
        return 0.0
    hits = 0
    for i in range(top_k):
        if i < len(ordered_norm_keys) and ordered_norm_keys[i] in gt_key_set:
            hits += 1
    return hits / float(top_k)


def key_hit_slots_for_json(gt_key_set: set, ordered_norm_keys: list[str], top_k: int) -> list:
    """Per rank: normalized key if hit else None (for detailed JSON)."""
    slot: list = []
    for i in range(top_k):
        if i < len(ordered_norm_keys) and ordered_norm_keys[i] in gt_key_set:
            slot.append(ordered_norm_keys[i])
        else:
            slot.append(None)
    return slot
