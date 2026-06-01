"""Pure-Python IVF-FLAT index used as a lightweight benchmark baseline."""

from typing import Any, List, Tuple

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x / (norm if norm > 1e-9 else 1.0)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return (x / norms).astype(np.float32)


def _kmeans_cosine(
    X: np.ndarray,
    nlist: int,
    max_iter: int = 25,
    seed: int = 42,
) -> np.ndarray:
    """Cluster normalized vectors with cosine k-means and return centers."""
    n_vectors, _ = X.shape
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_vectors, size=min(nlist, n_vectors), replace=False)
    centers = X[idx].copy()
    if centers.shape[0] < nlist:
        pad = np.tile(X[0:1], (nlist - centers.shape[0], 1))
        centers = np.vstack([centers, pad])
    centers = _normalize(centers)

    for _ in range(max_iter):
        sims = X @ centers.T
        labels = np.argmax(sims, axis=1)
        new_centers = np.zeros_like(centers)
        for c in range(nlist):
            mask = labels == c
            if np.sum(mask) == 0:
                new_centers[c] = centers[c]
                continue
            new_centers[c] = np.mean(X[mask], axis=0)
        new_centers = _normalize(new_centers)
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers.astype(np.float32)


def build_ivf_flat_index(
    matrix: np.ndarray,
    img_paths: List[Any],
    keys: List[Any],
    nlist: int = 256,
    max_iter: int = 25,
    seed: int = 42,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Build an IVF-FLAT index.

    Args:
        matrix: Normalized vector matrix with shape (N, D).
        img_paths: Stored item identifiers. Kept for API symmetry with search.
        keys: Stored item keys. Kept for API symmetry with search.
        nlist: Number of IVF buckets.
        max_iter: Maximum cosine k-means iterations.
        seed: Random seed for center initialization.

    Returns:
        A tuple of cluster centers and per-cluster vector-index buckets.
    """
    n_vectors = matrix.shape[0]
    nlist = min(nlist, n_vectors)
    if nlist <= 0:
        return np.zeros((0, matrix.shape[1]), dtype=np.float32), []

    centers = _kmeans_cosine(matrix, nlist, max_iter=max_iter, seed=seed)
    sims = matrix @ centers.T
    labels = np.argmax(sims, axis=1)

    buckets = [np.nonzero(labels == c)[0].astype(np.int64) for c in range(nlist)]
    return centers, buckets


def search_ivf_flat_index(
    vectors: np.ndarray,
    img_paths: List[Any],
    keys: List[Any],
    centers: np.ndarray,
    buckets: List[np.ndarray],
    query_vector: np.ndarray,
    top_k: int = 10,
    nprobe: int = 16,
) -> List[Tuple[Any, Any, float]]:
    """Search an IVF-FLAT index and return (item_id, key, cosine_distance) tuples."""
    q = _normalize(np.asarray(query_vector, dtype=np.float32).flatten())
    nlist = centers.shape[0]
    nprobe = min(nprobe, nlist)

    center_sims = centers @ q
    top_centers = np.argsort(-center_sims)[:nprobe]

    candidate_idx = np.concatenate([buckets[c] for c in top_centers]) if top_centers.size else np.array([], dtype=np.int64)
    if candidate_idx.size == 0:
        return []

    cand_vecs = vectors[candidate_idx]
    dists = 1.0 - (cand_vecs @ q)
    k = min(top_k, len(dists))
    if k <= 0:
        return []
    if k == len(dists):
        top_local = np.argsort(dists)
    else:
        part = np.argpartition(dists, k - 1)[:k]
        top_local = part[np.argsort(dists[part])]

    out = []
    for j in top_local:
        i = int(candidate_idx[j])
        out.append((img_paths[i], keys[i], float(dists[j])))
    return out
