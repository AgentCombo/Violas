"""Minimal Violas example with synthetic vectors.

This example does not require any dataset, embedding model, or external vector
database. It demonstrates the current public VectorMap API: create a semantic
group, then run entity-scoped retrieval.
"""

import numpy as np

from violas import VectorMap


def make_vectors(name: str, center: np.ndarray, count: int = 6) -> tuple[list[np.ndarray], list[dict]]:
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    vectors = [center + 0.04 * rng.normal(size=center.shape) for _ in range(count)]
    descriptions = [{"text": f"{name} item {i}"} for i in range(count)]
    return vectors, descriptions


def main() -> None:
    vector_map = VectorMap()

    for key, center in {
        "rhino": np.array([1.0, 0.0, 0.0, 0.0]),
        "duck": np.array([0.0, 1.0, 0.0, 0.0]),
        "norfloxacin": np.array([0.0, 0.0, 1.0, 0.0]),
    }.items():
        vectors, descriptions = make_vectors(key, center)
        vector_map.create_group(
            key=key,
            group_name=key,
            representative=np.mean(vectors, axis=0),
            rep_description=f"{key} representative",
            vectors=vectors,
            descriptions=descriptions,
            vector_type="demo",
            group_type="synthetic",
        )

    query = np.array([0.95, 0.05, 0.0, 0.0])
    results = vector_map.search_entity(query, key="rhino", top_k=3)

    for rank, result in enumerate(results, start=1):
        description = result.group.descriptions[result.vector_idx]
        print(
            f"{rank}. key={result.key} "
            f"group={result.group.group_name} "
            f"distance={result.distance:.4f} "
            f"text={description.get('text')}"
        )


if __name__ == "__main__":
    main()
