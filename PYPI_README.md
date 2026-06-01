# Violas

Violas is a lightweight Python package for grouped vector retrieval.

It exposes the core storage and retrieval primitives from the Violas
project without bundling the benchmark pipelines, datasets, or experiment
artifacts. The package is meant to be a small installable surface for:

- semantic-keyed vector groups
- representative-based retrieval
- relation-aware expansion
- HDMG-backed mixed semantic-vector search

## Install

```bash
pip install violas
```

Optional FAISS-backed acceleration:

```bash
pip install "violas[faiss]"
```

## Quick Start

```python
import numpy as np

from violas import VectorMap

vectors = [np.random.rand(4) for _ in range(5)]
vm = VectorMap()
vm.create_group(
    key="example",
    group_name="demo",
    representative=np.mean(vectors, axis=0),
    rep_description="demo representative",
    vectors=vectors,
    descriptions=[{"text": f"item {i}"} for i in range(5)],
)

query = np.random.rand(4)
results = vm.search_entity(query, key="example", top_k=3)
for rank, result in enumerate(results, start=1):
    text = result.group.descriptions[result.vector_idx]["text"]
    print(f"{rank}. key={result.key} distance={result.distance:.4f} text={text}")
```

## Public API

- `VectorGroup`: grouped retrieval unit with representative and member vectors
- `VectorMap`: semantic-keyed storage, indexing, and search entry point
- `VectorRef` and `VectorRelation`: relation primitives for contextual expansion
- `create_context_chain()` and relation helpers for linked retrieval workflows

The full research repository, documentation, and benchmark results live at:
https://github.com/DoubleNorth/Violas
