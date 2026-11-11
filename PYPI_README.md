# Violas

Violas is a lightweight Python package for grouped vector retrieval.

It exposes the core storage and retrieval primitives from the Violas
project without bundling the benchmark pipelines, datasets, or experiment
artifacts. The package is meant to be a small installable surface for:

- semantic-keyed vector groups
- representative-based retrieval
- relation-aware expansion
- HDMG-backed mixed semantic-vector search

The development source includes `IntentIndex` and the measured intent/facet
engine. The existing published 0.0.1 wheel is not updated by this repository
change; clone **AgentCombo/Violas**, branch **main**, to use this version.

## Install from this checkout

```bash
pip install -e .
```

Optional FAISS-backed acceleration:

```bash
pip install -e ".[faiss]"
```

## Quick Start

```python
import numpy as np

from violas import VectorGroup, VectorMap

vectors = [np.random.rand(4) for _ in range(5)]
group = VectorGroup(
    group_name="demo",
    representative=np.mean(vectors, axis=0),
    rep_description="demo representative",
    vectors=vectors,
    descriptions=[{"text": f"item {i}"} for i in range(5)],
)

vm = VectorMap()
vm.insert("example", group)

query = np.random.rand(4)
results = vm.search(query, top_k=3)
for result in results:
    print(result)
```

## Public API

- `VectorGroup`: grouped retrieval unit with representative and member vectors
- `VectorMap`: semantic-keyed storage, indexing, and search entry point
- `VectorRef` and `VectorRelation`: relation primitives for contextual expansion
- `create_context_chain()` and relation helpers for linked retrieval workflows

The full research repository, documentation, and benchmark results live at:
https://github.com/AgentCombo/Violas

## Intent-aware retrieval

Run `python -m examples.intent_retrieval` from the repository root.
`IntentIndex.from_vectormap(vm)` builds the graph/directory over existing groups;
`index.search(query_embedding, resolved_plan)` returns selected IDs, scores and
execution statistics. See `docs/api.md` for plan fields and snapshot semantics.
