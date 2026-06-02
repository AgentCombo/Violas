# Violas

**In-Memory Vector Group System for Semantic Search.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9--3.11-blue.svg)](https://www.python.org/)
[![PyPI version](https://badge.fury.io/py/violas.svg)](https://badge.fury.io/py/violas)

Violas is an in-memory retrieval framework for applications where a result is
more than one nearest embedding. It organizes vectors as semantic entities with
member observations, representative local regions, relevance links, and paired
modal evidence, then
uses HDMG to search those structured objects efficiently.

## Outline

- [Why Violas?](#why-violas)
- [Quick Start](#quick-start)
- [Typical Cases](#typical-cases)
- [How It Works](#how-it-works)
- [API Overview](#api-overview)
- [Installation Options](#installation-options)
- [Experiments](#experiments)
- [Reproducing Benchmarks](#reproducing-benchmarks)

## Why Violas?

Most vector search systems retrieve isolated embeddings. That works when
nearest-neighbor distance is the whole task, but many applications need richer
retrieval behavior:

- **Semantic-consistent retrieval**: retrieve results that stay consistent with the
  intended semantic entity. See [Entity Mismatch](#entity-mismatch).
- **Diversity-driven retrieval**: cover multiple poses, chunks, or local modes
  within the same entity. See [Diversity Loss](#diversity-loss).
- **Dependency-expanded retrieval**: expand a hit to linked context or temporal
  neighbors. See [Dependency Loss](#dependency-loss).
- **Cross-modal pairing**: keep text-image pairs inside the same
  retrieval object. See [Cross-Modal Evidence](#cross-modal-evidence).

Violas provides this behavior through `VectorGroup`, a semantic retrieval object
that keeps the entity, its members, local representatives, and relations in the
same searchable structure.

## Quick Start

Install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you want the packaged core library from PyPI later, the distribution name is
`violas`.

Run the minimal example. It uses synthetic vectors and does not require any
dataset, embedding model, or external vector database:

```bash
python examples/minimal_vectormap.py
```

Minimal API usage:

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
    vector_type="demo",
    group_type="synthetic",
)

query = np.random.rand(4)
results = vm.search_entity(query, key="example", top_k=3)
for rank, result in enumerate(results, start=1):
    text = result.group.descriptions[result.vector_idx]["text"]
    print(f"{rank}. key={result.key} distance={result.distance:.4f} text={text}")
```

## Typical Cases

The examples below correspond to the native capabilities evaluated in the
paper: semantic-consistent retrieval, diversity-driven retrieval,
dependency-expanded retrieval, and cross-modal pairing. We show them through
common flat-retrieval failure modes: entity mismatch, diversity loss,
dependency loss, and broken multimodal pairing. Entity mismatch is split into
two visible subcases: plausible mismatch, where the neighbor looks reasonable
but belongs to the wrong entity, and unjustified mismatch, where the neighbor
is close in embedding space without a clear semantic relation.

### Entity Mismatch

Flat nearest-neighbor search can drift into a different semantic entity. Violas
routes by semantic entity first, then ranks local members.

#### Plausible Entity Mismatch

Some wrong neighbors are visually or textually plausible because they share
coarse traits with the query, but they still violate the intended entity.

<table>
  <tr>
    <td width="50%" align="center" valign="top">
      <img src="docs/figures/readme/case-1-1.jpg" width="390" alt="Rhinoceros query retrieves visually similar but semantically wrong large animals"><br>
      <sub>Rhinoceros: large-animal shape is not enough</sub>
    </td>
    <td width="50%" align="center" valign="top">
      <img src="docs/figures/readme/case-1-2.jpg" width="390" alt="Anchor query retrieves symbols and tools that share low-level shape features"><br>
      <sub>Anchor: abstract shape should not override entity meaning</sub>
    </td>
  </tr>
  <tr>
     <td width="50%" align="center" valign="top">
      <img src="docs/figures/readme/case-1-3.jpg" width="390" alt="Platypus query retrieves visually similar aquatic animals instead of the intended entity"><br>
      <sub>Platypus: coarse appearance is not enough</sub>
    </td>
    <td width="50%" align="center" valign="top">
      <img src="docs/figures/readme/case-1-4.jpg" width="390" alt="Stegosaurus query retrieves creatures with similar texture or shape but different semantics"><br>
      <sub>Stegosaurus: semantic identity should dominate</sub>
    </td>
  </tr>
</table>


#### Unjustified Entity Mismatch

Some nearest neighbors are embedding-close without a clear category-level or
visual relation to the query. These cases expose a stronger form of entity
mismatch than visually plausible drift.

<table>
  <tr>
    <td width="50%" align="center" valign="top">
      <img src="docs/figures/readme/case-2-1.jpg" width="390" alt="Ant query retrieves unrelated visual concepts under flat vector search"><br>
      <sub>Ant: entity identity is not preserved</sub>
    </td>
    <td width="50%" align="center" valign="top">
      <img src="docs/figures/readme/case-2-3.jpg" width="390" alt="Wristwatch query retrieves unrelated small objects with low-level visual overlap"><br>
      <sub>Wristwatch: embedding proximity is not enough</sub>
    </td>
  </tr>
</table>



### Diversity Loss

Even when the entity is correct, top results can be too redundant. Violas uses
representative local regions to expose different poses, viewpoints, chunks, or
scene layouts under the same semantic entity.

<table>
  <tr>
    <td width="50%" align="center" valign="top">
      <img src="docs/figures/readme/case-3-1.jpg" width="390" alt=""><br>
      <sub>Bird: one entity, multiple useful views</sub>
    </td>
    <td width="50%" align="center" valign="top">
      <img src="docs/figures/readme/case-3-2.jpg" width="390" alt="Airplane coverage case: retrieval should cover different flight configurations and scene compositions under the same entity"><br>
      <sub>Airplane: cover flight configurations and scenes</sub>
    </td>
  </tr>
</table>



### Dependency Loss

Some useful answers require linked objects rather than a single nearest chunk.
In these OHSUMED cases, the middle segment is used as the query, while the
desired answer is the surrounding same-document evidence chain. Flat top-k
retrieval often returns isolated snippets that share surface vocabulary but
lose the dependency between setup, evidence, and conclusion.

<table>
  <tr>
    <td width="50%" align="center" valign="top">
      <img src="docs/figures/readme/case-5-1.jpg" width="390" alt="Norfloxacin query needs linked evidence rather than isolated clinical-trial chunks"><br>
      <sub>Norfloxacin: trial evidence should be retrieved as a chain</sub>
    </td>
    <td width="50%" align="center" valign="top">
      <img src="docs/figures/readme/case-5-2.jpg" width="390" alt="Candidiasis treatment query needs individualized therapy context rather than isolated symptom or drug fragments"><br>
      <sub>Candidiasis: treatment choices need patient context</sub>
    </td>
  </tr>
</table>



### Cross-Modal Evidence

Some retrieval tasks need paired evidence across modalities, such as a caption
and its corresponding image. Violas keeps these modality links inside the same
retrieval object instead of reconstructing them after a flat vector search.

<table>
  <tr>
    <td width="50%" align="center" valign="top">
      <img src="docs/figures/readme/case-4-1.jpg" width="390" alt="COCO multimodal query about a woman cutting a large white sheet cake"><br>
      <sub>Sheet cake: text query aligned with image evidence</sub>
    </td>
    <td width="50%" align="center" valign="top">
      <img src="docs/figures/readme/case-4-2.jpg" width="390" alt="COCO multimodal query about a motorbike on a dirt road in the countryside"><br>
      <sub>Motorbike: scene-level text and image agreement</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="top">
      <img src="docs/figures/readme/case-4-3.jpg" width="390" alt="COCO multimodal query about a girl holding a cat and wearing a colorful skirt"><br>
      <sub>Girl with cat: paired visual and caption evidence</sub>
    </td>
    <td width="50%" align="center" valign="top">
      <img src="docs/figures/readme/case-4-4.jpg" width="390" alt="COCO multimodal query about a girl preparing to blow out a candle"><br>
      <sub>Candle: multimodal evidence keeps context intact</sub>
    </td>
  </tr>
</table>


## How It Works

Most vector databases follow a flat vector-search paradigm: store each data
item as an embedding and retrieve approximate nearest neighbors in that
embedding space. This is effective when embedding proximity is enough, but it
leaves richer semantic information outside the retrieval model. In the cases
above, the missing information is exactly what the query needs: entity
consistency, coverage of diverse local forms, and linked evidence.

Violas turns these requirements into part of the indexed retrieval object. The
system first organizes objects by semantic entity, then exposes diversity inside
the entity through micro-clusters, and finally preserves object-level relations
for context, dependency, and multimodal expansion.

### Retrieval Paradigm

<p align="center">
  <img src="docs/figures/readme/paradigm.png" width="560" alt="Violas retrieval paradigm">
</p>

The retrieval pipeline follows this structured design:

1. route the query to candidate semantic groups
2. select micro-clusters or members with a mixed semantic-embedding score
3. expand through dependencies or paired modalities when the query requires
   linked evidence

This differs from post-processing a flat top-k result. Violas stores the
semantic key, internal diversity, and dependency structure before retrieval, so
the system does not need to reconstruct them after the nearest-neighbor search.

### Vector Group

<p align="center">
  <img src="docs/figures/readme/vectorgroup.png" width="720" alt="Vector Group structure">
</p>

`VectorGroup` is the semantic-first storage abstraction behind Violas. It is a
three-level structure:

| Layer | Role |
| --- | --- |
| Group header | Stores the semantic key and group-level semantic vector. |
| Micro-clusters | Represent diverse local forms inside the same entity. |
| Members | Store concrete objects, embeddings, metadata, and relations. |

This lets one entity, such as a class, document, event, or multimodal item, be
managed as one retrieval object instead of a loose collection of independent
embeddings. The same structure supports entity-scoped search, diversity-aware
composition, dependency expansion, and cross-modal pairing.

### HDMG Indexing

<p align="center">
  <img src="docs/figures/readme/HDMG.png" width="560" alt="HDMG structure">
</p>


On top of `VectorGroup`, Violas builds **HDMG** (Hierarchical Diversified Micro
Cluster Graph), an indexing mechanism for efficient semantic-first retrieval.
HDMG indexes micro-clusters rather than all raw vectors. It keeps search off the
full flat object space by:

- routing into semantically compatible groups first
- traversing representative micro-cluster nodes instead of scanning all members
- using heterogeneous traversal edges for embedding proximity and semantic reachability
- preserving relevance expansion inside the same retrieval substrate

## API Overview

Violas exposes the high-level retrieval capabilities described in the paper,
and backs them with concrete system APIs for object lifecycle management, index
construction, relation maintenance, query execution, and inspection. The goal is
to make the research abstraction usable as an implemented retrieval system, not
only as paper pseudocode.

### Paper-Aligned Capabilities

| Paper capability | Primary APIs | Typical output |
| --- | --- | --- |
| Semantic-consistent retrieval | `search_entity(...)`, `search(..., key=...)` | Results scoped to the intended entity. |
| Diversity-driven retrieval | `create_cluster(...)`, `search_diverse(...)`, `search_with_representative_rerank(...)` | Results composed across local forms of one entity. |
| Dependency-expanded retrieval | `add_relation(...)`, `search_dependency(...)`, `search_with_contextual_vectors(...)` | A seed hit plus linked context or evidence. |
| Cross-modal pairing | `add_pair_relation(...)`, `search_modal(...)`, `search_multimodal(...)` | Paired image-text or multi-view evidence. |

### Implemented System Surface

| Area | APIs | What it covers |
| --- | --- | --- |
| Create / insert | `create_group(...)`, `insert(...)`, `insert_object(...)`, `add_vector(...)`, `add_vector_list(...)` | Create semantic groups and insert individual or batched objects. |
| Read / access | `get(...)`, `get_all_keys(...)`, `get_group_by_name(...)`, `get_group_by_id(...)` | Access stored keys, metadata, groups, and group contents. |
| Update / move | `update(...)`, `update_object(...)`, `assign(...)`, `assign_object(...)` | Update object embeddings or metadata and move objects across groups. |
| Delete | `delete(...)`, `delete_object(...)` | Remove stored objects by reference. |
| Index construction | `build_index(...)`, `build_rep_index(...)`, `build_single_index(...)`, `set_key_vectors(...)`, `build_hdmg(...)`, `get_last_hdmg_search_stats(...)` | Build member indexes, representative indexes, semantic key state, and HDMG. |
| Relation management | `VectorRef`, `VectorRelation`, `add_relation(...)`, `remove_relation(...)`, `add_pair_relation(...)`, `add_tree_relation(...)`, `get_relations(...)` | Maintain context, temporal, hierarchy, dependency, and multimodal links. |
| Query execution | `search(...)`, `search_entity(...)`, `search_diverse(...)`, `search_dependency(...)`, `search_modal(...)`, `search_hdmg(...)` | Run standard, scoped, diverse, relation-aware, multimodal, and HDMG-backed retrieval. |
| Inspection | `get_all_keys(...)`, `get_group_by_name(...)`, `get_statistics(...)`, `analyze_relationships(...)` | Inspect stored keys, groups, index state, and relation coverage. |

Example: object lifecycle and relation setup:

```python
ref = vm.insert_object("paper-001", query_vec, {"text": "middle segment"})
ctx = vm.insert_object("paper-001", context_vec, {"text": "previous segment"})
vm.add_relation(ref, ctx, relation_type="context")
vm.update(ref, description={"section": "clinical evidence"})
```

Example: build indexes and run structured retrieval:

```python
vm.set_key_vectors(key_vectors)
vm.build_index()
vm.build_hdmg()

results = vm.search_hdmg(query_vector, query_key_vector=semantic_embedding, top_k=5)
context = vm.search_dependency(query_vector, relation_types=["context"], top_k=5)
```

For the fuller method list and retrieval patterns, see [docs/api.md](docs/api.md).

## Installation Options

The full benchmark suite uses several optional backends. For quick library
experiments, install only the minimal dependencies shown above.

For the full benchmark environment:

```bash
pip install -r requirements.txt
pip install -e .
```

Alternatively, use the full benchmark requirements file:

```bash
pip install -r requirements.txt
```

The full benchmark dependencies include optional-heavy packages such as CLIP,
Sentence-Transformers, FAISS, Milvus Lite, Qdrant, and Chroma because the
benchmark suite compares Violas with external vector database baselines.

## Repository Layout

```text
Violas/
  violas/
    storage/       # VectorMap, VectorGroup, relation helpers
    core/          # feature helpers, recall utilities, baseline indexes
  benchmarks/      # six benchmark pipelines plus diversity cases
  examples/        # small runnable examples
  scripts/         # benchmark wrapper scripts
  docs/            # API notes, data formats, benchmark results, case studies
```

## Experiments

![Violas Benchmark Overview](docs/figures/benchmarks/overview_grid.png)

Violas is evaluated on six vision and text benchmarks under the mixed
semantic-vector retrieval objective described in the paper. At the balanced
setting (`beta = 0.5`), the system delivers high retrieval quality with low
latency across all datasets.

| Dataset | HDMG Recall@3 | HDMG Latency (ms) | Representative Recall@3 | Representative Latency (ms) | Milvus Recall@3 | Milvus Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Caltech-101 | 0.9985 | 3.06 | 1.0000 | 131.53 | 0.8972 | 15.94 |
| CUB-200-2011 | 1.0000 | 2.91 | 1.0000 | 163.91 | 0.5253 | 15.69 |
| COCO | 0.9965 | 1.49 | 1.0000 | 115.75 | 0.6625 | 4.97 |
| 20 Newsgroups | 0.9987 | 2.50 | 0.9715 | 81.89 | 0.7832 | 5.16 |
| OHSUMED | 0.9809 | 2.71 | 0.9226 | 62.44 | 0.4586 | 4.57 |
| Yahoo Answers | 1.0000 | 2.94 | 0.9507 | 58.56 | 0.6132 | 5.23 |

What this shows:

- HDMG reaches near-perfect retrieval quality on every benchmark at `beta = 0.5`.
- Representative reranking remains accurate but is substantially slower than HDMG.
- Pure-vector baselines are competitive at `beta = 0.0`, but degrade once
  semantic control matters.

The benchmark environment uses an Intel Xeon Platinum 8350C CPU (2.60 GHz),
16 CPU cores, and 64 GB RAM. Image embeddings use CLIP ViT-B/32 and text
embeddings use Sentence-Transformers all-MiniLM-L6-v2 in the benchmark
pipelines.

More details:

- [Benchmark Results](docs/results.md)
- [Benchmark Notes](docs/benchmark.md)
- [Data Format](docs/data_format.md)

## Reproducing Benchmarks

The shell scripts assume a workspace where `Violas/` and `dataset/` are
siblings. Override the dataset variables if your data lives elsewhere.

```bash
bash scripts/run_bench_all.sh
bash scripts/run_bench_vision.sh
bash scripts/run_bench_text.sh
bash scripts/run_diversity_case.sh
```

Vision dataset variables:

- `CALTECH_ROOT`
- `CUB_ROOT`
- `COCO_ROOT`
- `COCO_JSON`

Text dataset variables:

- `NEWS20_ROOT`
- `OHSUMED_ROOT`
- `YAHOO_ROOT`

Saved artifacts are written under `outputs/<benchmark>/` by default. External
vector database baselines are disabled by default for portability. To enable
them:

```bash
export VIOLAS_ENABLE_EXTERNAL_DBS=1
```
