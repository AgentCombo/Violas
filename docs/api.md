# Violas API Notes

This document summarizes the higher-level retrieval capabilities enabled by the
`VectorMap` / `VectorGroup` data model. The goal is not to present Violas
as a single `search(top_k)` interface, but as a structured retrieval framework
in which keys, groups, representative vectors, semantic key vectors, context,
relations, and graph routing are all part of the same retrieval system.

Throughout this document:

- `key` denotes a semantic scope such as a class label, document id, paper title, or session id
- `group` denotes a local view, micro-cluster, or sub-structure under a key
- `representative` denotes the group-level representative vector
- `key vector` denotes a semantic vector attached to the key itself

## Paper-Aligned API Layer

The paper describes Violas through create, update, delete, and query
capabilities. The Python package exposes concrete `VectorMap` methods for those
operations. The table below keeps the paper/evaluation terminology aligned with
implemented APIs.

| Paper/evaluation capability | Implemented Python API | Behavior |
| --- | --- | --- |
| Insert semantic group | `VectorMap.insert(key, group, metadata=None)` | Register a `VectorGroup` under a semantic entity key. |
| Insert object | `VectorMap.insert_object(key, vector, description)` | Insert one object and return a `VectorRef`. |
| Create group | `VectorMap.create_group(key, vectors, descriptions)` | Create a semantic group and register it. |
| Create micro-cluster | `VectorMap.create_cluster(key, group, alpha=0.5)` | Split a group into local representative regions. |
| Add relation | `VectorMap.add_relation(source, target, relation_type)` | Preserve object-level dependencies. |
| Update object | `VectorMap.update(ref, vector=None, description=None)` / `VectorMap.update_object(...)` | Update an embedding or metadata payload. |
| Assign object | `VectorMap.assign(ref, key, group_name=None)` / `VectorMap.assign_object(...)` | Move one object into another semantic group. |
| Delete object | `VectorMap.delete(ref)` / `VectorMap.delete_object(ref)` | Remove one object by reference. |
| Remove relation | `VectorMap.remove_relation(source, target=None, relation_type=None)` | Remove stored object-level dependencies. |
| Semantic-consistent retrieval | `VectorMap.search_entity(query_vector, key=...)` | Retrieve inside selected semantic entities. |
| Diversity-driven retrieval | `VectorMap.search_diverse(query_vector, query_key_vector=None, beta=0.5)` | Retrieve across diverse local regions. |
| Dependency-expanded retrieval | `VectorMap.search_dependency(query_vector, relation_types=..., hops=1)` | Expand through relations or local context. |
| Cross-modal pairing | `VectorMap.search_modal(query_vectors, modality_weights=...)` | Fuse image, text, or other modality vectors. |

## Implemented API Surface

The table below groups the concrete Python methods by system responsibility.
It is meant to show that the paper-level abstraction is backed by a concrete
implementation: Violas supports object lifecycle management, index
construction, relation maintenance, inspection, and multiple query modes.

### Create / Insert

| API | Purpose |
| --- | --- |
| `VectorMap.create_group(key, vectors, descriptions, ...)` | Create and register a semantic group from member objects. |
| `VectorMap.insert(key, group, metadata=None)` | Register an existing `VectorGroup` under a semantic key. |
| `VectorMap.insert_object(key, vector, description=None, ...)` | Insert one object and return a `VectorRef`. |
| `VectorMap.add_vector(vector, description)` | Append one object to an existing group. |
| `VectorMap.add_vector_list(vectors, descriptions)` | Batch append objects and create context ids. |

### Read / Access

| API | Purpose |
| --- | --- |
| `VectorMap.get(key, return_type="all")` | Read stored data, metadata, or groups for one key. |
| `VectorMap.get_all_keys()` | List stored semantic keys. |
| `VectorMap.get_group_by_name(key, group_name)` | Retrieve one group by name. |
| `VectorMap.get_group_by_id(key, group_id)` | Retrieve one group by id. |

### Update / Move

| API | Purpose |
| --- | --- |
| `VectorMap.update(ref, vector=None, description=None)` | Update one stored object's embedding or metadata. |
| `VectorMap.assign(ref, key, group_name=None)` | Move one object to another semantic group. |

### Delete

| API | Purpose |
| --- | --- |
| `VectorMap.delete(ref)` | Delete one stored object. |

### Index Management

| API | Purpose |
| --- | --- |
| `VectorMap.build_index()` | Build available local search indexes. |
| `VectorMap.build_rep_index()` | Build the representative-vector index. |
| `VectorMap.build_single_index()` | Build the member-vector index. |
| `VectorMap.set_key_vectors(key_vectors)` | Attach semantic vectors to keys. |
| `VectorMap.set_key_vectors_from_predictor(predictor)` | Load semantic key vectors from a predictor object. |
| `VectorMap.get_key_vector(key)` | Resolve a key vector, including clustered key prefixes. |
| `VectorMap.create_cluster(key, group, alpha=0.5)` | Create micro-clusters under a semantic entity. |
| `VectorMap.insert_with_auto_cluster(key, group, alpha=0.5)` | Insert a group after automatic micro-clustering. |
| `VectorMap.build_hdmg(embedding_k=16, semantic_intra_k=4, ...)` | Build the HDMG index over vector groups and micro-clusters. |
| `VectorMap.get_last_hdmg_search_stats()` | Inspect the previous HDMG query traversal. |

### Relations and Context

| API | Purpose |
| --- | --- |
| `VectorRef` | Address one object by key, group, and vector index. |
| `VectorRelation` | Store typed links between object references. |
| `VectorMap.add_relation(source, target, relation_type, ...)` | Add an object-level dependency relation. |
| `VectorMap.remove_relation(source, target=None, relation_type=None)` | Remove object-level relations. |
| `VectorMap.add_pair_relation(key, group1_id, group2_id, ...)` | Link paired groups such as image-caption groups. |
| `VectorMap.add_tree_relation(key, parent_name, child_name, ...)` | Store hierarchy relations under a key. |
| `VectorMap.get_relations(key)` | Read stored pair and tree relations for a key. |
| `VectorMap.get_hierarchy_tree(key)` | Build a hierarchy view from stored tree relations. |
| `VectorMap.get_contextual_vectors(result, num=2)` | Expand one result to neighboring context objects. |
| `VectorMap.search_with_contextual_vectors(query_vector, top_k=5, num=2)` | Search while considering local context windows. |
| `VectorMap.search_with_relations(query_vector, relation_types=...)` | Search while following explicit relations. |

### Query APIs

| API | Purpose |
| --- | --- |
| `VectorMap.search(query_vector, top_k=5, key=None, mode="single")` | Standard vector search with optional key, group, type, and mode filters. |
| `VectorMap.search_entity(query_vector, key=..., top_k=5)` | Semantic-consistent retrieval scoped to one or more keys. |
| `VectorMap.search_diverse(query_vector, query_key_vector=None, beta=0.5)` | Diversity-driven retrieval using HDMG or representative reranking. |
| `VectorMap.search_dependency(query_vector, relation_types=..., hops=1)` | Dependency-expanded retrieval through relations or local context. |
| `VectorMap.search_modal(query_vectors, modality_weights=None)` | Cross-modal retrieval wrapper. |
| `VectorMap.search_multimodal(query_vectors, modality_weights=None)` | Fuse multiple modality-specific searches. |
| `VectorMap.search_with_rep_vec(query_vector, top_k=5)` | Search group representatives. |
| `VectorMap.search_with_representative_rerank(query_vector, query_key_vector=None, beta=0.5)` | Representative retrieval plus semantic reranking. |
| `VectorMap.search_with_mixed_key_rep_vec(query_vector, query_key_vector, beta=0.5)` | Combine key-level semantic distance and representative-vector distance. |
| `VectorMap.search_hdmg(query_vector, query_key_vector=None, alpha=0.5, top_k=5)` | Run HDMG-backed semantic-vector retrieval. |

### Introspection

| API | Purpose |
| --- | --- |
| `VectorMap.get_statistics()` | Summarize keys, groups, vectors, indexes, and relations. |
| `VectorMap.analyze_relationships(key=None)` | Inspect relation coverage and relation types. |

## 1. Key-Restricted Retrieval

Many practical tasks do not require an unconstrained global nearest-neighbor
search. Instead, the system should first restrict the search scope to one or
more semantic keys, and only then retrieve the most relevant items.

### 1.1 Key-Restricted Search

Example interface:

```python
search(query_vector, top_k=10, mode="single", key="anchor")
search(query_vector, top_k=10, mode="representative", key=["anchor", "inline_skate"])
```

Expected behavior:

- search only within the specified key or keys
- support both single-key and multi-key restriction
- support key-prefix expansion when sub-cluster keys exist, e.g., `accordion` to `accordion-0001`, `accordion-0002`, and so on

Why this matters:

In Violas, keys are not an external metadata filter layered on top of a
vector store. They are first-class structural units inside `VectorMap`. As a
result, scoped retrieval is part of the storage model itself.

Typical scenarios:

- image retrieval within a known class
- paragraph retrieval within a known paper or document subset
- session-local conversational retrieval

## 2. Multi-View Retrieval Within One Key

In many cases, the goal is not simply to return the closest top-k results under
one key, but to cover multiple groups, viewpoints, or local sub-clusters under
that key.

### 2.1 Multi-View Search Within One Key

Example interface:

```python
search_diverse_within_key(
    query_vector,
    key="anchor",
    top_k=10,
    max_per_group=2,
    ensure_distinct_groups=True,
)
```

Expected behavior:

- prefer returning results from different `VectorGroup` instances
- limit the number of returned items per group
- prioritize group coverage when enough groups are available

Why this is natural in Violas:

Because groups are explicit structural objects, the system already knows which
group each result belongs to. In a conventional vector database, this usually
requires a separate application-layer reranking step after ANN search.

Typical scenarios:

- multiple viewpoints of the same object category
- different argumentative or descriptive angles within one topic
- multiple modalities or expressions of the same entity

## 3. Context-Aware Retrieval

For long documents, conversations, and segmented content, the answer is often
not an isolated chunk. The user may need the matched segment together with its
local context.

### 3.1 Context-Aware Retrieval

Example interface:

```python
search_with_context(
    query_vector,
    top_k=5,
    context_window=1,
    rerank=True,
    weight_decay=0.8,
)
```

Expected behavior:

- return the matched segment
- optionally return preceding and following neighbors
- optionally use neighboring context for reranking rather than simple expansion

Implementation basis already present in the codebase:

- `search_with_contextual_vectors(...)`
- `context_map`
- per-item metadata such as `doc_id`, `segment_idx`, and `total_segments`

Typical scenarios:

- medical or scientific paragraph retrieval
- conversational utterance retrieval
- temporal or sequential data retrieval

### 3.2 Triple-Context Retrieval

Example interface:

```python
search_with_triple_context(
    q_prev,
    q_query,
    q_next,
    top_k=5,
    context_num=1,
    agg="mean",
)
```

Expected behavior:

- enforce consistency with the preceding, current, and following query context
- support query chains rather than isolated single-turn retrieval

This is particularly useful for document QA, multi-turn reasoning, and dialogue
retrieval where a single segment cannot be interpreted independently.

## 4. Mixed Semantic-Vector Retrieval

This is the core retrieval family in Violas. Ranking is determined jointly
by instance-level vector distance and key-level semantic distance.

### 4.1 Mixed Key-Vector Retrieval

Scoring rule:

`mixed_score = beta * semantic_distance + (1 - beta) * embedding_distance`

Existing interfaces:

```python
search_with_mixed_key_rep_vec(
    query_vector,
    query_key_vector,
    beta=0.5,
    top_k=10,
)

search_with_representative_rerank(
    query_vector,
    query_key_vector,
    beta=0.5,
    top_k=10,
    num_groups=20,
)
```

Expected behavior:

- use `key + representative` mixed scoring at the group-selection stage
- use `key + instance vector` mixed scoring at the intra-group reranking stage
- keep the same scoring principle across both stages

Typical scenarios:

- image retrieval where visual similarity and semantic class cues should be combined
- document retrieval where paragraph embeddings are close but topic semantics differ
- multimodal retrieval with both content vectors and semantic condition vectors

## 5. Graph-Based Retrieval

Violas does not only support mixed scoring. It also supports graph-based
mixed routing over groups.

### 5.1 HDMG Search

Existing interfaces:

```python
build_hdmg(...)
search_hdmg(
    query_vector,
    query_key_vector,
    alpha=0.5,
    top_k=10,
    cluster_pool_size=64,
)
```

Key characteristics:

- nodes correspond to `(key, group)` micro-clusters
- embedding edges capture representative-vector proximity
- semantic edges capture same-key adjacency or cross-key semantic bridges

Why it matters:

- avoids exhaustive global reranking over all groups
- exposes graph-based semantic routing rather than pure embedding navigation
- naturally fits the structured retrieval design of Violas

## 6. Relation-Aware and Structural Expansion

Violas descriptions may include `VectorRelation` and `VectorRef`, allowing
retrieval results to be expanded along explicit structure rather than only by
distance.

### 6.1 Relation-Aware Search

Example interface:

```python
search_with_relations(
    query_vector,
    top_k=5,
    relation_types=["pair", "tree"],
    hop=1,
)
```

Implementation basis already present:

- `VectorRelation`
- `VectorRef`
- `get_relations_from_description(...)`
- relation-aware retrieval helpers

Typical scenarios:

- image-caption pair retrieval
- section-to-subsection expansion
- entity retrieval followed by linked-child or cross-modal expansion

### 6.2 Result-Centered Expansion

Example interface:

```python
expand_from_result(
    result,
    include_same_group=True,
    include_context=True,
    include_relations=True,
)
```

Expected behavior:

- expand to same-group neighbors
- expand to local contextual neighbors
- expand to explicitly related objects

This supports a result-centric workflow:

`query -> result -> structured expansion`

rather than repeatedly issuing unrelated top-k searches.

## 7. Two-Stage Semantic Retrieval

Another common pattern is to first retrieve relevant semantic keys, and only
then retrieve vectors within those keys.

### 7.1 Key-Then-Local Search

Example interface:

```python
search_two_stage(
    context_vector,
    query_vector,
    top_keys=5,
    top_k=10,
)
```

Expected behavior:

- stage 1 retrieves candidate keys using key vectors
- stage 2 searches only within the groups under those keys

Typical scenarios:

- retrieve relevant papers, then retrieve paragraphs within them
- retrieve relevant documents, then retrieve answer spans within them
- retrieve relevant sessions, then locate the most relevant utterances

## 8. Diversity and Coverage Constraints

In realistic applications, retrieval often requires coverage, not only raw
similarity. Results should not collapse into one group or one relation type.

### 8.1 Diverse Top-k Search

Example interface:

```python
search_diverse(
    query_vector,
    top_k=10,
    diversify_by="group",
    max_per_bucket=2,
)
```

Possible diversification axes:

- by group
- by key
- by relation type
- by contextual segment

This is especially suitable for Violas because these structural attributes
already exist inside the retrieval model.

### 8.2 Coverage-Constrained Retrieval

Example interface:

```python
search_with_coverage(
    query_vector,
    top_k=10,
    require_keys=["method", "result"],
)
```

Variants of `require_keys` can be generalized to:

- specific groups
- specific sections
- specific relation categories
- specific viewpoints

This is useful when retrieval is expected to return a structured answer set
rather than a flat similarity list.

## 9. High-Level API Summary

For implementation-oriented presentation, the above capabilities can be grouped
into five high-level families:

### 9.1 Scoped Retrieval

Retrieve only within selected keys or candidate key subsets.

### 9.2 Group-Aware Retrieval

Retrieve diverse representations or groups under the same semantic key.

### 9.3 Context-Aware Retrieval

Retrieve results jointly with surrounding local context or query-chain context.

### 9.4 Joint Semantic-Vector Retrieval

Use a unified mixed score across groups, instances, and graph routing.

### 9.5 Relation-Aware Structured Retrieval

Expand, rerank, or constrain retrieval results according to explicit structure,
relations, and coverage requirements.

Taken together, these APIs highlight the main design goal of Violas:

Violas is not only a nearest-neighbor search layer. It is a structured
retrieval framework that integrates semantic scope, grouped storage, graph
routing, context, and relations into one unified retrieval model.
