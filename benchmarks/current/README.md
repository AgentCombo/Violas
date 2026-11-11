# Benchmark commands

Run from the repository root. The drivers call the implementation in
`violas/retrieval/`; frozen plans and results are in `workloads/` and `results/`.

## Quick checks

```bash
pip install -e .
python -m benchmarks.current.run verify
python -m benchmarks.current.run test
```

`verify` checks file hashes and recorded aggregates; `test` checks retrieval,
graph construction and incremental storage on synthetic data.

## Full experiments

Use Linux/Python 3.12, the dependencies below, and a prepared workspace with
the original vector caches and Milvus/Chroma stores plus a Qdrant server.
Datasets and prepared databases are external to this repository.
The input paths and backend settings are recorded in each workload's `config`.

```bash
pip install -r benchmarks/current/requirements.txt
python -m benchmarks.current.run check-data --workspace /path/to/WORKSPACE
python -m benchmarks.current.run run --workspace /path/to/WORKSPACE --dataset caltech101 --experiment conditional-vg
```

Datasets: `caltech101`, `cub200`, `flowers102`, `20news`, `ohsumed`, `yahoo`.
Use `--experiment` to choose a runner:

| Experiment | Task |
| --- | --- |
| `conditional-vg` | Violas and the no-HDMG ablation |
| `hard-baselines` | VDB prefilter and filtered-index baselines |
| `postfilter-ann` | VDB ANN top-10 followed by filtering and selection |
| `yahoo-optimized` | Recorded Yahoo update; use `--dataset yahoo` |
| `maintenance` | Build/index costs; `--method vg_hdmg`, `milvus`, `qdrant` or `chroma` |
| `crud` | Single-object CRUD |
| `embedding` | Pure embedding retrieval; requires `g++` |
| `soft-entity` | Additional soft-entity comparison |

Use a copy of the prepared workspace: experiment runners write outputs and
maintenance runners create stores. For moved inputs, pass `--path-map` using
`path-map.example.json`; set `VIOLAS_QDRANT_URL` for a different server address.

`postfilter-ann` supplies the current ANN postfilter measurements; the old
exact-scan postfilter branch retained in `hard-baselines` is not used for them.
The additional `soft-entity` comparison uses relevance-only VDB reranking;
it is separate from the matched-selector conditional comparison.
Source mapping, settings and validation records remain alongside the scripts.

Data paths are relative to `--workspace`; `@repo/` resolves to this checkout.
Copy `path-map.example.json` to the ignored `path-map.local.json` for local
remapping. Published path metadata is sanitized; measured values are unchanged.
