# Benchmark Notes

Violas includes six standard benchmark pipelines and one diversity-oriented
structured retrieval benchmark.

## Included Benchmarks

Vision:

- Caltech-101
- CUB-200-2011
- COCO

Text:

- 20 Newsgroups
- OHSUMED
- Yahoo Answers

Structured retrieval:

- Diversity benchmark in `benchmarks/diversity/diversity_case_bench.py`

See [results.md](results.md) for benchmark figures and compact result tables.

## Shared Evaluation Pattern

Each benchmark follows the same high-level flow:

1. load dataset-specific inputs
2. build image or text vectors
3. construct `VectorMap` and `VectorGroup` structures
4. build representative search and HDMG routing structures
5. evaluate mixed retrieval, representative reranking, and Python flat search
6. optionally evaluate external vector database baselines
7. save detailed outputs and summary artifacts under `outputs/`

## Entry Points

Main benchmark scripts:

- `benchmarks/caltech/caltech_bench.py`
- `benchmarks/cub/cub_bench.py`
- `benchmarks/coco/coco_bench.py`
- `benchmarks/news20/20news_bench.py`
- `benchmarks/ohsumed/ohsumed_bench.py`
- `benchmarks/yahoo/yahoo_answer_bench.py`

Convenience wrappers:

- `scripts/run_bench_all.sh`
- `scripts/run_bench_vision.sh`
- `scripts/run_bench_text.sh`
- `scripts/run_diversity_case.sh`

## Output Convention

Saved benchmark artifacts are written under `outputs/<benchmark-name>/` by
default. Typical artifacts include:

- detailed per-query JSON
- aggregated summary JSON
- optional case-study outputs

External vector database baselines are disabled by default for portability. Set
`VIOLAS_ENABLE_EXTERNAL_DBS=1` to enable them.

## Result Interpretation

The standard benchmark pages focus on three metrics:

- `Recall@3`
- `NDCG@3`
- average latency per query

The beta sweep controls the relative weight between semantic-key distance and
instance-vector distance. In practice:

- `beta = 0.0` emphasizes embedding similarity
- `beta = 0.5` is a balanced mixed retrieval setting
- `beta = 1.0` emphasizes semantic control
