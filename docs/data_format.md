# Data Format

## Expected Dataset Layout

The benchmarks assume dataset-specific roots supplied either through command-line arguments or environment variables.

Image datasets:

- category-folder layout under a dataset root
- one subdirectory per class
- image files stored inside each class directory

Text datasets:

- topic-folder layout under a dataset root
- one subdirectory per topic
- one text file per document

COCO:

- a dataset root containing images
- a JSON annotation file such as `coco_dataset_40.json`
- each JSON item points to an image path relative to the dataset root

## Script-Level Environment Variables

Vision benchmarks:

- `CALTECH_ROOT`
- `CUB_ROOT`
- `COCO_ROOT`
- `COCO_JSON`

Text benchmarks:

- `NEWS20_ROOT`
- `OHSUMED_ROOT`
- `YAHOO_ROOT`

## Sample Summary Format

Benchmark summary JSON files typically contain:

- `betas`
- recall curves for multiple methods
- per-method latency statistics
- benchmark metadata such as `alpha` and search expansion settings

## Search Results

The storage layer returns `SearchResult` objects carrying:

- `distance`
- `key`
- `group`
- `vector_idx`

This makes it possible to support not only plain nearest-neighbor retrieval, but also group-aware expansion, contextual expansion, and relation-aware retrieval.
