#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
PYTHON_BIN="${PYTHON_BIN:-../venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python
fi

: "${CALTECH_ROOT:=$(cd .. && pwd)/dataset/caltech-101/101_ObjectCategories}"
: "${CUB_ROOT:=$(cd .. && pwd)/dataset/CUB_200_2011/images}"
: "${COCO_ROOT:=$(cd .. && pwd)/dataset/coco}"
: "${COCO_JSON:=coco_dataset_40.json}"

"$PYTHON_BIN" benchmarks/caltech/caltech_bench.py --root "$CALTECH_ROOT"
"$PYTHON_BIN" benchmarks/cub/cub_bench.py --root "$CUB_ROOT"
"$PYTHON_BIN" benchmarks/coco/coco_bench.py --root "$COCO_ROOT" --json_file "$COCO_JSON"
