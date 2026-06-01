#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
PYTHON_BIN="${PYTHON_BIN:-../venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python
fi

"$PYTHON_BIN" benchmarks/diversity/diversity_case_bench.py
