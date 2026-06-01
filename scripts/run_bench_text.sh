#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
PYTHON_BIN="${PYTHON_BIN:-../venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python
fi

: "${NEWS20_ROOT:=$(cd .. && pwd)/dataset/20_newsgroups}"
: "${OHSUMED_ROOT:=$(cd .. && pwd)/dataset/ohsumed-all-docs/ohsumed-all}"
: "${YAHOO_ROOT:=$(cd .. && pwd)/dataset/Yahoo!-answer_processed}"

"$PYTHON_BIN" benchmarks/news20/20news_bench.py --root "$NEWS20_ROOT"
"$PYTHON_BIN" benchmarks/ohsumed/ohsumed_bench.py --root "$OHSUMED_ROOT"
"$PYTHON_BIN" benchmarks/yahoo/yahoo_answer_bench.py --root "$YAHOO_ROOT"
