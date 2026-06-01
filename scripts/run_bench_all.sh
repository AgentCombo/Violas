#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
bash run_bench_vision.sh
bash run_bench_text.sh
bash run_diversity_case.sh
