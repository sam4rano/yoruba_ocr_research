#!/usr/bin/env bash
# Phase 20: Surya v2 zero-shot recognition on line crops.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

require_python
run_py scripts/20_baseline_surya_v2.py \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --split "${EVAL_SPLIT:-test}" \
  --results-csv "${METRICS_CSV:-results/tables/metrics.csv}"
