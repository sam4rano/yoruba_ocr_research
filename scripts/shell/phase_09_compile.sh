#!/usr/bin/env bash
# Phase 9: compile metrics.csv into paper-ready Markdown/CSV tables.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

require_python
run_py scripts/11_compile_results.py \
  --results-csv "${METRICS_CSV:-results/tables/metrics.csv}" \
  --output-dir "${RESULTS_TABLES:-results/tables}"

# Warn if metrics.csv ``n`` does not match current label files (non-fatal; see JSON).
run_py scripts/13_verify_eval_alignment.py \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --metrics-csv "${METRICS_CSV:-results/tables/metrics.csv}" \
  --output-json "${RESULTS_TABLES:-results/tables}/eval_alignment_report.json"
