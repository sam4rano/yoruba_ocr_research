#!/usr/bin/env bash
# Phase 13: Compare metrics.csv ``n`` to current label files (dataset/eval alignment).
# Set VERIFY_STRICT=1 to fail the script on mismatch (e.g. before paper submission).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  export PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi

EXTRA=()
[[ "${VERIFY_STRICT:-0}" == "1" ]] && EXTRA+=(--strict)

require_python
run_py scripts/13_verify_eval_alignment.py \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --metrics-csv "${METRICS_CSV:-results/tables/metrics.csv}" \
  --output-json "${RESULTS_TABLES:-results/tables}/eval_alignment_report.json" \
  "${EXTRA[@]}"
