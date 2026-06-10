#!/usr/bin/env bash
# Phase 22: TrOCR-large-printed zero-shot (pretrained) test eval.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

require_python

run_py scripts/22_evaluate_trocr.py \
  --pretrained-model-id "${TROCR_PRETRAINED_ID:-microsoft/trocr-large-printed}" \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --split "${EVAL_SPLIT:-test}" \
  --results-csv "${METRICS_CSV:-results/tables/metrics.csv}"
