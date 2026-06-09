#!/usr/bin/env bash
# Phase 21: TrOCR-large-printed fine-tune + test eval → metrics.csv + JSONL.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

require_python

TROCR_ARGS=(
  --data-dir "${PROCESSED_DIR:-data/processed}"
  --epochs "${TROCR_EPOCHS:-10}"
  --batch-size "${TROCR_BATCH_SIZE:-8}"
  --output-dir "${TROCR_OUTPUT_DIR:-experiments/trocr_large_printed}"
)

if [[ "${TROCR_HOLD_OUT_TEST:-1}" == "1" ]]; then
  TROCR_ARGS+=(--hold-out-test)
fi
if [[ "${TROCR_RESUME:-1}" == "1" ]]; then
  TROCR_ARGS+=(--resume)
fi

run_py scripts/21_train_trocr.py "${TROCR_ARGS[@]}"

run_py scripts/22_evaluate_trocr.py \
  --model-dir "${TROCR_CKPT:-experiments/trocr_large_printed/best}" \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --split "${EVAL_SPLIT:-test}" \
  --results-csv "${METRICS_CSV:-results/tables/metrics.csv}"
