#!/usr/bin/env bash
# Phase 17: Evaluate fine-tuned PaddleOCR-VL-1.6 on Yorùbá line crops.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  export PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi

if [[ "${SKIP_VL16_FINETUNED_EVAL:-0}" == "1" ]]; then
  log "SKIP_VL16_FINETUNED_EVAL=1; skip phase 17"
  exit 0
fi

EXTRA=()
if [[ -n "${VL16_EVAL_MAX_SAMPLES:-}" ]]; then
  EXTRA+=(--max-samples "$VL16_EVAL_MAX_SAMPLES")
fi
if [[ "${VL16_EVAL_4BIT:-0}" == "1" ]]; then
  EXTRA+=(--quantize-4bit)
fi

require_python
run_py scripts/15_baseline_paddleocr_vl16.py \
  --model-id "${PADDLE_VL16_FINETUNED_DIR:-experiments/paddleocr_vl16_finetuned}" \
  --model-name "paddleocr_vl16_finetuned" \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --split "test" \
  --results-csv "${RESULTS_CSV:-results/tables/metrics.csv}" \
  --per-sample-log "${PROJECT_ROOT}/results/tables/paddleocr_vl16_finetuned_test.jsonl" \
  "${EXTRA[@]}"
