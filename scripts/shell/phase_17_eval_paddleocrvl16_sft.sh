#!/usr/bin/env bash
# Phase 17: Evaluate fine-tuned PaddleOCR-VL-1.6 on Yorùbá line crops.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  export PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi

if [[ "${SKIP_PADDLEOCRVL16_SFT_EVAL:-0}" == "1" ]]; then
  log "SKIP_PADDLEOCRVL16_SFT_EVAL=1; skip phase 17"
  exit 0
fi

EXTRA=()
if [[ -n "${PADDLEOCRVL16_EVAL_MAX_SAMPLES:-}" ]]; then
  EXTRA+=(--max-samples "$PADDLEOCRVL16_EVAL_MAX_SAMPLES")
fi
# PADDLEOCRVL16_EVAL_4BIT=0: use hardware-native dtype (bf16 on L4/A100, fp16 on T4).
# PADDLEOCRVL16_EVAL_4BIT=1: use 4-bit quantization only when VRAM is critically limited.
if [[ "${PADDLEOCRVL16_EVAL_4BIT:-0}" == "1" ]]; then
  EXTRA+=(--quantize-4bit)
fi

require_python
run_py scripts/eval_paddleocrvl16.py \
  --model-id "${PADDLEOCRVL16_SFT_DIR:-experiments/paddleocrvl16_sft}" \
  --model-name "paddleocrvl16_sft" \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --split "test" \
  --results-csv "${RESULTS_CSV:-results/tables/metrics.csv}" \
  --per-sample-log "${PROJECT_ROOT}/results/tables/paddleocrvl16_sft_test.jsonl" \
  "${EXTRA[@]}"
