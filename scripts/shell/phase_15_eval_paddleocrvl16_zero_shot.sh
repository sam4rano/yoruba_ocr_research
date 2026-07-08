#!/usr/bin/env bash
# Phase 15: PaddleOCR-VL-1.6 zero-shot eval.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ "${SKIP_PADDLEOCRVL16_ZERO_SHOT:-0}" == "1" ]]; then
  log "WARN: SKIP_PADDLEOCRVL16_ZERO_SHOT=1 — skipping PaddleOCR-VL-1.6 eval."
  exit 0
fi

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  export PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi

export HF_TRUST_REMOTE_CODE="${HF_TRUST_REMOTE_CODE:-1}"

PADDLEOCRVL16_ARGS=(
  scripts/eval_paddleocrvl16.py
  --data-dir "${PROCESSED_DIR:-data/processed}"
  --split "${EVAL_SPLIT:-test}"
  --results-csv "${METRICS_CSV:-results/tables/metrics.csv}"
)
[[ -n "${PADDLEOCRVL16_MAX_SAMPLES:-}" ]] && PADDLEOCRVL16_ARGS+=( --max-samples "$PADDLEOCRVL16_MAX_SAMPLES" )
# PADDLEOCRVL16_QUANTIZE_4BIT=0: use hardware-native dtype (bf16 on L4/A100, fp16 on T4).
# PADDLEOCRVL16_QUANTIZE_4BIT=1: use 4-bit quantization (useful for small VRAM, but record it).
[[ "${PADDLEOCRVL16_QUANTIZE_4BIT:-0}" == "1" ]] && PADDLEOCRVL16_ARGS+=( --quantize-4bit )

require_python
run_py "${PADDLEOCRVL16_ARGS[@]}"
