#!/usr/bin/env bash
# Phase 16: Fine-tune PaddleOCR-VL-1.6 language model on export from phase 14 (long-running GPU job).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  export PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi

if [[ "${SKIP_PADDLEOCRVL16_SFT_TRAIN:-0}" == "1" ]]; then
  log "SKIP_PADDLEOCRVL16_SFT_TRAIN=1 — skipping phase 16"
  exit 0
fi

EXTRA=()
if [[ -n "${PADDLEOCRVL16_SFT_MAX_SAMPLES:-}" ]]; then
  EXTRA+=(--max-samples "$PADDLEOCRVL16_SFT_MAX_SAMPLES")
fi
if [[ -n "${PADDLEOCRVL16_SFT_LR:-}" ]]; then
  EXTRA+=(--lr "$PADDLEOCRVL16_SFT_LR")
fi
if [[ -n "${PADDLEOCRVL16_SFT_GRAD_ACCUM:-}" ]]; then
  EXTRA+=(--gradient-accumulation-steps "$PADDLEOCRVL16_SFT_GRAD_ACCUM")
fi
if [[ -n "${PADDLEOCRVL16_SFT_MAX_PIXELS:-}" ]]; then
  EXTRA+=(--max-pixels "$PADDLEOCRVL16_SFT_MAX_PIXELS")
fi
if [[ -n "${PADDLEOCRVL16_SFT_EMPTY_CACHE_STEPS:-}" ]]; then
  EXTRA+=(--empty-cache-steps "$PADDLEOCRVL16_SFT_EMPTY_CACHE_STEPS")
fi
if [[ "${PADDLEOCRVL16_SFT_RESUME:-0}" == "1" ]]; then
  EXTRA+=(--resume)
fi

require_python
run_py scripts/train_paddleocrvl16_sft.py \
  --export-dir "${PADDLEOCRVL16_SFT_EXPORT_DIR:-data/paddleocrvl16_sft}" \
  --output-dir "${PADDLEOCRVL16_SFT_DIR:-experiments/paddleocrvl16_sft}" \
  --epochs "${PADDLEOCRVL16_SFT_EPOCHS:-5}" \
  "${EXTRA[@]}"
