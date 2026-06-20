#!/usr/bin/env bash
# Phase 16: Fine-tune PaddleOCR-VL-1.6 language model on export from phase 14 (long-running GPU job).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  export PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi

EXTRA=()
if [[ -n "${VL16_SFT_MAX_SAMPLES:-}" ]]; then
  EXTRA+=(--max-samples "$VL16_SFT_MAX_SAMPLES")
fi
if [[ -n "${VL16_SFT_LR:-}" ]]; then
  EXTRA+=(--lr "$VL16_SFT_LR")
fi
if [[ -n "${VL16_GRAD_ACCUM:-}" ]]; then
  EXTRA+=(--gradient-accumulation-steps "$VL16_GRAD_ACCUM")
fi
if [[ "${VL16_TRAIN_RESUME:-0}" == "1" ]]; then
  EXTRA+=(--resume)
fi

require_python
run_py scripts/16_train_paddleocr_vl.py \
  --export-dir "${PADDLE_VL16_EXPORT_DIR:-data/paddleocr_vl16_sft}" \
  --output-dir "${PADDLE_VL16_FINETUNED_DIR:-experiments/paddleocr_vl16_finetuned}" \
  --epochs "${VL16_SFT_EPOCHS:-5}" \
  "${EXTRA[@]}"
