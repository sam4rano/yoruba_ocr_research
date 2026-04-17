#!/usr/bin/env bash
# Phase 16: LoRA fine-tune PaddleOCR-VL-1.5 on export from phase 14 (long-running GPU job).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  export PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi

EXTRA=()
if [[ -n "${VL15_LORA_MAX_SAMPLES:-}" ]]; then
  EXTRA+=(--max-samples "$VL15_LORA_MAX_SAMPLES")
fi

require_python
run_py scripts/16_train_paddleocr_vl_lora.py \
  --export-dir "${PADDLE_VL15_EXPORT_DIR:-data/paddleocr_vl15_sft}" \
  --output-dir "${PADDLE_VL15_LORA_DIR:-experiments/paddleocr_vl15_lora}" \
  --epochs "${VL15_LORA_EPOCHS:-1}" \
  "${EXTRA[@]}"
