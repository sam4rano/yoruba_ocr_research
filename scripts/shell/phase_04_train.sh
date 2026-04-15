#!/usr/bin/env bash
# Phase 4: fine-tune PaddleOCR (GPU: --gpus 0; CPU/mac: --cpu on 04_train_paddleocr.py).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

require_python
TRAIN_ARGS=(
  scripts/04_train_paddleocr.py
  --config "${TRAIN_CONFIG:-configs/paddleocr_yoruba_rec.yml}"
  --paddle-dir "${PADDLE_DIR:-PaddleOCR}"
  --gpus "${TRAIN_GPUS:-0}"
  --log-file "${TRAIN_LOG:-results/tables/train_run.json}"
)
[[ -n "${TRAIN_EPOCHS_OVERRIDE:-}" ]] && TRAIN_ARGS+=( --epochs "${TRAIN_EPOCHS_OVERRIDE}" )
[[ -n "${TRAIN_BATCH_OVERRIDE:-}" ]] && TRAIN_ARGS+=( --batch-size "${TRAIN_BATCH_OVERRIDE}" )
[[ -n "${TRAIN_LR_OVERRIDE:-}" ]] && TRAIN_ARGS+=( --lr "${TRAIN_LR_OVERRIDE}" )
[[ "${TRAIN_CPU:-0}" == "1" ]] && TRAIN_ARGS+=( --cpu )

run_py "${TRAIN_ARGS[@]}"
