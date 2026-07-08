#!/usr/bin/env bash
# Phase 4: PP-OCR recognition fine-tuning through PaddleOCR/tools/train.py.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ "${SKIP_PADDLE_TRAIN:-1}" == "1" ]]; then
  log "SKIP_PADDLE_TRAIN=1 — skipping phase 04"
  exit 0
fi

EXTRA=()
if [[ "${TRAIN_CPU:-0}" == "1" ]]; then
  EXTRA+=(--cpu)
else
  EXTRA+=(--gpus "${TRAIN_GPUS:-0}")
fi
if [[ "${TRAIN_RESUME:-0}" == "1" ]]; then
  EXTRA+=(--resume)
fi

require_python
run_py scripts/train_paddleocr_recognition.py \
  --config "${PADDLE_REC_CONFIG:-configs/paddleocr_yoruba_rec.yml}" \
  --paddle-dir "${PADDLE_DIR:-PaddleOCR}" \
  --log-file "${TRAIN_RUN_LOG:-results/tables/train_run.json}" \
  "${EXTRA[@]}"
