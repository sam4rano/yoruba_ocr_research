#!/usr/bin/env bash
# Phase 5: PaddleOCR baselines + fine-tuned checkpoint evaluation → metrics.csv + JSONL.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

require_python
BASE_PRE="${BASELINE_PRETRAINED_DIR:-experiments/baseline/pretrained/en_PP-OCRv3_rec_train}"
FINETUNE_DIR="${FINETUNED_DIR:-experiments/finetuned}"
SPLIT="${EVAL_SPLIT:-test}"

GPU=( )
if [[ "${EVAL_USE_GPU:-0}" == "1" ]]; then
  GPU=( --use-gpu )
fi

REC_CONFIG="${PADDLE_REC_CONFIG:-configs/paddleocr_yoruba_rec.yml}"

run_py scripts/05_evaluate.py \
  --model-dir "$BASE_PRE" \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --split "$SPLIT" \
  --model-name baseline_english_pretrained \
  --rec-config "$REC_CONFIG" \
  "${GPU[@]}" \
  --allow-head-reinit \
  --paddle-dir "${PADDLE_DIR:-PaddleOCR}"

# PP-OCRv4 fine-tuned checkpoint is OPTIONAL (classical CRNN comparison).
# Skip automatically when:
#   - SKIP_PADDLE_FINETUNE=1 is set explicitly (recommended if phase 04 was not run), or
#   - the checkpoint directory does not exist.
if [[ "${SKIP_PADDLE_FINETUNE:-0}" == "1" ]]; then
  log "SKIP_PADDLE_FINETUNE=1 — skipping fine-tuned PP-OCRv4 evaluation"
  exit 0
fi
if [[ ! -d "$FINETUNE_DIR" ]]; then
  log "No fine-tuned checkpoint at $FINETUNE_DIR — skipping (run phase 04 first or set SKIP_PADDLE_FINETUNE=1)"
  exit 0
fi

run_py scripts/05_evaluate.py \
  --model-dir "$FINETUNE_DIR" \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --split "$SPLIT" \
  --model-name finetuned_paddleocr_v1 \
  --rec-config "$REC_CONFIG" \
  "${GPU[@]}" \
  --paddle-dir "${PADDLE_DIR:-PaddleOCR}"
