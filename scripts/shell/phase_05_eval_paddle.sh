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
  --paddle-dir "${PADDLE_DIR:-PaddleOCR}"

run_py scripts/05_evaluate.py \
  --model-dir "$FINETUNE_DIR" \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --split "$SPLIT" \
  --model-name finetuned_paddleocr_v1 \
  --rec-config "$REC_CONFIG" \
  "${GPU[@]}" \
  --paddle-dir "${PADDLE_DIR:-PaddleOCR}"
