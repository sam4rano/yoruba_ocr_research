#!/usr/bin/env bash
# Phase 27: Surya Foundation fine-tune (v0.15.x stack).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

require_python

ARGS=(
  --output-dir "${SURYA_FINETUNE_OUTPUT:-experiments/surya_finetune}"
  --epochs "${SURYA_FINETUNE_EPOCHS:-3}"
  --batch-size "${SURYA_FINETUNE_BATCH_SIZE:-4}"
  --lr "${SURYA_FINETUNE_LR:-2e-5}"
)

if [[ -n "${SURYA_FINETUNE_HUB_DATASET:-}" ]]; then
  ARGS+=(--hub-dataset --dataset-path "${SURYA_FINETUNE_HUB_DATASET}")
else
  ARGS+=(--dataset-path "${SURYA_FINETUNE_DATASET_DIR:-data/hf_surya_finetune}")
  if [[ "${SURYA_FINETUNE_EXPORT_FIRST:-1}" == "1" ]]; then
    ARGS+=(--export-first)
  fi
fi

if [[ -n "${SURYA_FINETUNE_MAX_STEPS:-}" ]]; then
  ARGS+=(--max-steps "${SURYA_FINETUNE_MAX_STEPS}")
fi

run_py scripts/27_train_surya_finetune.py "${ARGS[@]}"
