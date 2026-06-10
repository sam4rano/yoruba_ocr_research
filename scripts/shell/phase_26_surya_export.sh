#!/usr/bin/env bash
# Phase 26: Export HF dataset for Surya Foundation fine-tuning.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

require_python

ARGS=(--data-dir "${PROCESSED_DIR:-data/processed}")
if [[ -n "${SURYA_FINETUNE_DATASET_DIR:-}" ]]; then
  ARGS+=(--export-dir "${SURYA_FINETUNE_DATASET_DIR}")
fi
if [[ "${SURYA_EXPORT_PUSH:-0}" == "1" && -n "${SURYA_FINETUNE_HUB_DATASET:-}" ]]; then
  ARGS+=(--push --repo-id "${SURYA_FINETUNE_HUB_DATASET}")
fi

run_py scripts/26_export_surya_finetune.py "${ARGS[@]}"
