#!/usr/bin/env bash
# Phase 28: Evaluate fine-tuned Surya Foundation checkpoint.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

require_python

ARGS=(
  --data-dir "${PROCESSED_DIR:-data/processed}"
  --split "${EVAL_SPLIT:-test}"
  --results-csv "${METRICS_CSV:-results/tables/metrics.csv}"
)
if [[ -n "${SURYA_FINETUNE_CKPT:-}" ]]; then
  ARGS+=(--checkpoint "${SURYA_FINETUNE_CKPT}")
fi

run_py scripts/28_evaluate_surya_finetuned.py "${ARGS[@]}"
