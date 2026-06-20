#!/usr/bin/env bash
# Phase 15: PaddleOCR-VL-1.6 zero-shot eval.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ "${SKIP_VL16_ZERO_SHOT:-0}" == "1" ]]; then
  log "WARN: SKIP_VL16_ZERO_SHOT=1 — skipping PaddleOCR-VL-1.6 eval."
  exit 0
fi

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  export PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi

export HF_TRUST_REMOTE_CODE="${HF_TRUST_REMOTE_CODE:-1}"

VL16_ARGS=(
  scripts/15_baseline_paddleocr_vl16.py
  --data-dir "${PROCESSED_DIR:-data/processed}"
  --split "${EVAL_SPLIT:-test}"
  --results-csv "${METRICS_CSV:-results/tables/metrics.csv}"
)
[[ -n "${VL16_MAX_SAMPLES:-}" ]] && VL16_ARGS+=( --max-samples "$VL16_MAX_SAMPLES" )
[[ "${VL16_QUANTIZE_4BIT:-0}" == "1" ]] && VL16_ARGS+=( --quantize-4bit )

require_python
run_py "${VL16_ARGS[@]}"
