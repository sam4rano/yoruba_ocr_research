#!/usr/bin/env bash
# Phase 15: PaddleOCR-VL-1.5 zero-shot eval (GPU + HF download). Skipped by default.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ "${SKIP_VL15_EVAL:-1}" == "1" ]]; then
  log "WARN: SKIP_VL15_EVAL=1 (default) — skipping PaddleOCR-VL-1.5 eval."
  log "WARN: Set SKIP_VL15_EVAL=0 to run (needs GPU, transformers>=5)."
  exit 0
fi

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  export PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi

VL15_ARGS=(
  scripts/15_baseline_paddleocr_vl15.py
  --data-dir "${PROCESSED_DIR:-data/processed}"
  --split "${EVAL_SPLIT:-test}"
  --results-csv "${METRICS_CSV:-results/tables/metrics.csv}"
)
[[ -n "${VL15_MAX_SAMPLES:-}" ]] && VL15_ARGS+=( --max-samples "$VL15_MAX_SAMPLES" )
[[ "${VL15_QUANTIZE_4BIT:-0}" == "1" ]] && VL15_ARGS+=( --quantize-4bit )

require_python
run_py "${VL15_ARGS[@]}"
