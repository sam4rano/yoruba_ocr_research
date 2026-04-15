#!/usr/bin/env bash
# Phase 7: Qwen 2.5 VL zero-shot (heavy download + GPU RAM). Skipped by default.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ "${SKIP_QWEN:-1}" == "1" ]]; then
  log "SKIP_QWEN=1 (default) — set SKIP_QWEN=0 to run Qwen baseline"
  exit 0
fi

require_python
QWEN_ARGS=(
  scripts/09_baseline_qwen.py
  --data-dir "${PROCESSED_DIR:-data/processed}"
  --split "${EVAL_SPLIT:-test}"
  --results-csv "${METRICS_CSV:-results/tables/metrics.csv}"
  --batch-size "${QWEN_BATCH_LOG:-10}"
)
[[ -n "${QWEN_MODEL_ID:-}" ]] && QWEN_ARGS+=( --model-id "$QWEN_MODEL_ID" )
[[ -n "${QWEN_MAX_SAMPLES:-}" ]] && QWEN_ARGS+=( --max-samples "$QWEN_MAX_SAMPLES" )
[[ "${QWEN_QUANTIZE:-1}" == "1" ]] && QWEN_ARGS+=( --quantize )

run_py "${QWEN_ARGS[@]}"
