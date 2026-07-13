#!/usr/bin/env bash
# Phase: GLM-OCR zero-shot eval.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ "${SKIP_GLM_ZERO_SHOT:-0}" == "1" ]]; then
  log "WARN: SKIP_GLM_ZERO_SHOT=1 — skipping GLM-OCR eval."
  exit 0
fi
if model_result_complete glm_ocr_zero_shot "${EVAL_SPLIT:-test}"; then
  log "Complete GLM-OCR evidence already exists; skipping evaluation."
  exit 0
fi

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  export PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi

export HF_TRUST_REMOTE_CODE="${HF_TRUST_REMOTE_CODE:-1}"

GLM_ARGS=(
  scripts/eval_glm_ocr.py
  --data-dir "${PROCESSED_DIR:-data/processed}"
  --split "${EVAL_SPLIT:-test}"
  --results-csv "${METRICS_CSV:-results/tables/metrics.csv}"
)
[[ -n "${GLM_MAX_SAMPLES:-}" ]] && GLM_ARGS+=( --max-samples "$GLM_MAX_SAMPLES" )
[[ -n "${GLM_MAX_NEW_TOKENS:-}" ]] && GLM_ARGS+=( --max-new-tokens "$GLM_MAX_NEW_TOKENS" )
[[ -n "${GLM_BATCH_SIZE:-}" ]] && GLM_ARGS+=( --batch-size "$GLM_BATCH_SIZE" )
[[ "${GLM_NO_RESUME:-0}" == "1" ]] && GLM_ARGS+=( --no-resume )
[[ "${GLM_ALLOW_FAILURES:-0}" == "1" ]] && GLM_ARGS+=( --allow-failures )
# GLM_QUANTIZE_4BIT=0: use hardware-native dtype (bf16 on L4/A100, fp16 on T4).
# GLM_QUANTIZE_4BIT=1: use 4-bit quantization only when VRAM is critically limited.
[[ "${GLM_QUANTIZE_4BIT:-0}" == "1" ]] && GLM_ARGS+=( --quantize-4bit )

require_python
run_py "${GLM_ARGS[@]}"
