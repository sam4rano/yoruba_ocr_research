#!/usr/bin/env bash
# Phase 14: Export data/processed → JSONL for PaddleOCR-VL-1.6 SFT (read-only on source).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  export PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi

if [[ "${SKIP_PADDLEOCRVL16_SFT_EXPORT:-0}" == "1" ]]; then
  log "SKIP_PADDLEOCRVL16_SFT_EXPORT=1 — skipping phase 14"
  exit 0
fi

require_python
run_py scripts/export_paddleocrvl16_sft.py \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --out-dir "${PADDLEOCRVL16_SFT_EXPORT_DIR:-data/paddleocrvl16_sft}"
