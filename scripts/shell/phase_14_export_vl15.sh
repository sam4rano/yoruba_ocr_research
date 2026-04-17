#!/usr/bin/env bash
# Phase 14: Export data/processed → JSONL for PaddleOCR-VL-1.5 SFT (read-only on source).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  export PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi

require_python
run_py scripts/14_export_paddleocr_vl_sft.py \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --out-dir "${PADDLE_VL15_EXPORT_DIR:-data/paddleocr_vl15_sft}"
