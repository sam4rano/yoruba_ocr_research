#!/usr/bin/env bash
# Phase 1: merge raw exports into data/processed (skip if data already final).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

require_python
if [[ "${SKIP_CONSOLIDATE:-0}" == "1" ]]; then
  log "SKIP_CONSOLIDATE=1 — skipping phase 01"
  exit 0
fi

run_py scripts/01_consolidate_data.py \
  --raw-dir "${RAW_DIR:-data/raw}" \
  --output-dir "${PROCESSED_DIR:-data/processed}" \
  --log-file "${CONSOLIDATE_LOG:-results/tables/consolidation_report.json}"

run_py scripts/08_export_tesseract_finetune.py \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --out-dir "data/tesseract_finetune"

run_py scripts/08_export_qwen_finetune.py \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --out-dir "data/qwen_finetune"
