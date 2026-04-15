#!/usr/bin/env bash
# Phase 6: Tesseract baselines (requires tesseract binary + lang packs on PATH).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

require_python
if [[ "${SKIP_TESSERACT:-0}" == "1" ]]; then
  log "SKIP_TESSERACT=1 — skipping phase 06"
  exit 0
fi

if ! command -v tesseract >/dev/null 2>&1; then
  log "ERROR: tesseract not on PATH. On Debian/Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-yor"
  exit 1
fi

# Default languages: eng, yor, eng+yor (quote the last token).
if [[ -n "${TESSERACT_LANGS:-}" ]]; then
  # shellcheck disable=SC2206
  LANGS=( $TESSERACT_LANGS )
else
  LANGS=( eng yor "eng+yor" )
fi

run_py scripts/07_baseline_tesseract.py \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --split "${EVAL_SPLIT:-test}" \
  --langs "${LANGS[@]}" \
  --psm "${TESSERACT_PSM:-7}" \
  --results-csv "${METRICS_CSV:-results/tables/metrics.csv}"
