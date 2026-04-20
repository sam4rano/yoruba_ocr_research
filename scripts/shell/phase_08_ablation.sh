#!/usr/bin/env bash
# Phase 8: ablation study (multiple training runs). Skipped by default.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ "${SKIP_ABLATION:-1}" == "1" ]]; then
  log "WARN: SKIP_ABLATION=1 (default) — skipping ablation studies."
  log "WARN: You are not running the full paper evaluation. Set SKIP_ABLATION=0 to run."
  exit 0
fi

require_python
run_py scripts/10_ablation_study.py \
  --ablation "${ABLATION_WHICH:-all}" \
  --gpus "${ABLATION_GPUS:-0}" \
  --epochs "${ABLATION_EPOCHS:-40}"
