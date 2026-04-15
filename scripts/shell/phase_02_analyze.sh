#!/usr/bin/env bash
# Phase 2: EDA + figures under results/tables/figures/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

require_python
run_py scripts/02_analyze_dataset.py \
  --data-dir "${PROCESSED_DIR:-data/processed}" \
  --output-dir "${RESULTS_TABLES:-results/tables}" \
  --plot
