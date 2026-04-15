#!/usr/bin/env bash
# Phase 3: generate PaddleOCR YAML + fetch English pretrained weights if missing.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

require_python
CFG=(
  scripts/03_generate_config.py
  --data-dir "${PROCESSED_DIR:-data/processed}"
  --output-config "${TRAIN_CONFIG:-configs/paddleocr_yoruba_rec.yml}"
  --pretrained-dir "${PRETRAINED_ROOT:-experiments/baseline/pretrained}"
  --log-file "${CONFIG_LOG:-results/tables/config_generation.json}"
)
[[ -n "${CONFIG_EPOCHS:-}" ]] && CFG+=( --epochs "${CONFIG_EPOCHS}" )
[[ -n "${CONFIG_BATCH:-}" ]] && CFG+=( --batch-size "${CONFIG_BATCH}" )
[[ -n "${CONFIG_LR:-}" ]] && CFG+=( --lr "${CONFIG_LR}" )
[[ "${CONFIG_CPU:-0}" == "1" ]] && CFG+=( --cpu )
[[ "${CONFIG_FORCE_GPU:-0}" == "1" ]] && CFG+=( --force-gpu )
[[ "${CONFIG_SKIP_DOWNLOAD:-0}" == "1" ]] && CFG+=( --skip-download )

run_py "${CFG[@]}"
