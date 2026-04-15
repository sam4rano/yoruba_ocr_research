#!/usr/bin/env bash
# Run the full Yorùbá OCR pipeline in order (phases 01–09, optional 99).
# Usage:
#   cd /path/to/yoruba_ocr_research
#   export DRIVE_BACKUP_ROOT="/content/drive/MyDrive/backup"   # optional
#   export EVAL_USE_GPU=1 CONFIG_FORCE_GPU=1 TRAIN_CPU=0       # Colab T4 example
#   bash scripts/shell/run_all.sh
#
# Subset of phases:
#   PHASES="02 03 04 05 09 99" bash scripts/shell/run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

log "PROJECT_ROOT=$PROJECT_ROOT"
require_python

DEFAULT_PHASES="01 02 03 04 05 06 07 08 09 99"
PHASES="${PHASES:-$DEFAULT_PHASES}"

run_phase() {
  local id="$1"
  local f=""
  case "$id" in
    01) f="${SCRIPT_DIR}/phase_01_consolidate.sh" ;;
    02) f="${SCRIPT_DIR}/phase_02_analyze.sh" ;;
    03) f="${SCRIPT_DIR}/phase_03_config.sh" ;;
    04) f="${SCRIPT_DIR}/phase_04_train.sh" ;;
    05) f="${SCRIPT_DIR}/phase_05_eval_paddle.sh" ;;
    06) f="${SCRIPT_DIR}/phase_06_tesseract.sh" ;;
    07) f="${SCRIPT_DIR}/phase_07_qwen.sh" ;;
    08) f="${SCRIPT_DIR}/phase_08_ablation.sh" ;;
    09) f="${SCRIPT_DIR}/phase_09_compile.sh" ;;
    99) f="${SCRIPT_DIR}/phase_99_backup.sh" ;;
    *) die "unknown phase id: $id (use 01–09 or 99)" ;;
  esac
  [[ -f "$f" ]] || die "missing $f"
  log "========== Phase $id: $(basename "$f") =========="
  bash "$f"
}

for p in $PHASES; do
  run_phase "$p"
done

log "run_all.sh finished."
