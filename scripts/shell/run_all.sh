#!/usr/bin/env bash
# Run the Yorùbá OCR pipeline in order (default: phases 01–09, optional 99).
#
# DEFAULT DOES NOT include 15, 16, 17, 18 on purpose: they require GPU / Hugging Face
# downloads or heavier dependencies.
#
# Usage:
#   cd /path/to/yoruba_ocr_research
#   export DRIVE_BACKUP_ROOT="/content/drive/MyDrive/backup"   # optional
#   export EVAL_USE_GPU=1 CONFIG_FORCE_GPU=1 TRAIN_CPU=0       # Colab T4 example
#   bash scripts/shell/run_all.sh
#
# Subset of phases:
#   PHASES="01 02 03 04 05 09 99" SKIP_PADDLE_TRAIN=0 bash scripts/shell/run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

log "PROJECT_ROOT=$PROJECT_ROOT"
require_python
check_deps

# Default: core pipeline without long training.
# For VLM/SFT runs, include phases 14, 15, 16, 17, and 18 explicitly.
DEFAULT_PHASES="01 02 03 05 09 99"
PHASES="${PHASES:-$DEFAULT_PHASES}"

run_phase() {
  local id="$1"
  local f=""
  case "$id" in
    01) f="${SCRIPT_DIR}/phase_01_consolidate.sh" ;;
    02) f="${SCRIPT_DIR}/phase_02_analyze.sh" ;;
    03) f="${SCRIPT_DIR}/phase_03_config.sh" ;;
    04) f="${SCRIPT_DIR}/phase_04_train_paddleocr_recognition.sh" ;;
    05) f="${SCRIPT_DIR}/phase_05_eval_paddleocr_recognition.sh" ;;
    14) f="${SCRIPT_DIR}/phase_14_export_paddleocrvl16_sft.sh" ;;
    15) f="${SCRIPT_DIR}/phase_15_eval_paddleocrvl16_zero_shot.sh" ;;
    16) f="${SCRIPT_DIR}/phase_16_train_paddleocrvl16_sft.sh" ;;
    17) f="${SCRIPT_DIR}/phase_17_eval_paddleocrvl16_sft.sh" ;;
    18) f="${SCRIPT_DIR}/phase_18_eval_glm_ocr_zero_shot.sh" ;;
    09) f="${SCRIPT_DIR}/phase_09_compile.sh" ;;
    12) f="${SCRIPT_DIR}/phase_12_diagnose.sh" ;;
    13) f="${SCRIPT_DIR}/phase_13_verify_eval.sh" ;;
    99) f="${SCRIPT_DIR}/phase_99_backup.sh" ;;
    *) die "unknown phase id: $id (use 01-05, 09, 12, 13, 14, 15, 16, 17, 18, or 99)" ;;
  esac
  [[ -f "$f" ]] || die "missing $f"
  log "========== Phase $id: $(basename "$f") =========="
  bash "$f"
}

for p in $PHASES; do
  run_phase "$p"
done

log "run_all.sh finished."
