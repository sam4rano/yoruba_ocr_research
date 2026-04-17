#!/usr/bin/env bash
# Phase 12: Separate data vs eval vs setup (no GPU). Default: eval + identity + data.
# Optional: DIAG_ONLY=replay DIAG_JSONL=path/to.jsonl bash scripts/shell/phase_12_diagnose.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  export PYTHON="${PROJECT_ROOT}/.venv/bin/python"
fi

PROCESSED="${PROCESSED_DIR:-data/processed}"
SPLIT="${EVAL_SPLIT:-test}"

if [[ -n "${DIAG_ONLY:-}" ]]; then
  case "${DIAG_ONLY}" in
    eval)
      run_py scripts/12_diagnose_hypotheses.py eval
      ;;
    identity)
      run_py scripts/12_diagnose_hypotheses.py identity --data-dir "$PROCESSED" --split "$SPLIT"
      ;;
    data)
      run_py scripts/12_diagnose_hypotheses.py data \
        --data-dir "$PROCESSED" \
        --split "$SPLIT" \
        --sample "${DIAG_SAMPLE:-20}" \
        --seed "${DIAG_SEED:-42}"
      ;;
    replay)
      run_py scripts/12_diagnose_hypotheses.py replay \
        --jsonl "${DIAG_JSONL:-results/tables/qwen25_vl_zero_shot_test.jsonl}"
      ;;
    setup-hint)
      run_py scripts/12_diagnose_hypotheses.py setup-hint
      ;;
    *)
      die "DIAG_ONLY must be eval|identity|data|replay|setup-hint (got ${DIAG_ONLY})"
      ;;
  esac
  exit 0
fi

run_py scripts/12_diagnose_hypotheses.py eval
run_py scripts/12_diagnose_hypotheses.py identity --data-dir "$PROCESSED" --split "$SPLIT"
run_py scripts/12_diagnose_hypotheses.py data \
  --data-dir "$PROCESSED" \
  --split "$SPLIT" \
  --sample "${DIAG_SAMPLE:-20}" \
  --seed "${DIAG_SEED:-42}"
log "Optional: after Qwen eval, run DIAG_ONLY=replay bash scripts/shell/phase_12_diagnose.sh"
