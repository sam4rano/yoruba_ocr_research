#!/usr/bin/env bash
# Phase 99: copy results (+ optional experiments) to a persistent path (e.g. Google Drive).
# Does not replace git; use GIT_SNAPSHOT=1 for an optional commit of results/tables.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

DEST="${DRIVE_BACKUP_ROOT:-}"
if [[ -z "$DEST" ]]; then
  log "DRIVE_BACKUP_ROOT not set — nothing to copy (set to e.g. /content/drive/MyDrive/yoruba_ocr_backup)"
  exit 0
fi

LABEL="${BACKUP_LABEL:-$(date +%Y%m%d_%H%M%S)}"
backup_artifacts "$DEST" "$LABEL"
snapshot_git "${GIT_COMMIT_MSG:-chore(results): pipeline backup ${LABEL}}"
