#!/usr/bin/env bash
# Shared helpers for Yorùbá OCR pipeline shell scripts.
# shellcheck disable=SC2034  # exported vars used by callers

set -euo pipefail

# Resolve project root: .../yoruba_ocr_research (parent of scripts/)
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_script_dir}/../.." && pwd)}"
export PYTHON="${PYTHON:-python3}"

log() {
  printf '[%s] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

cd_project() {
  cd "$PROJECT_ROOT" || die "cannot cd to PROJECT_ROOT=$PROJECT_ROOT"
}

run_py() {
  cd_project
  log "RUN: $PYTHON $*"
  "$PYTHON" "$@"
}

run_py_keep() {
  cd_project
  log "RUN: $PYTHON $*"
  "$PYTHON" "$@"
}

require_python() {
  command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON not found"
}

# Backup results (and optionally experiments) to a second location (e.g. Google Drive).
# Usage: backup_artifacts /path/to/Drive/backup_root [run_label]
backup_artifacts() {
  local dest_root="${1:-}"
  local label="${2:-$(date +%Y%m%d_%H%M%S)}"
  [[ -n "$dest_root" ]] || die "backup_artifacts: destination required"

  cd_project
  local dest="${dest_root%/}/${label}"
  mkdir -p "$dest"

  if [[ -d results ]]; then
    log "Backup: results/ -> $dest/results"
    cp -a results "$dest/" || die "copy results failed"
  else
    log "WARN: no results/ to backup"
  fi

  if [[ "${BACKUP_EXPERIMENTS:-1}" == "1" ]] && [[ -d experiments ]]; then
    log "Backup: experiments/ -> $dest/experiments (large)"
    cp -a experiments "$dest/" || die "copy experiments failed"
  fi

  if [[ -f configs/paddleocr_yoruba_rec.yml ]]; then
    mkdir -p "$dest/configs_snippet"
    cp -a configs/paddleocr_yoruba_rec.yml "$dest/configs_snippet/" 2>/dev/null || true
  fi

  # Optional: final training YAML used for the main run
  if [[ -f configs/paddleocr_yoruba_rec_final.yml ]]; then
    mkdir -p "$dest/configs_snippet"
    cp -a configs/paddleocr_yoruba_rec_final.yml "$dest/configs_snippet/" 2>/dev/null || true
  fi

  mkdir -p "${PROJECT_ROOT}/results/tables"
  echo "$dest" > "${PROJECT_ROOT}/results/tables/.last_drive_backup_path.txt"
  log "Backup complete: $dest"
  log "Recorded path in results/tables/.last_drive_backup_path.txt"
}

# "Repo" copy: artifacts already live under PROJECT_ROOT; optional git snapshot message.
snapshot_git() {
  local msg="${1:-chore: pipeline artifacts}"
  cd_project
  if ! command -v git >/dev/null 2>&1; then
    log "WARN: git not found; skip snapshot_git"
    return 0
  fi
  if [[ ! -d .git ]]; then
    log "WARN: not a git repo; skip snapshot_git"
    return 0
  fi
  if [[ "${GIT_SNAPSHOT:-0}" != "1" ]]; then
    log "GIT_SNAPSHOT!=1; skip git add/commit (set GIT_SNAPSHOT=1 to enable)"
    return 0
  fi
  git add -A results/tables 2>/dev/null || true
  git status --short
  git commit -m "$msg" || log "WARN: git commit skipped (nothing to commit or hook failed)"
}
