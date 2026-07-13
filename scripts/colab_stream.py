"""
Stream subprocess output line-by-line for Jupyter / Colab notebooks.

Python block-buffers stdout when not attached to a TTY; long pipeline steps then
appear silent until the process exits. ``run_cmd`` sets ``PYTHONUNBUFFERED`` and
reads stdout incrementally so epoch logs and eval progress show in the cell.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from shlex import split
from typing import Mapping, Sequence, Union

Cmd = Union[str, Sequence[str]]


LEGACY_SCRIPT_NAMES = {
    "scripts/audit_data_quality.py": "scripts/02b_data_quality_audit.py",
    "scripts/refresh_dataset_report.py": "scripts/02c_refresh_dataset_report.py",
    "scripts/smoke_test_runtime.py": "scripts/24_colab_smoke_test.py",
}


def _validate_local_command(argv: Sequence[str], cwd: str | Path | None) -> None:
    """Fail clearly when a notebook and its checked-out scripts are out of sync."""
    if len(argv) < 2 or not argv[1].startswith("scripts/"):
        return

    root = Path(cwd) if cwd is not None else Path.cwd()
    requested = root / argv[1]
    if requested.is_file():
        return

    legacy_name = LEGACY_SCRIPT_NAMES.get(argv[1])
    legacy_hint = ""
    if legacy_name and (root / legacy_name).is_file():
        legacy_hint = (
            f" Found legacy file {legacy_name}; this confirms that the notebook "
            "is newer than the checked-out repository."
        )
    raise FileNotFoundError(
        f"Required command file is missing: {requested}.{legacy_hint} "
        "Rerun the notebook's 'Pull code from GitHub' cell and confirm its "
        "project-layout check passes before continuing."
    )


def run_cmd(
    cmd: Cmd,
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    label: str | None = None,
) -> int:
    """
    Run a command and stream combined stdout/stderr to the notebook cell.

    Raises ``subprocess.CalledProcessError`` on non-zero exit.
    """
    argv = split(cmd) if isinstance(cmd, str) else list(cmd)
    _validate_local_command(argv, cwd)
    merged = os.environ.copy()
    merged["PYTHONUNBUFFERED"] = "1"
    if env:
        merged.update({k: str(v) for k, v in env.items()})

    if label:
        print(f"▶ {label}", flush=True)
    print(f"$ {' '.join(argv)}", flush=True)

    proc = subprocess.Popen(
        argv,
        cwd=str(cwd) if cwd is not None else None,
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)

    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, argv)
    return rc


def run_phase(
    script_name: str,
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    label: str | None = None,
) -> int:
    """Run ``scripts/shell/<script_name>`` with live streaming."""
    phase_label = label or script_name
    return run_cmd(
        ["bash", f"scripts/shell/{script_name}"],
        env=env,
        cwd=cwd,
        label=phase_label,
    )


def ensure_importable() -> None:
    """Insert ``scripts/`` on ``sys.path`` when the notebook cwd is the repo root."""
    scripts = Path.cwd() / "scripts"
    if scripts.is_dir():
        root = str(scripts)
        if root not in sys.path:
            sys.path.insert(0, root)


def main() -> None:
    """CLI wrapper: ``python scripts/colab_stream.py bash scripts/shell/phase_04_train_paddleocr_recognition.sh``."""
    import argparse

    parser = argparse.ArgumentParser(description="Run a command with streamed output.")
    parser.add_argument("cmd", nargs="+", help="Command and arguments.")
    parser.add_argument("--cwd", type=Path, default=None)
    args = parser.parse_args()
    run_cmd(args.cmd, cwd=args.cwd)


if __name__ == "__main__":
    main()
