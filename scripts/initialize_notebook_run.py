"""Initialize notebook outputs exactly once for a stable experiment run ID."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from metrics_lifecycle import reset_generated_artifacts

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$")


def initialize_run(run_id: str, tables_dir: Path, reset: bool) -> bool:
    """Reset generated outputs once for ``run_id`` and return whether reset ran."""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must be 3-80 characters using letters, numbers, dot, dash, or underscore"
        )
    tables_dir.mkdir(parents=True, exist_ok=True)
    marker = tables_dir / ".active_notebook_run_id"
    active = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    if active == run_id:
        print(f"Notebook run already initialized: {run_id}")
        return False
    if not reset:
        raise RuntimeError(
            f"Active notebook run is {active or 'unset'}, not {run_id}. "
            "Set RUN_RESET=True once when intentionally starting this run ID."
        )
    reset_generated_artifacts(tables_dir)
    marker.write_text(run_id + "\n", encoding="utf-8")
    print(f"Initialized fresh notebook run: {run_id}")
    return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Reset generated OCR artifacts once per notebook run ID."
    )
    parser.add_argument("--run-id", required=True, help="Stable ID for this experiment run.")
    parser.add_argument(
        "--tables-dir", type=Path, default=Path("results/tables")
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Permit reset when the requested run ID is not already active.",
    )
    return parser.parse_args()


def main() -> None:
    """Initialize the selected notebook run."""
    args = parse_args()
    initialize_run(args.run_id, args.tables_dir, args.reset)


if __name__ == "__main__":
    main()
