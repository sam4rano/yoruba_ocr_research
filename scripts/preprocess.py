"""Preprocess raw Yorùbá OCR data into a trainable structure."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def log_result(results_path: Path, payload: dict) -> None:
    """Append a preprocessing result payload to a JSONL file."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_preprocess(input_dir: Path, output_dir: Path) -> dict:
    """Perform a placeholder preprocessing pass over input files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images = list(input_dir.rglob("*"))
    file_count = len([item for item in images if item.is_file()])
    return {
        "stage": "preprocess",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "file_count": file_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess Yorùbá OCR data.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw OCR inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to write processed artifacts.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("results/tables/preprocess-log.jsonl"),
        help="JSONL file for reproducible run logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute preprocessing and log the run metadata."""
    args = parse_args()
    result = run_preprocess(args.input_dir, args.output_dir)
    log_result(args.log_file, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
