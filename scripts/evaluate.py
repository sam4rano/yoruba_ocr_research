"""Evaluate Yorùbá OCR checkpoints and log CER/WER/DER metrics."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def log_jsonl(results_path: Path, payload: dict) -> None:
    """Append evaluation output to a JSONL log."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log_csv(table_path: Path, payload: dict) -> None:
    """Append evaluation output to the project metrics table."""
    table_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = table_path.exists()
    fieldnames = ["model", "cer", "wer", "der", "notes"]
    with table_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "model": payload["model"],
                "cer": payload["cer"],
                "wer": payload["wer"],
                "der": payload["der"],
                "notes": payload["notes"],
            }
        )


def run_evaluation(model_name: str, checkpoint: Path, split_file: Path) -> dict:
    """Run a placeholder evaluation and return traceable metric values."""
    _ = checkpoint, split_file
    return {
        "stage": "evaluate",
        "model": model_name,
        "cer": 0.0,
        "wer": 0.0,
        "der": 0.0,
        "notes": "placeholder metrics; replace with real evaluator output",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Yorùbá OCR model.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="baseline",
        help="Model label for results table.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/baseline/checkpoint-last.pt"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("data/splits/val.csv"),
        help="Validation/test split file path.",
    )
    parser.add_argument(
        "--json-log-file",
        type=Path,
        default=Path("results/tables/eval-log.jsonl"),
        help="JSONL log for evaluation runs.",
    )
    parser.add_argument(
        "--table-file",
        type=Path,
        default=Path("results/tables/metrics.csv"),
        help="CSV table for manuscript-ready metric entries.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute evaluation and persist reproducible outputs."""
    args = parse_args()
    result = run_evaluation(
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        split_file=args.split_file,
    )
    log_jsonl(args.json_log_file, result)
    log_csv(args.table_file, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
