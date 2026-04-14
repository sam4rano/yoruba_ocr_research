"""Train Yorùbá OCR models and log reproducible metadata."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def log_result(results_path: Path, payload: dict) -> None:
    """Append a training result payload to a JSONL file."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_training(
    data_split: Path,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
) -> dict:
    """Run a placeholder training routine and return run metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint-last.pt"
    checkpoint_path.write_text("placeholder checkpoint", encoding="utf-8")
    return {
        "stage": "train",
        "split_file": str(data_split),
        "output_dir": str(output_dir),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "checkpoint": str(checkpoint_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training."""
    parser = argparse.ArgumentParser(description="Train Yorùbá OCR model.")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("data/splits/train.csv"),
        help="Training split file path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/baseline"),
        help="Directory for model artifacts.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("results/tables/train-log.jsonl"),
        help="JSONL file for reproducible run logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute training and record run metadata."""
    args = parse_args()
    result = run_training(
        data_split=args.split_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    log_result(args.log_file, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
