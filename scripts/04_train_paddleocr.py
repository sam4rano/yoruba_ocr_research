"""
Run PaddleOCR fine-tuning and log training metadata.

Expects:
  - PaddleOCR repo cloned at --paddle-dir (default: ./PaddleOCR)
  - Fine-tuning config at --config (default: configs/paddleocr_yoruba_rec.yml)
  - Consolidated data already built by 01_consolidate_data.py
  - Config paths already resolved by 03_generate_config.py

Setup (run once before this script):
    git clone https://github.com/PaddlePaddle/PaddleOCR PaddleOCR
    pip install -r requirements.txt

Usage:
    python scripts/04_train_paddleocr.py
    python scripts/04_train_paddleocr.py \
        --config configs/paddleocr_yoruba_rec.yml \
        --paddle-dir PaddleOCR \
        --gpus 0 \
        --log-file results/tables/train_run.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def verify_prerequisites(paddle_dir: Path, config: Path) -> None:
    """Check that the PaddleOCR repo and config file exist before starting."""
    if not paddle_dir.exists():
        log.error(
            "PaddleOCR directory not found: %s\n"
            "Clone it with: git clone https://github.com/PaddlePaddle/PaddleOCR %s",
            paddle_dir,
            paddle_dir,
        )
        sys.exit(1)
    if not (paddle_dir / "tools" / "train.py").exists():
        log.error("tools/train.py not found in %s — check your clone.", paddle_dir)
        sys.exit(1)
    if not config.exists():
        log.error(
            "Config not found: %s\n"
            "Run scripts/03_generate_config.py first.",
            config,
        )
        sys.exit(1)


def build_train_command(
    paddle_dir: Path,
    config: Path,
    gpus: str,
    extra_overrides: list[str],
) -> list[str]:
    """
    Construct the PaddleOCR training command.

    Multi-GPU training uses `python -m paddle.distributed.launch`.
    Single-GPU or CPU training uses plain `python tools/train.py`.
    """
    n_gpus = len(gpus.split(",")) if gpus else 0
    # Must be absolute: cwd is paddle_dir, so a relative train path would double the segment.
    train_script = str((paddle_dir / "tools" / "train.py").resolve())

    if n_gpus > 1:
        cmd = [
            sys.executable,
            "-m", "paddle.distributed.launch",
            f"--gpus={gpus}",
            train_script,
        ]
    else:
        cmd = [sys.executable, train_script]

    cmd += ["-c", str(config.resolve())]
    if extra_overrides:
        cmd += ["-o"] + extra_overrides

    return cmd


def run_training(
    cmd: list[str],
    paddle_dir: Path,
    log_file: Path,
) -> dict:
    """
    Execute the training command and stream output to the terminal.

    Returns a metadata dict with timing and exit status.
    """
    log.info("Starting training:\n  %s", " ".join(cmd))
    start = time.time()

    env = os.environ.copy()
    # Ensure PaddleOCR's own modules are importable
    env["PYTHONPATH"] = str(paddle_dir) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        cwd=str(paddle_dir),
        env=env,
    )

    elapsed = round(time.time() - start, 1)
    success = result.returncode == 0

    if success:
        log.info("Training finished in %.1f s.", elapsed)
    else:
        log.error(
            "Training exited with code %d after %.1f s.",
            result.returncode,
            elapsed,
        )

    return {
        "stage": "train",
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "success": success,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR fine-tuning for Yorùbá OCR."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paddleocr_yoruba_rec.yml"),
        help="PaddleOCR YAML config file.",
    )
    parser.add_argument(
        "--paddle-dir",
        type=Path,
        default=Path("PaddleOCR"),
        help="Path to the cloned PaddleOCR repository.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs (e.g. '0,1'). Leave empty for CPU.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch count from config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("results/tables/train_run.json"),
        help="JSON file to record training metadata.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Override config: Global.use_gpu=false (macOS / CPU-only Paddle).",
    )
    return parser.parse_args()


def main() -> None:
    """Verify prerequisites, launch training, and persist run metadata."""
    args = parse_args()

    verify_prerequisites(args.paddle_dir, args.config)

    # Build any -o override flags from CLI
    overrides: list[str] = []
    if args.epochs is not None:
        overrides.append(f"Global.epoch_num={args.epochs}")
    if args.batch_size is not None:
        overrides.append(f"Train.loader.batch_size_per_card={args.batch_size}")
    if args.lr is not None:
        overrides.append(f"Optimizer.lr.learning_rate={args.lr}")
    if args.cpu:
        overrides.append("Global.use_gpu=false")

    cmd = build_train_command(args.paddle_dir, args.config, args.gpus, overrides)
    result = run_training(cmd, args.paddle_dir, args.log_file)

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    log.info("Training metadata saved to %s", args.log_file)

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
