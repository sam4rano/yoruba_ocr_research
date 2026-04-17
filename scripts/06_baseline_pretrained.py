"""
Evaluate the English pretrained PaddleOCR recognition model (no fine-tuning)
on the test set to establish the baseline CER/WER/DER.

Important: `paddleocr` (pip) v3.x is a PaddleX pipeline and its Python API is
not compatible with the legacy `ocr(..., det=..., cls=...)` call signature.
For consistent evaluation (and to ensure we are comparing like-for-like),
this script delegates to `scripts/05_evaluate.py`, which loads checkpoints
via the cloned PaddleOCR repo (`ppocr.*`).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "baseline_english_pretrained"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for baseline evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the English pretrained PaddleOCR model on Yorùbá OCR test data."
        )
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        type=Path,
        default=Path("data/processed"),
        help="Consolidated dataset root.",
    )
    parser.add_argument(
        "--dict-path",
        "--dict_path",
        dest="dict_path",
        type=Path,
        default=None,
        help="Yorùbá character dictionary. Defaults to data-dir/dictionary/yoruba_char_dict.txt.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--use-gpu",
        "--use_gpu",
        dest="use_gpu",
        action="store_true",
        default=False,
        help="Run inference on GPU (only meaningful on CUDA machines).",
    )
    parser.add_argument(
        "--paddle-dir",
        "--paddle_dir",
        dest="paddle_dir",
        type=Path,
        default=Path("PaddleOCR"),
        help="Path to the cloned PaddleOCR repository.",
    )
    parser.add_argument(
        "--pretrained-dir",
        "--pretrained_dir",
        dest="pretrained_dir",
        type=Path,
        default=Path("experiments/baseline/pretrained/en_PP-OCRv3_rec_train"),
        help="Directory containing the extracted pretrained checkpoint files.",
    )
    parser.add_argument(
        "--results-csv",
        "--results_csv",
        dest="results_csv",
        type=Path,
        default=Path("results/tables/metrics.csv"),
        help="Shared results table — appended on each run.",
    )
    parser.add_argument(
        "--per-sample-log",
        "--per_sample_log",
        dest="per_sample_log",
        type=Path,
        default=Path(f"results/tables/{MODEL_NAME}_test.jsonl"),
        help="JSONL file for per-sample predictions.",
    )
    return parser.parse_args()


def run_baseline(args: argparse.Namespace) -> None:
    """
    Load English pretrained model, run inference, compute and save metrics.

    The English model is loaded without a custom rec_model_dir — PaddleOCR
    will download and cache it automatically on first run.
    """
    dict_path = args.dict_path or (args.data_dir / "dictionary" / "yoruba_char_dict.txt")

    # Delegate evaluation to scripts/05_evaluate.py for consistent checkpoint loading.
    cmd = [
        sys.executable,
        "scripts/05_evaluate.py",
        "--model-dir",
        str(args.pretrained_dir / "best_accuracy.pdparams"),
        "--data-dir",
        str(args.data_dir),
        "--split",
        args.split,
        "--model-name",
        MODEL_NAME,
        "--dict-path",
        str(dict_path),
        "--results-csv",
        str(args.results_csv),
        "--per-sample-log",
        str(args.per_sample_log),
        "--paddle-dir",
        str(args.paddle_dir),
    ]
    if args.use_gpu:
        cmd.append("--use-gpu")

    log.info("Running baseline evaluation:\n  %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    """Entry point for baseline evaluation."""
    args = parse_args()
    run_baseline(args)


if __name__ == "__main__":
    main()
