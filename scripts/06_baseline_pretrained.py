"""
Evaluate the English pretrained PaddleOCR recognition model (no fine-tuning)
on the test set to establish the baseline CER/WER/DER.

Critical (P4 fix): the English PP-OCRv3 checkpoint has a CTC head whose
output dimension matches the *English* dictionary (95 characters + blank).
Loading it against the Yorùbá dictionary (98 chars) produces a shape
mismatch that PaddleOCR's ``load_model`` silently skips, leaving a
randomly-initialised head. That was the root cause of the previous
"baseline CER ≈ 1.0" phantom result.

This script therefore evaluates the model with its own English dict so
the head weights actually load. The reported CER/WER are the honest
baseline for "English OCR applied to Yorùbá text": the model can output
ASCII characters approximating base letters but cannot produce
combining diacritics, so DER will be near 1.0.

Important: ``paddleocr`` (pip) v3.x is a PaddleX pipeline and its Python
API is not compatible with the legacy ``ocr(..., det=..., cls=...)``
signature. For consistent evaluation this script delegates to
``scripts/05_evaluate.py``, which loads checkpoints via the cloned
PaddleOCR repo (``ppocr.*``).
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
        help=(
            "Character dictionary passed to 05_evaluate.py. Defaults to "
            "PaddleOCR/ppocr/utils/en_dict.txt so the English pretrained "
            "checkpoint's CTC head loads correctly. Override with the "
            "Yorùbá dict only if you deliberately want to reproduce the "
            "phantom-head baseline (which the integrity gate will reject)."
        ),
    )
    parser.add_argument(
        "--use-yoruba-dict",
        "--use_yoruba_dict",
        dest="use_yoruba_dict",
        action="store_true",
        help=(
            "Force evaluation with the Yorùbá dictionary. Causes a CTC "
            "head shape mismatch and will trigger PhantomCheckpointError "
            "unless --allow-head-reinit is also passed through."
        ),
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
        "--rec-config",
        "--rec_config",
        dest="rec_config",
        type=Path,
        default=Path("configs/paddleocr_yoruba_rec.yml"),
        help=(
            "YAML passed to scripts/05_evaluate.py — must match the PP-OCRv3/v4 family of "
            "--pretrained-dir. Default matches 03_generate_config / 04_train."
        ),
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
    parser.add_argument(
        "--allow-head-reinit",
        "--allow_head_reinit",
        dest="allow_head_reinit",
        action="store_true",
        help=(
            "Pass --allow-head-reinit through to 05_evaluate.py. Only use "
            "together with --use-yoruba-dict to reproduce the phantom "
            "baseline on purpose."
        ),
    )
    return parser.parse_args()


def run_baseline(args: argparse.Namespace) -> None:
    """
    Load English pretrained model, run inference, compute and save metrics.

    The English model is loaded without a custom rec_model_dir — PaddleOCR
    will download and cache it automatically on first run.
    """
    if args.dict_path is not None:
        dict_path = args.dict_path
    elif args.use_yoruba_dict:
        dict_path = args.data_dir / "dictionary" / "yoruba_char_dict.txt"
    else:
        dict_path = args.paddle_dir / "ppocr" / "utils" / "en_dict.txt"

    if not dict_path.exists():
        log.error(
            "Dictionary not found at %s. The English baseline expects "
            "ppocr/utils/en_dict.txt inside the cloned PaddleOCR repo. "
            "Check --paddle-dir or pass --dict-path explicitly.",
            dict_path,
        )
        sys.exit(1)

    log.info("Using character dict: %s", dict_path)

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
    if args.allow_head_reinit:
        cmd.append("--allow-head-reinit")

    cmd += ["--rec-config", str(args.rec_config)]

    log.info("Running baseline evaluation:\n  %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    """Entry point for baseline evaluation."""
    args = parse_args()
    run_baseline(args)


if __name__ == "__main__":
    main()
