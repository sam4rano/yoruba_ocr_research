"""
Fine-tune Surya Foundation OCR on Yorùbá line crops (datalab-to/surya v0.15.x).

Surya v2 (``surya-ocr>=0.20``) does not ship a public fine-tune entry point.
This script installs **surya 0.15.3** (Foundation stack), exports or loads the
HF dataset from ``26_export_surya_finetune.py``, and runs ``finetune_ocr.py``.

After training, evaluate with ``28_evaluate_surya_finetuned.py``. Re-install
``surya-ocr>=0.20`` if you need Surya v2 zero-shot again in the same env.

Usage:
    python scripts/26_export_surya_finetune.py
    python scripts/27_train_surya_finetune.py --epochs 1 --max-steps 500
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SURYA_FINETUNE_TAG = "v0.15.3"
DEFAULT_OUTPUT = Path("experiments/surya_finetune")
DEFAULT_DATASET_DIR = Path("data/hf_surya_finetune")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Surya Foundation OCR on Yorùbá line crops."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Local HF dataset (``save_to_disk``) or Hub repo id when ``--hub-dataset``.",
    )
    parser.add_argument(
        "--hub-dataset",
        action="store_true",
        help="Treat ``--dataset-path`` as a Hub dataset id string.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Training output directory.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Training epochs (HF TrainingArguments).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Cap steps (-1 = epoch-limited).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip install of surya@0.15.3.",
    )
    parser.add_argument(
        "--export-first",
        action="store_true",
        help="Run ``26_export_surya_finetune.py`` before training.",
    )
    return parser.parse_args()


def ensure_surya_finetune_stack() -> None:
    """Install surya 0.15.3 for Foundation fine-tuning."""
    log.info("Installing surya %s (Foundation fine-tune stack)...", SURYA_FINETUNE_TAG)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            f"git+https://github.com/datalab-to/surya@{SURYA_FINETUNE_TAG}",
        ]
    )


def resolve_dataset_name(args: argparse.Namespace) -> str:
    """Return HF ``load_dataset`` name/path for finetune_ocr.py."""
    if args.hub_dataset:
        return str(args.dataset_path)
    if not args.dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset_path}\n"
            "Run: python scripts/26_export_surya_finetune.py"
        )
    return str(args.dataset_path.resolve())


def _training_precision_flags() -> list[str]:
    """Return HF TrainingArguments precision flags safe for the current GPU."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    major, _minor = torch.cuda.get_device_capability(0)
    if major >= 8:
        return ["--bf16"]
    return ["--fp16"]


def run_finetune(args: argparse.Namespace, dataset_name: str) -> None:
    """Invoke surya ``finetune_ocr.py`` (with local ``save_to_disk`` patch if needed)."""
    import datasets as hf_datasets

    local_disk = (not args.hub_dataset) and Path(dataset_name).is_dir()
    if local_disk:
        from datasets import load_from_disk

        local_dsd = load_from_disk(dataset_name)
        real_load = hf_datasets.load_dataset

        def _patched_load(name, *load_args, **load_kwargs):
            if str(Path(name).resolve()) == str(Path(dataset_name).resolve()):
                split = load_kwargs.get("split", "train")
                if split not in local_dsd:
                    raise KeyError(f"Split {split!r} not in {dataset_name}")
                return local_dsd[split]
            return real_load(name, *load_args, **load_kwargs)

        hf_datasets.load_dataset = _patched_load
        dataset_name = str(Path(dataset_name).resolve())

    import surya.scripts.finetune_ocr as finetune_mod  # type: ignore

    args.output_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "finetune_ocr.py",
        f"--dataset_name={dataset_name}",
        f"--output_dir={args.output_dir.resolve()}",
        f"--num_train_epochs={args.epochs}",
        f"--per_device_train_batch_size={args.batch_size}",
        f"--learning_rate={args.lr}",
        *_training_precision_flags(),
        "--save_strategy=epoch",
        "--logging_steps=10",
        "--dataloader_num_workers=2",
    ]
    if args.max_steps > 0:
        argv.append(f"--max_steps={args.max_steps}")

    log.info("RUN finetune in-process: %s", " ".join(argv))
    prev_argv = sys.argv
    sys.argv = argv
    try:
        finetune_mod.main()
    finally:
        sys.argv = prev_argv


def main() -> None:
    """Export (optional), install stack, and fine-tune Surya."""
    args = parse_args()

    if args.export_first:
        subprocess.check_call([sys.executable, "scripts/26_export_surya_finetune.py"])

    if not args.skip_install:
        ensure_surya_finetune_stack()

    dataset_name = resolve_dataset_name(args)
    run_finetune(args, dataset_name)
    log.info("Surya fine-tune finished. Checkpoint under %s", args.output_dir)


if __name__ == "__main__":
    main()
