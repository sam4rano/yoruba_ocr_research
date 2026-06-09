"""
Fine-tune microsoft/trocr-large-printed on Yorùbá line crops.

Trains an encoder–decoder OCR model (ViT + RoBERTa decoder) on PNG line images
and NFC-normalised transcriptions from ``data/processed/labels/*.txt``.

Default training uses all labelled splits (~2,945 lines: train+val+test). Pass
``--hold-out-test`` to exclude the test split (~2,650 lines) for strict
held-out evaluation.

Usage:
    pip install transformers datasets torch torchvision accelerate
    python scripts/21_train_trocr.py --epochs 10
    python scripts/21_train_trocr.py --hold-out-test --epochs 15 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "microsoft/trocr-large-printed"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune TrOCR-large-printed on Yorùbá line crops."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Consolidated dataset root.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face TrOCR checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/trocr_large_printed"),
        help="Training output directory.",
    )
    parser.add_argument(
        "--hold-out-test",
        action="store_true",
        help="Exclude test split from training (train+val only, ~2650 lines).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap training samples (debug).",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split for validation during training (ignored if empty).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable validation during training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume Hugging Face Trainer from last checkpoint in --output-dir.",
    )
    return parser.parse_args()


def load_split_pairs(data_dir: Path, split: str):
    """Load (image_path, text) pairs for one split."""
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_utils import load_test_pairs  # noqa: E402

    return load_test_pairs(data_dir, split)


def build_hf_dataset(pairs: list[tuple[Path, str]]):
    """Convert path/text pairs to a Hugging Face Dataset with image paths."""
    from datasets import Dataset  # type: ignore

    return Dataset.from_dict(
        {
            "image_path": [str(p) for p, _ in pairs],
            "text": [t for _, t in pairs],
        }
    )


def main() -> None:
    """Fine-tune TrOCR and save the best checkpoint."""
    args = parse_args()

    try:
        import numpy as np
        import torch
        from PIL import Image  # type: ignore
        from transformers import (  # type: ignore
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            TrOCRProcessor,
            VisionEncoderDecoderModel,
            default_data_collator,
            set_seed,
        )
    except ImportError as exc:
        raise ImportError(
            "Run: pip install transformers datasets torch torchvision accelerate"
        ) from exc

    set_seed(args.seed)

    train_splits = ("train", "val") if args.hold_out_test else ("train", "val", "test")
    train_pairs: list[tuple[Path, str]] = []
    for split in train_splits:
        train_pairs.extend(load_split_pairs(args.data_dir, split))
    if args.max_samples:
        train_pairs = train_pairs[: args.max_samples]
    log.info(
        "Training on %d lines from splits %s.",
        len(train_pairs),
        ", ".join(train_splits),
    )

    eval_pairs: list[tuple[Path, str]] = []
    if not args.no_eval:
        eval_pairs = load_split_pairs(args.data_dir, args.eval_split)
        if args.max_samples:
            eval_pairs = eval_pairs[: min(args.max_samples, len(eval_pairs))]

    processor = TrOCRProcessor.from_pretrained(args.model_id)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_id)

    if torch.cuda.is_available():
        model = model.to("cuda")
        train_device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        model = model.to("mps")
        train_device = "mps"
    else:
        train_device = "cpu"
    log.info("TrOCR training device: %s", train_device)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    class TrocrLineDataset(torch.utils.data.Dataset):
        """Line-crop image + transcription for TrOCR."""

        def __init__(self, pairs: list[tuple[Path, str]], proc: TrOCRProcessor):
            self.pairs = pairs
            self.processor = proc

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int) -> dict:
            img_path, text = self.pairs[idx]
            with Image.open(img_path).convert("RGB") as img:
                pixel_values = self.processor(img, return_tensors="pt").pixel_values
            labels = self.processor.tokenizer(
                text,
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            return {
                "pixel_values": pixel_values.squeeze(),
                "labels": labels.squeeze(),
            }

    train_dataset = TrocrLineDataset(train_pairs, processor)
    eval_dataset = TrocrLineDataset(eval_pairs, processor) if eval_pairs else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_total_limit=2,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    resume_ckpt = None
    if args.resume:
        checkpoints = sorted(
            args.output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        if checkpoints:
            resume_ckpt = str(checkpoints[-1])
            log.info("Resuming TrOCR from %s", resume_ckpt)

    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    best_dir = args.output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))

    manifest = {
        "model_id": args.model_id,
        "train_splits": list(train_splits),
        "n_train": len(train_pairs),
        "eval_split": args.eval_split if eval_dataset else None,
        "n_eval": len(eval_pairs),
        "hold_out_test": args.hold_out_test,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "checkpoint_dir": str(best_dir),
        "train_loss": train_result.training_loss,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = args.output_dir / "train_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("Saved checkpoint to %s", best_dir)
    log.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
