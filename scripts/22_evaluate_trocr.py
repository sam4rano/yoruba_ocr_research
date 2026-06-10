"""
Evaluate a fine-tuned TrOCR checkpoint on a labelled split (CER/WER/DER).

Loads ``experiments/trocr_large_printed/best`` by default (output of
``21_train_trocr.py``).

Usage:
    python scripts/22_evaluate_trocr.py --split test
    python scripts/22_evaluate_trocr.py \
        --model-dir experiments/trocr_large_printed/best \
        --split test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_FINETUNED_DIR = Path("experiments/trocr_large_printed/best")
DEFAULT_PRETRAINED_ID = "microsoft/trocr-large-printed"
MODEL_LABEL_FINETUNED = "trocr_large_printed_finetuned"
MODEL_LABEL_ZERO_SHOT = "trocr_large_printed_zero_shot"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate TrOCR (fine-tuned checkpoint or HF pretrained) on line crops."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory with saved TrOCR weights + processor (fine-tuned eval).",
    )
    parser.add_argument(
        "--pretrained-model-id",
        type=str,
        default=None,
        help="Hugging Face model id for zero-shot eval (e.g. microsoft/trocr-large-printed).",
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default=None,
        help="Row label in metrics.csv (default: finetuned or zero-shot).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Consolidated dataset root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap images (smoke test).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/tables/metrics.csv"),
        help="Shared results table.",
    )
    parser.add_argument(
        "--per-sample-log",
        type=Path,
        default=None,
        help="JSONL log (default: results/tables/{model}_{split}.jsonl).",
    )
    return parser.parse_args()


def decode_predictions(processor, generated_ids) -> list[str]:
    """Decode model output token ids to strings."""
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [t.strip() for t in texts]


def run_inference(
    model,
    processor,
    pairs: list[tuple[Path, str]],
    batch_size: int,
) -> list[tuple[str, str]]:
    """Batch-generate transcriptions for image/GT pairs."""
    import torch
    from PIL import Image  # type: ignore

    device = model.device
    results: list[tuple[str, str]] = []

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        images = []
        for img_path, _ in chunk:
            images.append(Image.open(img_path).convert("RGB"))
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            generated = model.generate(pixel_values, max_length=128)
        preds = decode_predictions(processor, generated)
        for pred, (_, gt) in zip(preds, chunk, strict=True):
            results.append((pred, gt))
        for img in images:
            img.close()
        if start and start % (batch_size * 10) == 0:
            log.info("Inferred %d / %d", start, len(pairs))

    return results


def main() -> None:
    """Load TrOCR checkpoint or pretrained weights and evaluate."""
    args = parse_args()
    zero_shot = bool(args.pretrained_model_id)
    model_label = args.model_label or (
        MODEL_LABEL_ZERO_SHOT if zero_shot else MODEL_LABEL_FINETUNED
    )
    if args.per_sample_log is None:
        args.per_sample_log = Path("results/tables") / f"{model_label}_{args.split}.jsonl"

    model_dir = args.model_dir or DEFAULT_FINETUNED_DIR
    if zero_shot:
        load_path = args.pretrained_model_id
    else:
        if not model_dir.is_dir():
            log.error(
                "Checkpoint not found: %s\nRun: python scripts/21_train_trocr.py",
                model_dir,
            )
            sys.exit(1)
        load_path = str(model_dir)

    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_utils import aggregate_metrics, load_test_pairs, save_results  # noqa: E402

    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Run: pip install transformers torch torchvision"
        ) from exc

    pairs = load_test_pairs(args.data_dir, args.split)
    if args.max_samples:
        pairs = pairs[: args.max_samples]

    processor = TrOCRProcessor.from_pretrained(load_path)
    model = VisionEncoderDecoderModel.from_pretrained(load_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    log.info("Running TrOCR inference on %d images (%s) ...", len(pairs), device)
    pred_pairs = run_inference(model, processor, pairs, args.batch_size)

    metrics = aggregate_metrics(pred_pairs)
    log.info(
        "%s — CER: %.4f  WER: %.4f  DER: %.4f  (n=%d)",
        model_label,
        metrics["cer"],
        metrics["wer"],
        metrics["der"],
        metrics["n"],
    )

    ckpt_file = None
    if not zero_shot:
        ckpt_file = model_dir / "pytorch_model.bin"
        if not ckpt_file.exists():
            ckpt_file = model_dir / "model.safetensors"
    provenance = {
        "model_kind": "trocr",
        "base_model_id": DEFAULT_PRETRAINED_ID,
        "checkpoint_dir": None if zero_shot else str(model_dir),
        "pretrained_model_id": args.pretrained_model_id if zero_shot else None,
        "checkpoint_sha256": _sha256_file(ckpt_file) if ckpt_file and ckpt_file.exists() else None,
        "batch_size": args.batch_size,
        "data_dir": str(args.data_dir),
        "n_images": len(pairs),
    }
    save_results(
        metrics,
        model_name=model_label,
        split=args.split,
        csv_path=args.results_csv,
        jsonl_path=args.per_sample_log,
        provenance=provenance,
    )
    log.info("Results appended to %s", args.results_csv)


def _sha256_file(path: Path) -> str | None:
    """Return hex SHA-256 of a file."""
    try:
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


if __name__ == "__main__":
    main()
