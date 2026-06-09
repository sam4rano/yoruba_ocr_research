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

MODEL_LABEL = "trocr_large_printed_finetuned"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned TrOCR on Yorùbá line crops."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("experiments/trocr_large_printed/best"),
        help="Directory with saved TrOCR weights + processor.",
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
    """Load TrOCR checkpoint and evaluate."""
    args = parse_args()
    if args.per_sample_log is None:
        args.per_sample_log = Path("results/tables") / f"{MODEL_LABEL}_{args.split}.jsonl"

    if not args.model_dir.is_dir():
        log.error(
            "Checkpoint not found: %s\nRun: python scripts/21_train_trocr.py",
            args.model_dir,
        )
        sys.exit(1)

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

    processor = TrOCRProcessor.from_pretrained(args.model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    log.info("Running TrOCR inference on %d images (%s) ...", len(pairs), device)
    pred_pairs = run_inference(model, processor, pairs, args.batch_size)

    metrics = aggregate_metrics(pred_pairs)
    log.info(
        "%s — CER: %.4f  WER: %.4f  DER: %.4f  (n=%d)",
        MODEL_LABEL,
        metrics["cer"],
        metrics["wer"],
        metrics["der"],
        metrics["n"],
    )

    ckpt_file = args.model_dir / "pytorch_model.bin"
    if not ckpt_file.exists():
        ckpt_file = args.model_dir / "model.safetensors"
    provenance = {
        "model_kind": "trocr",
        "base_model_id": "microsoft/trocr-large-printed",
        "checkpoint_dir": str(args.model_dir),
        "checkpoint_sha256": _sha256_file(ckpt_file) if ckpt_file.exists() else None,
        "batch_size": args.batch_size,
        "data_dir": str(args.data_dir),
        "n_images": len(pairs),
    }
    save_results(
        metrics,
        model_name=MODEL_LABEL,
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
