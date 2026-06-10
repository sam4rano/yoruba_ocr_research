"""
Evaluate a fine-tuned Surya Foundation checkpoint on Yorùbá line crops.

Uses the **surya 0.15.x** Foundation + Recognition stack (not Surya v2 / vLLM).
Each line crop is recognised with ``TaskNames.ocr_without_boxes`` and a
full-image bounding box — matching the benchmark's pre-segmented line setting.

Requires the same surya version used for training (``27_train_surya_finetune.py``).

Usage:
    python scripts/28_evaluate_surya_finetuned.py --split test
    python scripts/28_evaluate_surya_finetuned.py \\
        --checkpoint experiments/surya_finetune/checkpoint-500 --split test
"""

from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_LABEL = "surya_finetuned"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Surya Foundation OCR on line crops."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Fine-tuned checkpoint dir (default: latest under experiments/surya_finetune).",
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
        default=8,
        help="Recognition batch size.",
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


def resolve_checkpoint(path: Path | None) -> Path:
    """Pick explicit checkpoint or the newest ``checkpoint-*`` under output dir."""
    if path is not None and path.is_dir():
        return path
    root = Path("experiments/surya_finetune")
    if not root.is_dir():
        raise FileNotFoundError(
            "No Surya checkpoint found. Run scripts/27_train_surya_finetune.py first."
        )
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        if (root / "pytorch_model.bin").exists() or any(root.glob("*.safetensors")):
            return root
        raise FileNotFoundError(f"No checkpoint-* under {root}")
    return candidates[0]


def full_image_bboxes(images) -> list[list[list[int]]]:
    """One axis-aligned bbox covering each PIL image."""
    bboxes: list[list[list[int]]] = []
    for im in images:
        w, h = im.size
        bboxes.append([[0, 0], [w, 0], [w, h], [0, h]])
    return bboxes


def ocr_result_to_text(result) -> str:
    """Join ``OCRResult.text_lines`` into one NFC string."""
    parts = [line.text for line in result.text_lines if line.text]
    return unicodedata.normalize("NFC", " ".join(parts).strip())


def run_recognition(
    predictor,
    image_paths: list[Path],
    batch_size: int,
) -> list[str]:
    """Batch recognition on line crops."""
    from PIL import Image  # type: ignore
    from surya.common.surya.schema import TaskNames  # type: ignore

    texts: list[str] = []
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in chunk]
        try:
            preds = predictor(
                images,
                task_names=[TaskNames.ocr_without_boxes] * len(images),
                bboxes=full_image_bboxes(images),
                recognition_batch_size=batch_size,
                math_mode=False,
            )
            texts.extend(ocr_result_to_text(p) for p in preds)
        finally:
            for im in images:
                im.close()
        if start and start % (batch_size * 10) == 0:
            log.info("Inferred %d / %d", start, len(image_paths))
    return texts


def main() -> None:
    """Load checkpoint and evaluate."""
    args = parse_args()
    if args.per_sample_log is None:
        args.per_sample_log = Path("results/tables") / f"{MODEL_LABEL}_{args.split}.jsonl"

    checkpoint = resolve_checkpoint(args.checkpoint)
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_utils import aggregate_metrics, load_test_pairs, save_results  # noqa: E402

    try:
        from surya.foundation import FoundationPredictor  # type: ignore
        from surya.recognition import RecognitionPredictor  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Surya Foundation stack not installed.\n"
            "Run: python scripts/27_train_surya_finetune.py (installs surya v0.15.3)"
        ) from exc

    pairs = load_test_pairs(args.data_dir, args.split)
    if args.max_samples:
        pairs = pairs[: args.max_samples]

    foundation = FoundationPredictor(checkpoint=str(checkpoint))
    predictor = RecognitionPredictor(foundation)

    log.info(
        "Evaluating %s on %d images (%s) checkpoint=%s",
        MODEL_LABEL,
        len(pairs),
        args.split,
        checkpoint,
    )
    preds = run_recognition(predictor, [p for p, _ in pairs], args.batch_size)
    pred_pairs = list(zip(preds, [gt for _, gt in pairs], strict=True))

    metrics = aggregate_metrics(pred_pairs)
    log.info(
        "%s — CER: %.4f  WER: %.4f  DER: %.4f  (n=%d)",
        MODEL_LABEL,
        metrics["cer"],
        metrics["wer"],
        metrics["der"],
        metrics["n"],
    )

    provenance = {
        "model_kind": "surya_foundation_finetuned",
        "checkpoint_dir": str(checkpoint),
        "surya_stack": "0.15.x Foundation",
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


if __name__ == "__main__":
    main()
