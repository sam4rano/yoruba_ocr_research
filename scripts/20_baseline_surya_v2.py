"""
Surya OCR v2 zero-shot baseline on Yorùbá line crops (recognition only).

Each PNG line crop is passed directly to ``RecognitionPredictor`` with no layout
or text-detection stage — equivalent to treating the crop as a single-page OCR
target. This matches the benchmark setting where ground-truth boxes are already
provided.

Requires ``surya-ocr>=0.20`` and an inference backend (``llama.cpp`` on CPU /
Apple Silicon, or ``vllm`` + GPU). See https://github.com/datalab-to/surya .

Usage:
    pip install surya-ocr
    brew install llama.cpp   # macOS CPU/MPS backend
    python scripts/20_baseline_surya_v2.py --split test --max-samples 5
    python scripts/20_baseline_surya_v2.py --split test
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import unicodedata
from html import unescape
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_LABEL = "surya_v2_zero_shot"
_TAG_RE = re.compile(r"<[^>]+>")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Surya v2 zero-shot recognition on Yorùbá line crops."
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
        help="Dataset split to evaluate.",
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
        default=1,
        help="Images per RecognitionPredictor call.",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Leave Surya inference server running after exit.",
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


def html_blocks_to_text(blocks) -> str:
    """
    Concatenate plain text from Surya v2 ``PageOCRResult.blocks`` HTML fields.

    Blocks may expose ``html`` as an attribute or dict key.
    """
    parts: list[str] = []
    for block in blocks or []:
        if isinstance(block, dict):
            html = block.get("html") or ""
        else:
            html = getattr(block, "html", "") or ""
        if not html:
            continue
        plain = unescape(_TAG_RE.sub("", html))
        plain = " ".join(plain.split())
        if plain:
            parts.append(plain)
    return " ".join(parts).strip()


def extract_page_text(page_result) -> str:
    """Extract NFC-normalised transcription from one Surya page result."""
    blocks = getattr(page_result, "blocks", None)
    if blocks is None and isinstance(page_result, dict):
        blocks = page_result.get("blocks")
    text = html_blocks_to_text(blocks)
    return unicodedata.normalize("NFC", text)


def load_surya_predictor(keep_server: bool):
    """Initialise Surya v2 inference manager and recognition predictor."""
    try:
        from surya.inference import SuryaInferenceManager  # type: ignore
        from surya.recognition import RecognitionPredictor  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install Surya v2: pip install surya-ocr\n"
            "Then provide an inference backend (llama.cpp or vllm). "
            "See https://github.com/datalab-to/surya"
        ) from exc

    manager = SuryaInferenceManager()
    if keep_server:
        import os

        os.environ["SURYA_INFERENCE_KEEP_ALIVE"] = "true"
    predictor = RecognitionPredictor(manager)
    return manager, predictor


def run_recognition_batch(predictor, image_paths: list[Path]) -> list[str]:
    """
    Run recognition on a batch of line-crop images (no detection/layout).

    Returns one NFC string per image, in input order.
    """
    from PIL import Image  # type: ignore

    images = [Image.open(p).convert("RGB") for p in image_paths]
    try:
        predictions = predictor(images)
    finally:
        for img in images:
            img.close()

    texts: list[str] = []
    for page in predictions:
        texts.append(extract_page_text(page))
    return texts


def main() -> None:
    """Run Surya v2 zero-shot evaluation."""
    args = parse_args()
    if args.per_sample_log is None:
        args.per_sample_log = Path("results/tables") / f"{MODEL_LABEL}_{args.split}.jsonl"

    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_utils import aggregate_metrics, load_test_pairs, save_results  # noqa: E402

    pairs = load_test_pairs(args.data_dir, args.split)
    if args.max_samples:
        pairs = pairs[: args.max_samples]
        log.info("Limited to %d samples.", args.max_samples)

    manager, predictor = load_surya_predictor(args.keep_server)

    pred_pairs: list[tuple[str, str]] = []
    failed = 0
    batch_size = max(1, args.batch_size)
    try:
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            paths = [img for img, _ in chunk]
            gts = [gt for _, gt in chunk]
            try:
                preds = run_recognition_batch(predictor, paths)
            except Exception as exc:  # noqa: BLE001
                log.warning("Surya batch failed at %d: %s", start, exc)
                preds = [""] * len(chunk)
                failed += len(chunk)
            for pred, gt in zip(preds, gts, strict=True):
                pred_pairs.append((pred, gt))
            if (start // batch_size) % 25 == 0 and start:
                log.info("Processed %d / %d", start, len(pairs))
    finally:
        shutdown = getattr(manager, "shutdown", None)
        if callable(shutdown) and not args.keep_server:
            shutdown()

    if failed:
        log.warning("Surya failed on %d images.", failed)

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
        "model_kind": "surya_v2",
        "mode": "recognition_only_line_crops",
        "batch_size": batch_size,
        "keep_server": args.keep_server,
        "data_dir": str(args.data_dir),
        "n_images": len(pairs),
        "note_sha256": hashlib.sha256(
            b"surya_v2 RecognitionPredictor on line crops; no detection/layout"
        ).hexdigest(),
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
