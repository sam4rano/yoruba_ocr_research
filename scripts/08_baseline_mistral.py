"""
Mistral OCR zero-shot baseline for the Yorùbá OCR benchmark.

Sends each line-crop image to the Mistral `mistral-ocr-latest` endpoint
with a structured prompt requesting verbatim transcription of the Yorùbá text,
including all diacritics.

API key must be set in the environment:
    export MISTRAL_API_KEY=your_key_here

Requires:
    pip install mistralai Pillow

Cost note: Each image is one API call. At ~500 test images, budget ~0.10–0.50 USD
depending on the current mistral-ocr pricing tier.

Usage:
    python scripts/08_baseline_mistral.py
    python scripts/08_baseline_mistral.py \
        --data-dir data/processed \
        --split test \
        --max-samples 50      # limit during development
        --results-csv results/tables/metrics.csv
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
import time
import unicodedata
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "mistral_ocr_zero_shot"
MISTRAL_MODEL = "mistral-ocr-latest"

# Prompt that instructs Mistral to transcribe verbatim without corrections
SYSTEM_PROMPT = (
    "You are a precise OCR transcription engine. "
    "Transcribe the Yorùbá text in the image EXACTLY as it appears, "
    "preserving all diacritics (tone marks and subdots). "
    "Output ONLY the transcribed text — no explanations, no punctuation changes."
)


def encode_image_b64(img_path: Path) -> str:
    """Return a base64-encoded string of the image file."""
    with img_path.open("rb") as fh:
        return base64.standard_b64encode(fh.read()).decode("utf-8")


def call_mistral_ocr(
    client,  # mistralai.Mistral
    img_path: Path,
    retry_on_rate_limit: bool = True,
) -> str:
    """
    Send one line-crop to the Mistral OCR endpoint and return the transcription.

    Retries once after a 60-second backoff on rate-limit (429) errors.
    Returns an empty string on unrecoverable failure.
    """
    b64 = encode_image_b64(img_path)

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }

    try:
        response = client.chat.complete(**payload)
        text = response.choices[0].message.content.strip()
        return unicodedata.normalize("NFC", text)
    except Exception as exc:  # noqa: BLE001
        err_str = str(exc).lower()
        if "429" in err_str and retry_on_rate_limit:
            log.warning("Rate limited — waiting 60 s before retry.")
            time.sleep(60)
            return call_mistral_ocr(client, img_path, retry_on_rate_limit=False)
        log.debug("Mistral call failed for %s: %s", img_path.name, exc)
        return ""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Mistral OCR zero-shot baseline for Yorùbá OCR."
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
        help="Limit the number of images evaluated (useful for cost control).",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/tables/metrics.csv"),
        help="Shared results table — appended on each run.",
    )
    parser.add_argument(
        "--per-sample-log",
        type=Path,
        default=Path(f"results/tables/{MODEL_NAME}_test.jsonl"),
        help="Per-sample JSONL prediction log.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Mistral OCR zero-shot evaluation."""
    args = parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        log.error(
            "MISTRAL_API_KEY environment variable not set.\n"
            "Export it with: export MISTRAL_API_KEY=your_key_here"
        )
        sys.exit(1)

    try:
        from mistralai import Mistral  # type: ignore
    except ImportError:
        raise ImportError("Run: pip install mistralai")

    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_utils import load_test_pairs, aggregate_metrics, save_results  # type: ignore  # noqa: E402

    pairs = load_test_pairs(args.data_dir, args.split)
    if args.max_samples:
        pairs = pairs[: args.max_samples]
        log.info("Limited to %d samples for cost control.", args.max_samples)
    log.info("Evaluating %d samples with Mistral OCR ...", len(pairs))

    client = Mistral(api_key=api_key)

    pred_pairs: list[tuple[str, str]] = []
    for i, (img_path, gt) in enumerate(pairs, 1):
        pred = call_mistral_ocr(client, img_path)
        pred_pairs.append((pred, gt))
        if i % 20 == 0:
            log.info("  Processed %d / %d images ...", i, len(pairs))

    metrics = aggregate_metrics(pred_pairs)
    log.info(
        "%s — CER: %.4f  WER: %.4f  DER: %.4f  (n=%d)",
        MODEL_NAME,
        metrics["cer"],
        metrics["wer"],
        metrics["der"],
        metrics["n"],
    )
    save_results(
        metrics,
        model_name=MODEL_NAME,
        split=args.split,
        csv_path=args.results_csv,
        jsonl_path=args.per_sample_log,
    )
    log.info("Results appended to %s", args.results_csv)


if __name__ == "__main__":
    main()
