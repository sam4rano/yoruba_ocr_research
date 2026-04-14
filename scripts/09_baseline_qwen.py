"""
Qwen 2.5 VL zero-shot baseline for the Yorùbá OCR benchmark.

Loads Qwen/Qwen2.5-VL-7B-Instruct from Hugging Face and prompts it to
transcribe each line-crop image verbatim, preserving all Yorùbá diacritics.

This is a zero-shot multimodal LLM evaluation — no fine-tuning is performed.
The model demonstrates whether large vision-language models trained on
multilingual web data can handle Yorùbá tone marks without task-specific
supervision.

Hardware: requires ~16 GB VRAM for 7B in bfloat16, or use 4-bit quantisation
(--quantize) on smaller GPUs.

Install:
    pip install transformers accelerate qwen-vl-utils torch
    # For 4-bit:
    pip install bitsandbytes

Usage:
    python scripts/09_baseline_qwen.py
    python scripts/09_baseline_qwen.py \
        --model-id Qwen/Qwen2.5-VL-7B-Instruct \
        --data-dir data/processed \
        --split test \
        --quantize \
        --batch-size 4
"""

from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_LABEL = "qwen25_vl_zero_shot"

# Prompt that requests verbatim Yorùbá transcription
USER_PROMPT = (
    "Transcribe the text in this image exactly as it appears. "
    "The text is in Yorùbá. Preserve all diacritics (tone marks such as "
    "à, á, â and subdots such as ẹ, ọ, ṣ) precisely. "
    "Output ONLY the transcribed text — no explanation."
)


def load_model_and_processor(model_id: str, quantize: bool):
    """
    Load the Qwen2.5-VL model and processor.

    With quantize=True, loads in 4-bit using BitsAndBytes to fit
    the 7B model on 8–12 GB VRAM.
    """
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig  # type: ignore
    except ImportError:
        raise ImportError(
            "Run: pip install transformers accelerate qwen-vl-utils torch\n"
            "For quantisation: pip install bitsandbytes"
        )

    log.info("Loading %s (quantize=%s) ...", model_id, quantize)

    kwargs: dict = {
        "trust_remote_code": True,
    }
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, processor


def transcribe_image(
    img_path: Path,
    model,
    processor,
    max_new_tokens: int = 128,
) -> str:
    """
    Run Qwen2.5-VL inference on a single line-crop image.

    Returns the model's text prediction, NFC-normalised.
    """
    try:
        import torch
        from qwen_vl_utils import process_vision_info  # type: ignore
    except ImportError:
        raise ImportError("Run: pip install qwen-vl-utils")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": USER_PROMPT},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Trim prompt tokens from the output
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs["input_ids"], output_ids)
    ]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return unicodedata.normalize("NFC", decoded[0].strip())


def evaluate_in_batches(
    pairs: list[tuple[Path, str]],
    model,
    processor,
    batch_size: int,
) -> list[tuple[str, str]]:
    """
    Run inference over all image pairs, logging progress every batch.

    Single-image inference is used (batch_size controls logging frequency
    only — Qwen2.5-VL processes one image at a time through process_vision_info).
    """
    results = []
    failed = 0
    for i, (img_path, gt) in enumerate(pairs, 1):
        try:
            pred = transcribe_image(img_path, model, processor)
        except Exception as exc:  # noqa: BLE001
            log.debug("Qwen inference failed on %s: %s", img_path.name, exc)
            pred = ""
            failed += 1
        results.append((pred, gt))
        if i % batch_size == 0:
            log.info("  Processed %d / %d ...", i, len(pairs))

    if failed:
        log.warning("Failed on %d images.", failed)
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen 2.5 VL zero-shot baseline for Yorùbá OCR."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Hugging Face model ID.",
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
        help="Cap the number of images (useful for quick sanity checks).",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Load model in 4-bit precision (requires bitsandbytes).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Progress logging interval.",
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
        default=Path(f"results/tables/{MODEL_LABEL}_test.jsonl"),
        help="Per-sample JSONL prediction log.",
    )
    return parser.parse_args()


def main() -> None:
    """Load model and run zero-shot evaluation."""
    args = parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_utils import load_test_pairs, aggregate_metrics, save_results  # type: ignore  # noqa: E402

    pairs = load_test_pairs(args.data_dir, args.split)
    if args.max_samples:
        pairs = pairs[: args.max_samples]
        log.info("Limited to %d samples.", args.max_samples)

    model, processor = load_model_and_processor(args.model_id, args.quantize)

    log.info("Running zero-shot inference on %d images ...", len(pairs))
    pred_pairs = evaluate_in_batches(pairs, model, processor, args.batch_size)

    metrics = aggregate_metrics(pred_pairs)
    log.info(
        "%s — CER: %.4f  WER: %.4f  DER: %.4f  (n=%d)",
        MODEL_LABEL,
        metrics["cer"],
        metrics["wer"],
        metrics["der"],
        metrics["n"],
    )
    save_results(
        metrics,
        model_name=MODEL_LABEL,
        split=args.split,
        csv_path=args.results_csv,
        jsonl_path=args.per_sample_log,
    )
    log.info("Results appended to %s", args.results_csv)


if __name__ == "__main__":
    main()
