"""
Evaluate PaddleOCR-VL-1.5 (Hugging Face) on Yorùbá line crops — zero-shot or LoRA adapter.

Requires ``transformers>=5`` per the upstream model card. Does **not** modify ``data/processed``.

See: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5

Usage:
    python scripts/15_baseline_paddleocr_vl15.py --split test
    python scripts/15_baseline_paddleocr_vl15.py --adapter-path experiments/paddleocr_vl15_lora/adapter
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_ZERO_SHOT = "paddleocr_vl15_zero_shot"
MODEL_FINETUNED = "paddleocr_vl15_lora_finetuned"


def _hash_adapter(adapter_path: Path | None) -> dict:
    """Return ``{file: sha256}`` for the key files in a PEFT adapter dir."""
    if adapter_path is None or not adapter_path.is_dir():
        return {}
    tracked = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
    out: dict[str, str] = {}
    for name in tracked:
        p = adapter_path / name
        if not p.exists():
            continue
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        out[name] = h.hexdigest()
    return out


def _sha256_text(text: str) -> str:
    """Return the hex SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL-1.5 baseline/finetuned eval (CER/WER/DER)."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="PaddlePaddle/PaddleOCR-VL-1.5",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional PEFT adapter directory (from 16_train_paddleocr_vl_lora.py).",
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
        help="Cap number of images (debug).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Generation length cap.",
    )
    parser.add_argument(
        "--quantize-4bit",
        action="store_true",
        help="Load base model in 4-bit (requires bitsandbytes).",
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
    )
    parser.add_argument(
        "--per-sample-log",
        type=Path,
        default=None,
        help="JSONL path (default: results/tables/<model>_<split>.jsonl).",
    )
    return parser.parse_args()


def transcribe_one(
    img_path: Path,
    model,
    processor,
    device: str,
    user_prompt: str,
    max_new_tokens: int,
) -> str:
    """
    Run a single line image through PaddleOCR-VL-1.5 and return cleaned text.
    """
    from PIL import Image
    import torch

    sys.path.insert(0, str(Path(__file__).parent))
    from paddle_vl_shared import clean_vl_transcript  # noqa: E402

    image = Image.open(img_path).convert("RGB")

    # Cap resolution to match training (16_train_paddleocr_vl_lora.py).
    # Avoids train/eval distribution shift in the vision encoder.
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
    image.thumbnail((800, 800), resample_filter)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    # Must match training max_pixels (768 * 28 * 28 = 602,112)
    max_pixels = 768 * 28 * 28
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "max_pixels": max_pixels,
            "size": {
                "shortest_edge": getattr(
                    processor.image_processor, "min_pixels", 28 * 28 * 4
                ),
                "longest_edge": max_pixels,
            }
        },
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    inp = inputs["input_ids"]
    new_tokens = output_ids[0][inp.shape[-1] :]
    raw = processor.decode(new_tokens, skip_special_tokens=True)
    return clean_vl_transcript(raw)


def load_model_and_processor(
    model_id: str,
    adapter_path: Path | None,
    quantize_4bit: bool,
):
    """Load HF model, optionally merge PEFT adapter."""
    import torch

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Install transformers>=5 for PaddleOCR-VL-1.5: pip install 'transformers>=5'"
        ) from exc

    # transformers>=5 ships native PaddleOCR-VL; trust_remote_code pulls hub config
    # that mismatches modeling (text_config vs get_text_config). See PaddleOCR#17666.
    kwargs: dict = {"trust_remote_code": False}
    if quantize_4bit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
        except ImportError as exc:
            raise ImportError("For --quantize-4bit install bitsandbytes") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs["device_map"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)

    if adapter_path is not None:
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        try:
            from peft import PeftModel  # type: ignore
        except ImportError as exc:
            raise ImportError("Install peft for adapter eval: pip install peft") from exc
        log.info("Loading PEFT adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model, processor


def main() -> None:
    """Run evaluation and append metrics."""
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_utils import aggregate_metrics, load_test_pairs, save_results  # noqa: E402
    from paddle_vl_shared import USER_TEXT_OCR_YORUBA  # noqa: E402

    import torch

    model_label = MODEL_FINETUNED if args.adapter_path else MODEL_ZERO_SHOT
    if args.per_sample_log is None:
        args.per_sample_log = Path(f"results/tables/{model_label}_{args.split}.jsonl")

    pairs = load_test_pairs(args.data_dir, args.split)
    if args.max_samples:
        pairs = pairs[: args.max_samples]

    model, processor = load_model_and_processor(
        args.model_id, args.adapter_path, args.quantize_4bit
    )
    device = str(next(model.parameters()).device)

    results: list[tuple[str, str]] = []
    for i, (img_path, gt) in enumerate(pairs, 1):
        try:
            pred = transcribe_one(
                img_path,
                model,
                processor,
                device,
                USER_TEXT_OCR_YORUBA,
                args.max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed on %s: %s", img_path.name, exc)
            pred = ""
        results.append((pred, gt))
        if i % args.batch_size == 0:
            log.info("  Processed %d / %d ...", i, len(pairs))

    metrics = aggregate_metrics(results)
    log.info(
        "%s — CER: %.4f  WER: %.4f  DER: %.4f  (n=%d)",
        model_label,
        metrics["cer"],
        metrics["wer"],
        metrics["der"],
        metrics["n"],
    )
    provenance: dict = {
        "model_kind": "paddleocr_vl",
        "base_model_id": args.model_id,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "adapter_sha256": _hash_adapter(args.adapter_path),
        "quantize_4bit": bool(args.quantize_4bit),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "prompt": USER_TEXT_OCR_YORUBA,
        "prompt_sha256": _sha256_text(USER_TEXT_OCR_YORUBA),
        "data_dir": str(args.data_dir),
        "n_images": len(pairs),
        "device": device,
        "torch_dtype": (
            "bfloat16" if (not args.quantize_4bit and torch.cuda.is_available())
            else ("4bit" if args.quantize_4bit else "float32")
        ),
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


if __name__ == "__main__":
    main()
