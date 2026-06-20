"""
Evaluate GLM-OCR (Hugging Face) on Yorùbá line crops — zero-shot.

Requires ``transformers>=4.46.0`` and ``accelerate``.
Does **not** modify ``data/processed``.

See: https://huggingface.co/zai-org/GLM-OCR

Usage:
    python scripts/16_baseline_glm_ocr.py --split test
    python scripts/16_baseline_glm_ocr.py --split val --max-samples 50
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_LABEL = "glm_ocr_zero_shot"
DEFAULT_MODEL_ID = "zai-org/GLM-OCR"
DEFAULT_PROMPT = "Text Recognition:"


def _sha256_text(text: str) -> str:
    """Return the hex SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="GLM-OCR zero-shot eval (CER/WER/DER)."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id (default: zai-org/GLM-OCR).",
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
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt tag to trigger OCR mode (default: 'Text Recognition:').",
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
    prompt: str,
    max_new_tokens: int,
) -> str:
    """
    Run a single line image through GLM-OCR and return cleaned text.
    """
    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).parent))
    from paddle_vl_shared import clean_vl_transcript  # noqa: E402

    image = Image.open(img_path).convert("RGB")

    # Cap resolution to prevent OOM on T4 (matches eval spec).
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
    image.thumbnail((800, 800), resample_filter)

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    # Let's handle decoder-only prompt stripping if needed.
    inp_ids = inputs.get("input_ids")
    if inp_ids is not None and outputs.shape[-1] > inp_ids.shape[-1]:
        # If output includes input_ids, slice it
        if torch.equal(outputs[0][:inp_ids.shape[-1]], inp_ids[0]):
            new_tokens = outputs[0][inp_ids.shape[-1]:]
        else:
            new_tokens = outputs[0]
    else:
        new_tokens = outputs[0]

    raw = processor.decode(new_tokens, skip_special_tokens=True)
    
    raw_stripped = raw.strip()
    if raw_stripped.lower().startswith(prompt.lower()):
        raw_stripped = raw_stripped[len(prompt):].strip()

    return clean_vl_transcript(raw_stripped)


def load_model_and_processor(
    model_id: str,
    quantize_4bit: bool,
):
    """Load HF GLM-OCR model and processor (zero-shot, no adapters)."""
    import torch

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Install transformers>=4.46.0: pip install 'transformers>=4.46.0'"
        ) from exc

    sys.path.insert(0, str(Path(__file__).parent))
    from paddle_vl_shared import (  # noqa: E402
        hf_trust_remote_code_model,
        hf_trust_remote_code_processor,
    )

    kwargs: dict = {"trust_remote_code": hf_trust_remote_code_model()}
    if quantize_4bit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
        except ImportError as exc:
            raise ImportError("For --quantize-4bit install bitsandbytes") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        kwargs["device_map"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=hf_trust_remote_code_processor()
    )
    model.eval()
    return model, processor


def main() -> None:
    """Run zero-shot evaluation and append metrics."""
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent))
    import torch
    from evaluate_utils import (
        aggregate_metrics,  # noqa: E402
        load_test_pairs,
        save_results,
    )

    if args.per_sample_log is None:
        args.per_sample_log = Path(f"results/tables/{MODEL_LABEL}_{args.split}.jsonl")

    pairs = load_test_pairs(args.data_dir, args.split)
    if args.max_samples:
        pairs = pairs[: args.max_samples]

    model, processor = load_model_and_processor(args.model_id, args.quantize_4bit)
    device = str(next(model.parameters()).device)

    from tqdm import tqdm
    results: list[tuple[str, str]] = []
    for img_path, gt in tqdm(pairs, desc="Evaluating GLM-OCR", unit="img"):
        try:
            pred = transcribe_one(
                img_path,
                model,
                processor,
                device,
                args.prompt,
                args.max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed on %s: %s", img_path.name, exc)
            pred = ""
        results.append((pred, gt))

    metrics = aggregate_metrics(results)
    log.info(
        "%s — CER: %.4f  WER: %.4f  DER: %.4f  (n=%d)",
        MODEL_LABEL,
        metrics["cer"],
        metrics["wer"],
        metrics["der"],
        metrics["n"],
    )
    provenance: dict = {
        "model_kind": "glm_ocr",
        "base_model_id": args.model_id,
        "quantize_4bit": bool(args.quantize_4bit),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "prompt": args.prompt,
        "prompt_sha256": _sha256_text(args.prompt),
        "data_dir": str(args.data_dir),
        "n_images": len(pairs),
        "device": device,
        "torch_dtype": (
            "bfloat16"
            if (not args.quantize_4bit and torch.cuda.is_available())
            else ("4bit" if args.quantize_4bit else "float32")
        ),
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
