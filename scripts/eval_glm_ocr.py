"""
Evaluate GLM-OCR (Hugging Face) on Yorùbá line crops — zero-shot.

Requires ``transformers>=4.46.0`` and ``accelerate``.
Does **not** modify ``data/processed``.

See: https://huggingface.co/zai-org/GLM-OCR

Usage:
    python scripts/eval_glm_ocr.py --split test
    python scripts/eval_glm_ocr.py --split val --max-samples 50
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
        default=False,
        help="Load base model in 4-bit (requires bitsandbytes). Disabled by default; prefer hardware-native float16/bfloat16 for reproducible precision alignment.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Images per generation batch. Failed/OOM batches retry one image at a time.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard a compatible partial checkpoint and restart inference.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Publish metrics with failed samples counted as empty predictions. Default: stop before publishing metrics.",
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


def transcribe_batch(
    image_paths: list[Path],
    model,
    processor,
    device: str,
    prompt: str,
    max_new_tokens: int,
) -> list[str]:
    """Transcribe a batch of line images with one generation call."""
    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).parent))
    from paddle_vl_shared import clean_vl_transcript  # noqa: E402

    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for image in images
    ]

    inputs = processor.apply_chat_template(
        conversations if len(conversations) > 1 else conversations[0],
        tokenize=True,
        add_generation_prompt=True,
        padding=len(conversations) > 1,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    inp_ids = inputs.get("input_ids")
    input_width = inp_ids.shape[-1] if inp_ids is not None else None
    predictions = []
    for generated in outputs:
        if input_width is not None and generated.shape[-1] > input_width:
            generated = generated[input_width:]
        raw = processor.decode(generated, skip_special_tokens=True)
        if "assistant\n" in raw:
            raw = raw.split("assistant\n", 1)[1]
        elif "assistant" in raw:
            raw = raw.split("assistant", 1)[1]
        raw = raw.strip()
        if raw.lower().startswith(prompt.lower()):
            raw = raw[len(prompt):].strip()
        predictions.append(clean_vl_transcript(raw))
    return predictions


def load_model_and_processor(
    model_id: str,
    quantize_4bit: bool,
):
    """Load HF GLM-OCR model and processor (zero-shot, no adapters).

    Precision policy:
      - GPU with bf16 support: bfloat16
      - GPU without bf16:      float16 (e.g. T4)
      - CPU (no CUDA):         float32
      - 4-bit quantization:   only when --quantize-4bit is explicitly passed
    """
    import torch

    try:
        import transformers
        import accelerate
        from transformers import AutoModelForImageTextToText, AutoProcessor
        
        # GLM-OCR requires transformers>=5.0.0. Ensure the version is correct.
        tf_major = int(transformers.__version__.split(".")[0])
        if tf_major < 5:
            raise ImportError(
                f"transformers>=5.0.0 is required for GLM-OCR (found {transformers.__version__})."
            )

        # transformers>=5 requires accelerate>=1.1.0 for device_map="auto" loading.
        acc_major = int(accelerate.__version__.split(".")[0])
        acc_minor = int(accelerate.__version__.split(".")[1])
        if acc_major < 1 or (acc_major == 1 and acc_minor < 1):
            raise ImportError(
                f"accelerate>=1.1.0 is required for device mapping in transformers 5 (found {accelerate.__version__})."
            )
    except ImportError as exc:
        log.error("=" * 80)
        log.error("CRITICAL: transformers>=5.0.0 and accelerate>=1.1.0 are required to recognize and load GLM-OCR.")
        log.error("Please run: pip install -U 'transformers>=5' 'accelerate>=1.1.0' 'huggingface_hub>=1.5.0'")
        log.error("And RESTART the Colab runtime session (Runtime > Restart session).")
        log.error("=" * 80)
        raise ImportError(
            f"Install dependencies: pip install -U 'transformers>=5' 'accelerate>=1.1.0'"
        ) from exc

    sys.path.insert(0, str(Path(__file__).parent))
    from paddle_vl_shared import (  # noqa: E402
        hf_trust_remote_code_model,
        hf_trust_remote_code_processor,
        select_torch_dtype,
    )

    # Flush Paddle/other framework CUDA contexts before loading a PyTorch model
    # to prevent context conflicts that cause torch.cuda.is_available() to return False.
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    _dtype, _dtype_label = select_torch_dtype()
    log.info("Loading GLM-OCR in %s (CUDA=%s)", _dtype_label, torch.cuda.is_available())

    kwargs: dict = {"trust_remote_code": hf_trust_remote_code_model()}
    if quantize_4bit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
        except ImportError as exc:
            raise ImportError("For --quantize-4bit install bitsandbytes") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["device_map"] = "auto"
        log.info("4-bit quantization enabled (overrides %s default).", _dtype_label)
    else:
        kwargs["dtype"] = _dtype
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=hf_trust_remote_code_processor()
        )
    except (KeyError, ValueError) as exc:
        log.error("=" * 80)
        log.error("CRITICAL ERROR: Failed to load GLM-OCR model checkpoints.")
        log.error("This is usually because the active python environment has an outdated transformers")
        log.error("or accelerate version loaded in memory. To fix this:")
        log.error("1. Run the dependencies installation cell.")
        log.error("2. Restart the Colab session (Runtime > Restart session).")
        log.error("=" * 80)
        raise RuntimeError(
            "GLM-OCR architecture loading failed. Please install transformers>=5.0.0 and accelerate>=1.1.0 and restart your kernel."
        ) from exc

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
    from vl_eval_runtime import evaluate_resumable, run_fingerprint, sample_id  # noqa: E402

    if args.per_sample_log is None:
        args.per_sample_log = Path(f"results/tables/{MODEL_LABEL}_{args.split}.jsonl")

    pairs = load_test_pairs(args.data_dir, args.split)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be at least 1")
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise ValueError(f"No readable samples found for split={args.split}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    fingerprint_payload = {
        "model_id": args.model_id,
        "model_name": MODEL_LABEL,
        "split": args.split,
        "prompt_sha256": _sha256_text(args.prompt),
        "max_new_tokens": args.max_new_tokens,
        "quantize_4bit": bool(args.quantize_4bit),
        "samples": [sample_id(path, gt) for path, gt in pairs],
    }
    fingerprint = run_fingerprint(fingerprint_payload)
    partial_path = args.per_sample_log.with_suffix(args.per_sample_log.suffix + ".partial")
    if args.no_resume and partial_path.exists():
        partial_path.unlink()

    model, processor = load_model_and_processor(args.model_id, args.quantize_4bit)
    device = str(next(model.parameters()).device)

    def infer(paths: list[Path]) -> list[str]:
        return transcribe_batch(
            paths,
            model,
            processor,
            device,
            args.prompt,
            args.max_new_tokens,
        )

    results, sample_rows = evaluate_resumable(
        pairs,
        transcribe_batch=infer,
        batch_size=args.batch_size,
        partial_path=partial_path,
        fingerprint=fingerprint,
        resume=not args.no_resume,
        description="Evaluating GLM-OCR",
    )
    failures = [row for row in sample_rows if row["status"] != "ok"]
    if failures and not args.allow_failures:
        examples = "; ".join(
            f"{Path(row['image_path']).name}: {row['error']}" for row in failures[:3]
        )
        raise RuntimeError(
            f"{len(failures)} inference samples failed; metrics were not published. "
            f"Fix the runtime and rerun to resume from {partial_path}. Examples: {examples}"
        )

    metrics = aggregate_metrics(results)
    cer_pct = f"{metrics['cer'] * 100:.2f}%" if metrics["cer"] is not None else "—"
    wer_pct = f"{metrics['wer'] * 100:.2f}%" if metrics["wer"] is not None else "—"
    der_pct = f"{metrics['der'] * 100:.2f}%" if metrics["der"] is not None else "—"
    log.info(
        "%s — CER: %s  WER: %s  DER: %s  (n=%d)",
        MODEL_LABEL,
        cer_pct,
        wer_pct,
        der_pct,
        metrics["n"],
    )
    # Determine actual dtype string for provenance logging
    if args.quantize_4bit:
        _recorded_dtype = "4bit"
    elif torch.cuda.is_available():
        from paddle_vl_shared import select_torch_dtype
        _recorded_dtype = select_torch_dtype()[1]
    else:
        _recorded_dtype = "float32"

    provenance: dict = {
        "model_kind": "glm_ocr",
        "base_model_id": args.model_id,
        "base_model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
        "processor_revision": getattr(processor, "_commit_hash", None),
        "quantize_4bit": bool(args.quantize_4bit),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "prompt": args.prompt,
        "prompt_sha256": _sha256_text(args.prompt),
        "data_dir": str(args.data_dir),
        "n_images": len(pairs),
        "batch_size": args.batch_size,
        "resume_enabled": not args.no_resume,
        "run_fingerprint": fingerprint,
        "failure_count": len(failures),
        "failures_allowed": bool(args.allow_failures),
        "device": device,
        "torch_dtype": _recorded_dtype,
    }
    save_results(
        metrics,
        model_name=MODEL_LABEL,
        split=args.split,
        csv_path=args.results_csv,
        jsonl_path=args.per_sample_log,
        provenance=provenance,
        sample_metadata=sample_rows,
    )
    partial_path.unlink(missing_ok=True)
    log.info("Results appended to %s", args.results_csv)


if __name__ == "__main__":
    main()
