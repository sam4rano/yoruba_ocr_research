"""
Evaluate PaddleOCR-VL-1.6 (Hugging Face) on Yorùbá line crops — zero-shot.

Requires ``transformers>=5`` per the upstream model card.
Does **not** modify ``data/processed``.

See: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6

Usage:
    python scripts/eval_paddleocrvl16.py --split test
    python scripts/eval_paddleocrvl16.py --split val --max-samples 50
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_LABEL = "paddleocrvl16_zero_shot"
DEFAULT_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"


def _sha256_text(text: str) -> str:
    """Return the hex SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL-1.6 zero-shot eval (CER/WER/DER)."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id (default: PaddlePaddle/PaddleOCR-VL-1.6).",
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
        default=False,
        help="Load base model in 4-bit (requires bitsandbytes). Disabled by default; prefer hardware-native float16/bfloat16 for reproducible precision alignment.",
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
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Explicit model display label for results table.",
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
    Run a single line image through PaddleOCR-VL-1.6 and return cleaned text.
    """
    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).parent))
    from paddle_vl_shared import clean_vl_transcript  # noqa: E402

    image = Image.open(img_path).convert("RGB")


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    max_pixels = 768 * 28 * 28
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "images_kwargs": {
                "size": {
                    "shortest_edge": getattr(
                        processor.image_processor, "min_pixels", 28 * 28 * 4
                    ),
                    "longest_edge": max_pixels,
                }
            }
        },
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    inp = inputs["input_ids"]
    new_tokens = output_ids[0][inp.shape[-1]:]
    raw = processor.decode(new_tokens, skip_special_tokens=True)
    return clean_vl_transcript(raw)


def load_model_and_processor(
    model_id: str,
    quantize_4bit: bool,
):
    """Load HF PaddleOCR-VL model and processor (zero-shot, no adapters).

    Precision policy (aligned with SFT training in train_paddleocrvl16_sft.py):
      - GPU with bf16 support: bfloat16
      - GPU without bf16:      float16 (e.g. T4)
      - CPU (no CUDA):         float32
      - 4-bit quantization:   only when --quantize-4bit is explicitly passed
    """
    import torch

    try:
        import transformers
        import accelerate
        from transformers import AutoModel, AutoProcessor

        # PaddleOCR-VL-1.6 requires transformers>=5.0.0. Ensure the version is correct.
        tf_major = int(transformers.__version__.split(".")[0])
        if tf_major < 5:
            raise ImportError(
                f"transformers>=5.0.0 is required for PaddleOCR-VL (found {transformers.__version__})."
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
        log.error("CRITICAL: transformers>=5.0.0 and accelerate>=1.1.0 are required.")
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
    log.info("Loading PaddleOCR-VL-1.6 in %s (CUDA=%s)", _dtype_label, torch.cuda.is_available())

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
        import transformers.modeling_rope_utils
        if "default" not in transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS:
            def compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
                import torch
                config.standardize_rope_params()
                rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
                base = rope_parameters_dict.get("rope_theta", 10000.0)
                partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
                head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
                dim = int(head_dim * partial_rotary_factor)
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
                return inv_freq, 1.0
            transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = compute_default_rope_parameters

        # Monkeypatch PreTrainedModel._init_weights to attach compute_default_rope_parameters dynamically
        from transformers import PreTrainedModel
        orig_init_weights = PreTrainedModel._init_weights
        def patched_init_weights(self, module):
            if "RotaryEmbedding" in module.__class__.__name__ and not hasattr(module, "compute_default_rope_parameters"):
                default_fn = transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS.get("default")
                if default_fn:
                    module.compute_default_rope_parameters = default_fn
            return orig_init_weights(self, module)
        PreTrainedModel._init_weights = patched_init_weights

        import transformers.masking_utils
        if not hasattr(transformers.masking_utils, "_is_patched_causal_mask"):
            orig_create_causal_mask = transformers.masking_utils.create_causal_mask
            def patched_create_causal_mask(*args, **kwargs):
                kwargs.pop("cache_position", None)
                return orig_create_causal_mask(*args, **kwargs)
            transformers.masking_utils.create_causal_mask = patched_create_causal_mask
            transformers.masking_utils._is_patched_causal_mask = True

        model = AutoModel.from_pretrained(model_id, **kwargs)

        # Monkeypatch prepare_inputs_for_generation to handle cache_position=None (transformers 5 compatibility)
        if not hasattr(model.__class__, "_is_patched_prep"):
            orig_prep = model.__class__.prepare_inputs_for_generation
            def patched_prep(self, *args, **kwargs):
                cache_position = kwargs.get("cache_position")
                if cache_position is None and len(args) >= 5:
                    cache_position = args[4]
                if cache_position is None:
                    device = next(self.parameters()).device
                    cache_position = torch.tensor([0], device=device)
                    kwargs["cache_position"] = cache_position
                    if len(args) >= 5:
                        args = list(args)
                        args[4] = cache_position
                        args = tuple(args)
                return orig_prep(self, *args, **kwargs)
            model.__class__.prepare_inputs_for_generation = patched_prep
            model.__class__._is_patched_prep = True

        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=hf_trust_remote_code_processor()
        )
    except (KeyError, ValueError) as exc:
        log.error("=" * 80)
        log.error("CRITICAL ERROR: Failed to load PaddleOCR-VL model checkpoints.")
        log.error("This is usually because the active python environment has an outdated transformers")
        log.error("or accelerate version loaded in memory. To fix this:")
        log.error("1. Run the dependencies installation cell.")
        log.error("2. Restart the Colab session (Runtime > Restart session).")
        log.error("=" * 80)
        raise RuntimeError(
            "PaddleOCR-VL architecture loading failed. Please install transformers>=5.0.0 and accelerate>=1.1.0 and restart your kernel."
        ) from exc

    model.eval()
    return model, processor


def main() -> None:
    """Run evaluation and append metrics."""
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent))
    import torch
    from evaluate_utils import (
        aggregate_metrics,  # noqa: E402
        load_test_pairs,
        save_results,
    )
    from paddle_vl_shared import USER_TEXT_OCR_YORUBA  # noqa: E402

    model_name = args.model_name
    if not model_name:
        if "finetuned" in args.model_id or "experiments" in args.model_id:
            model_name = "paddleocrvl16_sft"
        else:
            model_name = "paddleocrvl16_zero_shot"

    if args.per_sample_log is None:
        args.per_sample_log = Path(f"results/tables/{model_name}_{args.split}.jsonl")

    pairs = load_test_pairs(args.data_dir, args.split)
    if args.max_samples:
        pairs = pairs[: args.max_samples]

    model, processor = load_model_and_processor(args.model_id, args.quantize_4bit)
    device = str(next(model.parameters()).device)

    from tqdm import tqdm
    results: list[tuple[str, str]] = []
    for img_path, gt in tqdm(pairs, desc="Evaluating PaddleOCR-VL-1.6", unit="img"):
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

    metrics = aggregate_metrics(results)
    cer_pct = f"{metrics['cer'] * 100:.2f}%" if metrics["cer"] is not None else "—"
    wer_pct = f"{metrics['wer'] * 100:.2f}%" if metrics["wer"] is not None else "—"
    der_pct = f"{metrics['der'] * 100:.2f}%" if metrics["der"] is not None else "—"
    log.info(
        "%s — CER: %s  WER: %s  DER: %s  (n=%d)",
        model_name,
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
        "model_kind": "paddleocr_vl",
        "base_model_id": args.model_id,
        "quantize_4bit": bool(args.quantize_4bit),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "prompt": USER_TEXT_OCR_YORUBA,
        "prompt_sha256": _sha256_text(USER_TEXT_OCR_YORUBA),
        "data_dir": str(args.data_dir),
        "n_images": len(pairs),
        "device": device,
        "torch_dtype": _recorded_dtype,
    }
    save_results(
        metrics,
        model_name=model_name,
        split=args.split,
        csv_path=args.results_csv,
        jsonl_path=args.per_sample_log,
        provenance=provenance,
    )
    log.info("Results appended to %s", args.results_csv)


if __name__ == "__main__":
    main()
