"""
Fine-tuning for PaddleOCR-VL-1.6 on exported JSONL (see ``export_paddleocrvl16_sft.py``).

**Training objective:** causal LM loss with **assistant tokens only** (standard SFT):
prompt positions (vision + user text + generation header) are masked with ``-100`` in
``labels``, matching common HF/TRL practice. Use ``--full-sequence-loss`` only for
debugging or ablations.

Optional **gradient accumulation** (``--gradient-accumulation-steps``) reduces optimizer
frequency and can improve stability; micro-batch size remains one image (typical for VL).

Outputs a fine-tuned model under ``--output-dir``; evaluate with
``eval_paddleocrvl16.py --model-id <that dir>``.

Usage:
    python scripts/export_paddleocrvl16_sft.py
    python scripts/train_paddleocrvl16_sft.py --epochs 5 --max-samples 500
    python scripts/train_paddleocrvl16_sft.py --gradient-accumulation-steps 16
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune PaddleOCR-VL-1.6 on paddleocrvl16_sft JSONL export."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("data/paddleocrvl16_sft"),
        help="Directory containing train.jsonl from script 14.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="PaddlePaddle/PaddleOCR-VL-1.6",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/paddleocrvl16_sft"),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--train-scope",
        choices=["lm_head", "non_vision", "all"],
        default="lm_head",
        help=(
            "Which parameters to update. 'lm_head' is safer on small Colab "
            "runs; 'non_vision' matches the earlier full language-side SFT; "
            "'all' also updates the vision tower and is not recommended on T4."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap training samples (sanity / debug).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Optimizer step every N forward passes (default 16).",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=512 * 28 * 28,
        help=(
            "Image processor longest_edge pixel cap. Lower values reduce VRAM "
            "during SFT. Default 401408 (512*28*28) is safer on Colab T4."
        ),
    )
    parser.add_argument(
        "--empty-cache-steps",
        type=int,
        default=25,
        help="Call gc.collect() + torch.cuda.empty_cache() every N micro-steps.",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=64,
        help="Evaluate this many validation samples after each epoch (0 disables).",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=256,
        help="Generation cap for epoch validation.",
    )
    parser.add_argument(
        "--full-sequence-loss",
        action="store_true",
        help="Train on full sequence (no label masking). Not recommended for SFT.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from model weights + training_state.json under --output-dir.",
    )
    return parser.parse_args()


def load_train_samples(export_dir: Path, max_samples: int | None) -> list[dict]:
    """Load records from ``train.jsonl``."""
    path = export_dir / "train.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/export_paddleocrvl16_sft.py"
        )
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No samples in {path}")
    return rows


def load_jsonl_samples(path: Path, max_samples: int | None = None) -> list[dict]:
    """Load JSONL records with an optional sample cap."""
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def is_vision_param(name: str) -> bool:
    """Heuristic for PaddleOCR-VL visual-tower parameter names."""
    lname = name.lower()
    return any(token in lname for token in ("visual", "vision", "image", "vit"))


def is_lm_head_param(name: str) -> bool:
    """Heuristic for output-head parameters across HF remote-code variants."""
    lname = name.lower()
    return any(
        token in lname
        for token in (
            "lm_head",
            "embed_out",
            "output_projection",
            "output_layer",
            "language_model.output",
            "model.output",
        )
    )


def apply_train_scope(model, train_scope: str) -> str:
    """Set requires_grad according to train scope and return effective scope."""
    if train_scope == "lm_head":
        output_param_ids: set[int] = set()
        if hasattr(model, "get_output_embeddings"):
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None:
                output_param_ids = {id(param) for param in output_embeddings.parameters()}

        matched = 0
        for name, param in model.named_parameters():
            trainable = id(param) in output_param_ids or is_lm_head_param(name)
            param.requires_grad = trainable
            matched += int(trainable)
        if matched:
            return "lm_head"
        raise RuntimeError(
            "train_scope=lm_head could not identify the model output embeddings. "
            "Refusing to fall back to non_vision because that can exceed Colab T4 VRAM."
        )

    for name, param in model.named_parameters():
        if train_scope == "all":
            param.requires_grad = True
        elif train_scope == "non_vision":
            param.requires_grad = not is_vision_param(name)
        else:
            raise ValueError(f"Unsupported train_scope={train_scope}")
    return train_scope


def extract_image_path(record: dict) -> str | None:
    """Return the image path from an exported SFT record."""
    for part in record.get("messages", [{}])[0].get("content", []):
        if isinstance(part, dict) and part.get("type") == "image":
            return part.get("image")
    return None


def extract_assistant_text(record: dict) -> str:
    """Return assistant transcript from an exported SFT record."""
    messages = record.get("messages", [])
    if len(messages) < 2:
        return ""
    content = messages[1].get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
        return "".join(texts)
    return str(content)


def transcribe_for_validation(
    *,
    model,
    processor,
    image,
    device,
    user_prompt: str,
    max_pixels: int,
    max_new_tokens: int,
) -> str:
    """Generate one transcript during epoch validation."""
    import torch

    from paddle_vl_shared import clean_vl_transcript  # noqa: E402

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
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
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return clean_vl_transcript(processor.decode(new_tokens, skip_special_tokens=True))


def validate_epoch(
    *,
    model,
    processor,
    val_samples: list[dict],
    device,
    max_pixels: int,
    max_new_tokens: int,
    user_prompt: str,
) -> dict:
    """Run a small validation subset and return aggregate metrics."""
    from PIL import Image
    from evaluate_utils import aggregate_metrics  # noqa: E402

    if not val_samples:
        return {"n": 0}
    was_training = model.training
    model.eval()
    pairs: list[tuple[str, str]] = []
    failed = 0
    for rec in val_samples:
        image_path = extract_image_path(rec)
        gt = extract_assistant_text(rec)
        if not image_path or not gt:
            failed += 1
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            pred = transcribe_for_validation(
                model=model,
                processor=processor,
                image=image,
                device=device,
                user_prompt=user_prompt,
                max_pixels=max_pixels,
                max_new_tokens=max_new_tokens,
            )
            pairs.append((pred, gt))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log.warning("Validation sample failed (%s): %s", image_path, exc)
    if was_training:
        model.train()
    metrics = aggregate_metrics(pairs) if pairs else {"n": 0}
    metrics["failed"] = failed
    return metrics


def build_labels_assistant_only(
    processor,
    image,
    assistant_text: str,
    full_inputs: dict,
    device: Any,
    max_pixels: int,
    user_prompt: str,
) -> tuple[Any, bool]:
    """
    Build ``labels`` for supervised fine-tuning: ``-100`` on non-assistant positions.

    Uses the same pattern as TRL/HF docs: tokenize user (+ image) with
    ``add_generation_prompt=True``, then mask ``labels[:, :prefix_len] = -100``.
    Returns (labels, ok). If ok is False (mismatch, empty assistant, or no trainable
    tokens), the caller should skip the optimization step.
    """
    import torch

    images_kwargs = {
        "size": {
            "shortest_edge": getattr(
                processor.image_processor, "min_pixels", 28 * 28 * 4
            ),
            "longest_edge": max_pixels,
        }
    }
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    prompt_inputs = processor.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"images_kwargs": images_kwargs},
    )
    if hasattr(prompt_inputs, "to"):
        prompt_inputs = prompt_inputs.to(device)
    else:
        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}

    input_ids = full_inputs["input_ids"]
    labels = input_ids.clone()
    prefix_len = int(prompt_inputs["input_ids"].shape[-1])
    seq_len = int(input_ids.shape[-1])

    if prefix_len >= seq_len:
        log.warning(
            "prefix_len=%d >= seq_len=%d; skip assistant-only masking this step.",
            prefix_len,
            seq_len,
        )
        return labels, False

    p_ids = prompt_inputs["input_ids"][0, :prefix_len]
    f_ids = input_ids[0, :prefix_len]
    if not torch.equal(p_ids, f_ids):
        log.warning(
            "Prompt tokenization mismatch vs full sequence (prefix_len=%d); skip step.",
            prefix_len,
        )
        return labels, False

    labels[:, :prefix_len] = -100
    if "attention_mask" in full_inputs:
        am = full_inputs["attention_mask"]
        labels = labels.masked_fill(am == 0, -100)
    trainable = labels != -100
    if not trainable.any():
        log.warning("No trainable label positions after masking; skip step.")
        return labels, False
    return labels, True


def main() -> None:
    """Run VLM fine-tuning and save model."""
    args = parse_args()
    import random

    import numpy as np
    import torch
    from PIL import Image

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        import transformers
        import accelerate
        from transformers import AutoModel, AutoProcessor
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
        log.error("CRITICAL: transformers>=5.0.0 and accelerate>=1.1.0 are required to load and fine-tune PaddleOCR-VL.")
        log.error("Please run: pip install -U 'transformers>=5' 'accelerate>=1.1.0' 'huggingface_hub>=1.5.0'")
        log.error("And RESTART the Colab runtime session (Runtime > Restart session).")
        log.error("=" * 80)
        raise ImportError(
            "Install: pip install 'transformers>=5' 'accelerate>=1.1.0' torch"
        ) from exc

    samples = load_train_samples(args.export_dir, args.max_samples)
    log.info("Training samples: %d", len(samples))

    training_state_path = args.output_dir / "training_state.json"
    start_epoch = 0
    load_model_id: str | Path = args.model_id
    if args.resume and training_state_path.is_file():
        try:
            state = json.loads(training_state_path.read_text(encoding="utf-8"))
            saved_scope = state.get("train_scope")
            if saved_scope and saved_scope != args.train_scope:
                raise ValueError(
                    f"checkpoint train_scope={saved_scope!r} does not match "
                    f"requested train_scope={args.train_scope!r}"
                )
            if not saved_scope:
                raise ValueError(
                    "checkpoint predates train-scope metadata; start a clean run "
                    "to avoid mixing incompatible experiments"
                )
            start_epoch = int(state.get("completed_epochs", 0))
            load_model_id = args.output_dir
            log.info(
                "Resuming weights from epoch %d checkpoint %s.",
                start_epoch,
                args.output_dir,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            start_epoch = 0
            load_model_id = args.model_id
            log.warning("Resume checkpoint rejected: %s. Starting cleanly.", exc)

    sys.path.insert(0, str(Path(__file__).parent))
    from paddle_vl_shared import (  # noqa: E402
        hf_trust_remote_code_model,
        hf_trust_remote_code_processor,
        select_torch_dtype,
    )

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

        processor = AutoProcessor.from_pretrained(
            args.model_id, trust_remote_code=hf_trust_remote_code_processor()
        )

        # Flush Paddle/other CUDA contexts before loading PyTorch model
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _dtype, _dtype_label = select_torch_dtype()
        log.info("Loading PaddleOCR-VL-1.6 for SFT training in %s (CUDA=%s)", _dtype_label, torch.cuda.is_available())

        model_kwargs = {
            "dtype": _dtype,
            "trust_remote_code": hf_trust_remote_code_model(),
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        model = AutoModel.from_pretrained(
            load_model_id,
            **model_kwargs,
        )

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
    except (KeyError, ValueError) as exc:
        log.error("=" * 80)
        log.error("CRITICAL ERROR: Failed to load PaddleOCR-VL model for SFT training.")
        log.error("This is usually because the active python environment has an outdated transformers")
        log.error("or accelerate version loaded in memory. To fix this:")
        log.error("1. Run the dependencies installation cell.")
        log.error("2. Restart the Colab session (Runtime > Restart session).")
        log.error("=" * 80)
        raise RuntimeError(
            "PaddleOCR-VL architecture loading failed. Please install transformers>=5.0.0 and accelerate>=1.1.0 and restart your kernel."
        ) from exc

    effective_train_scope = apply_train_scope(model, args.train_scope)
    log.info(
        "SFT train scope after resume/load: requested=%s effective=%s",
        args.train_scope,
        effective_train_scope,
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    if trainable_params == 0:
        raise RuntimeError(f"No trainable parameters selected by train_scope={args.train_scope}")
    log.info(
        "Post-resume trainable params: %s || All params: %s || Trainable%%: %.4f",
        f"{trainable_params:,}",
        f"{all_params:,}",
        100 * trainable_params / all_params,
    )
    log.info("Post-resume trainable parameter groups preview: %s", trainable_names[:12])

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # Learning rate scheduler: linear warmup + cosine decay
    remaining_epochs = max(0, args.epochs - start_epoch)
    steps_per_epoch = math.ceil(
        len(samples) / max(1, int(args.gradient_accumulation_steps))
    )
    total_steps = max(1, steps_per_epoch * max(1, remaining_epochs))
    warmup_steps = max(1, total_steps // 10)  # 10% warmup

    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(opt, lr_lambda)
    log.info(
        "LR schedule: %d warmup → cosine decay over %d total steps",
        warmup_steps,
        total_steps,
    )

    sys.path.insert(0, str(Path(__file__).parent))
    from paddle_vl_shared import USER_TEXT_OCR_YORUBA  # noqa: E402

    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_samples = load_jsonl_samples(args.export_dir / "val.jsonl", args.val_samples)
    if args.val_samples and not val_samples:
        log.warning("Validation requested but no val.jsonl samples were found under %s.", args.export_dir)
    if val_samples:
        log.info("Epoch validation enabled: %d samples", len(val_samples))
    train_log_path = args.output_dir / "training_log.jsonl"
    best_dir = args.output_dir / "best"
    best_validation_path = args.output_dir / "best_validation.json"
    best_cer = float("inf")
    if start_epoch > 0 and best_validation_path.is_file():
        try:
            best_state = json.loads(best_validation_path.read_text(encoding="utf-8"))
            best_cer = float(best_state.get("cer", float("inf")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            log.warning("Could not read %s; best-model tracking restarts.", best_validation_path)
    if start_epoch == 0 and train_log_path.exists():
        train_log_path.unlink()
    max_pixels = int(args.max_pixels)
    log.info("SFT image max_pixels=%d", max_pixels)
    device = next(model.parameters()).device
    grad_accum = max(1, int(args.gradient_accumulation_steps))
    empty_cache_steps = max(0, int(args.empty_cache_steps))

    from tqdm import tqdm

    for epoch in range(start_epoch, args.epochs):
        # Shuffle training data each epoch to prevent order memorization
        random.shuffle(samples)

        total_loss = 0.0
        micro_steps = 0
        opt_steps = 0
        skipped = 0
        accum_counter = 0
        opt.zero_grad(set_to_none=True)

        epoch_iterator = tqdm(
            samples,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            unit="img",
            leave=True
        )

        for rec_idx, rec in enumerate(epoch_iterator):
            image_path = extract_image_path(rec)
            if not image_path:
                skipped += 1
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except (OSError, ValueError) as exc:
                log.warning("Unreadable training image %s: %s", image_path, exc)
                skipped += 1
                continue
            out = None
            loss = None


            assistant_text = extract_assistant_text(rec)
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": USER_TEXT_OCR_YORUBA},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                },
            ]
            inputs = processor.apply_chat_template(
                user_messages,
                add_generation_prompt=False,
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

            if args.full_sequence_loss:
                labels = inputs["input_ids"].clone()
                if "attention_mask" in inputs:
                    labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)
            else:
                labels, ok = build_labels_assistant_only(
                    processor,
                    image,
                    assistant_text,
                    inputs,
                    device,
                    max_pixels,
                    USER_TEXT_OCR_YORUBA,
                )
                if not ok:
                    del inputs, labels, image
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    skipped += 1
                    continue

            # OOM-safe forward pass: skip sample on memory error instead of crashing
            try:
                out = model(**inputs, labels=labels)
            except torch.cuda.OutOfMemoryError:
                log.warning("OOM on sample %d — clearing cache and skipping.", rec_idx)
                del inputs, labels, image
                torch.cuda.empty_cache()
                opt.zero_grad(set_to_none=True)
                accum_counter = 0
                skipped += 1
                continue

            try:
                loss = out.loss
                if loss is None or not torch.isfinite(loss):
                    skipped += 1
                    del out, loss, inputs, labels, image
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                loss = loss / grad_accum
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                log.warning(
                    "OOM during backward on sample %d — clearing cache and skipping.",
                    rec_idx,
                )
                del out, loss, inputs, labels, image
                torch.cuda.empty_cache()
                opt.zero_grad(set_to_none=True)
                accum_counter = 0
                skipped += 1
                continue
            accum_counter += 1
            total_loss += float(loss.item()) * grad_accum
            micro_steps += 1
            del out, loss, inputs, labels, image

            if accum_counter >= grad_accum:
                # Gradient clipping to prevent training instability
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                accum_counter = 0
                opt_steps += 1
                current_lr = scheduler.get_last_lr()[0]
                mean_loss = total_loss / max(micro_steps, 1)
                epoch_iterator.set_postfix(
                    loss=f"{mean_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    skipped=skipped
                )
                if opt_steps % 10 == 0:
                    log.info(
                        "epoch %d step %d/%d micro=%d loss=%.4f lr=%.2e skipped=%d",
                        epoch + 1,
                        opt_steps,
                        total_steps,
                        micro_steps,
                        mean_loss,
                        current_lr,
                        skipped,
                    )
            if (
                empty_cache_steps
                and micro_steps > 0
                and micro_steps % empty_cache_steps == 0
                and torch.cuda.is_available()
            ):
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()

        if accum_counter > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)
            opt_steps += 1

        train_mean_loss = total_loss / max(micro_steps, 1)
        if micro_steps == 0:
            raise RuntimeError(
                "No training samples produced a valid optimization step. "
                "Check the exported image paths and assistant-token masking."
            )
        val_metrics = {}
        if val_samples:
            val_metrics = validate_epoch(
                model=model,
                processor=processor,
                val_samples=val_samples,
                device=device,
                max_pixels=max_pixels,
                max_new_tokens=int(args.eval_max_new_tokens),
                user_prompt=USER_TEXT_OCR_YORUBA,
            )
            log.info(
                "epoch %d validation: n=%s CER=%s WER=%s DER=%s failed=%s",
                epoch + 1,
                val_metrics.get("n"),
                (
                    f"{val_metrics['cer'] * 100:.2f}%"
                    if val_metrics.get("cer") is not None
                    else "—"
                ),
                (
                    f"{val_metrics['wer'] * 100:.2f}%"
                    if val_metrics.get("wer") is not None
                    else "—"
                ),
                (
                    f"{val_metrics['der'] * 100:.2f}%"
                    if val_metrics.get("der") is not None
                    else "—"
                ),
                val_metrics.get("failed", 0),
            )
            current_cer = val_metrics.get("cer")
            if current_cer is not None and float(current_cer) < best_cer:
                best_cer = float(current_cer)
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
                best_validation_path.write_text(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "cer": best_cer,
                            "wer": val_metrics.get("wer"),
                            "der": val_metrics.get("der"),
                            "n": val_metrics.get("n"),
                            "checkpoint": str(best_dir),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                log.info(
                    "New best validation CER %.2f%% at epoch %d → %s",
                    best_cer * 100,
                    epoch + 1,
                    best_dir,
                )

        log.info(
            "epoch %d done: micro_steps=%d opt_steps=%d skipped=%d mean_loss=%.4f",
            epoch + 1,
            micro_steps,
            opt_steps,
            skipped,
            train_mean_loss,
        )
        with train_log_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "micro_steps": micro_steps,
                        "opt_steps": opt_steps,
                        "skipped": skipped,
                        "mean_loss": train_mean_loss,
                        "val": val_metrics,
                        "train_scope": effective_train_scope,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        training_state_path.write_text(
            json.dumps(
                {
                    "completed_epochs": epoch + 1,
                    "total_epochs": args.epochs,
                    "train_scope": effective_train_scope,
                    "base_model_id": args.model_id,
                    "learning_rate": args.lr,
                    "gradient_accumulation_steps": grad_accum,
                    "max_pixels": max_pixels,
                    "full_sequence_loss": bool(args.full_sequence_loss),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        log.info("Checkpoint saved after epoch %d → %s", epoch + 1, args.output_dir)

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    log.info("Saved fine-tuned model and processor to %s", args.output_dir)


if __name__ == "__main__":
    main()
