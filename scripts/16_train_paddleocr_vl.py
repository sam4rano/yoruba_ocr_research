"""
Fine-tuning for PaddleOCR-VL-1.6 on exported JSONL (see ``14_export_paddleocr_vl_sft.py``).

**Training objective:** causal LM loss with **assistant tokens only** (standard SFT):
prompt positions (vision + user text + generation header) are masked with ``-100`` in
``labels``, matching common HF/TRL practice. Use ``--full-sequence-loss`` only for
debugging or ablations.

Optional **gradient accumulation** (``--gradient-accumulation-steps``) reduces optimizer
frequency and can improve stability; micro-batch size remains one image (typical for VL).

Outputs a fine-tuned model under ``--output-dir``; evaluate with
``15_baseline_paddleocr_vl16.py --model-id <that dir>``.

Usage:
    python scripts/14_export_paddleocr_vl_sft.py
    python scripts/16_train_paddleocr_vl.py --epochs 5 --max-samples 500
    python scripts/16_train_paddleocr_vl.py --gradient-accumulation-steps 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune PaddleOCR-VL-1.6 on paddleocr_vl16_sft JSONL export."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("data/paddleocr_vl16_sft"),
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
        default=Path("experiments/paddleocr_vl16_finetuned"),
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
            f"Missing {path}. Run: python scripts/14_export_paddleocr_vl_sft.py"
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

    sys.path.insert(0, str(Path(__file__).parent))
    from paddle_vl_shared import (  # noqa: E402
        hf_trust_remote_code_model,
        hf_trust_remote_code_processor,
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
        model_kwargs = {
            "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "trust_remote_code": hf_trust_remote_code_model(),
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        model = AutoModel.from_pretrained(
            args.model_id,
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

    # Freeze the vision tower to save VRAM and avoid overfitting to scan artefacts.
    for name, param in model.named_parameters():
        if "visual" in name or "vision" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Enable gradient checkpointing to save memory
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Print parameter training stats
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    log.info(
        f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.4f}"
    )

    training_state_path = args.output_dir / "training_state.json"
    start_epoch = 0

    if args.resume and training_state_path.is_file():
        try:
            # Standalone weights are loaded in-place when resuming
            resume_kwargs = {
                "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                "trust_remote_code": hf_trust_remote_code_model(),
            }
            if torch.cuda.is_available():
                resume_kwargs["device_map"] = "auto"
            model = AutoModel.from_pretrained(
                args.output_dir,
                **resume_kwargs,
            )
            state = json.loads(training_state_path.read_text(encoding="utf-8"))
            start_epoch = int(state.get("completed_epochs", 0))
            log.info("Resuming training from epoch %d with checkpoint %s.", start_epoch, args.output_dir)
        except Exception as e:
            log.warning("Could not resume checkpoint weights: %s. Starting from scratch.", e)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # Learning rate scheduler: linear warmup + cosine decay
    total_steps = (len(samples) * args.epochs) // max(
        1, int(args.gradient_accumulation_steps)
    )
    warmup_steps = max(1, total_steps // 10)  # 10% warmup
    import math

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
    # Must match eval (15_baseline_paddleocr_vl15.py) resolution to avoid
    # train/eval distribution shift.  768 * 28 * 28 = 602,112 is safe for T4.
    max_pixels = 768 * 28 * 28
    device = next(model.parameters()).device
    grad_accum = max(1, int(args.gradient_accumulation_steps))

    from tqdm import tqdm

    for epoch in range(start_epoch, args.epochs):
        # Shuffle training data each epoch to prevent order memorization
        random.shuffle(samples)

        total_loss = 0.0
        micro_steps = 0
        opt_steps = 0
        skipped = 0
        accum_counter = 0
        opt.zero_grad()

        epoch_iterator = tqdm(
            samples,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            unit="img",
            leave=True
        )

        for rec_idx, rec in enumerate(epoch_iterator):
            msgs = rec["messages"]
            image_path = None
            for part in msgs[0]["content"]:
                if part.get("type") == "image":
                    image_path = part["image"]
                    break
            if not image_path:
                continue
            image = Image.open(image_path).convert("RGB")


            raw_asst = msgs[1]["content"]
            assistant_text = raw_asst if isinstance(raw_asst, str) else str(raw_asst)
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
                    continue

            # OOM-safe forward pass: skip sample on memory error instead of crashing
            try:
                out = model(**inputs, labels=labels)
            except torch.cuda.OutOfMemoryError:
                log.warning("OOM on sample %d — clearing cache and skipping.", rec_idx)
                torch.cuda.empty_cache()
                opt.zero_grad()
                accum_counter = 0
                skipped += 1
                continue

            loss = out.loss
            if loss is None or torch.isnan(loss):
                skipped += 1
                continue
            loss = loss / grad_accum
            loss.backward()
            accum_counter += 1
            total_loss += float(loss.item()) * grad_accum
            micro_steps += 1

            if accum_counter >= grad_accum:
                # Gradient clipping to prevent training instability
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                opt.step()
                scheduler.step()
                opt.zero_grad()
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

        if accum_counter > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            opt.step()
            scheduler.step()
            opt.zero_grad()
            opt_steps += 1

        log.info(
            "epoch %d done: micro_steps=%d opt_steps=%d skipped=%d mean_loss=%.4f",
            epoch + 1,
            micro_steps,
            opt_steps,
            skipped,
            total_loss / max(micro_steps, 1),
        )
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        training_state_path.write_text(
            json.dumps(
                {
                    "completed_epochs": epoch + 1,
                    "total_epochs": args.epochs,
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
