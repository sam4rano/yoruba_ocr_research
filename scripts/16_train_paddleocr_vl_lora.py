"""
LoRA fine-tuning for PaddleOCR-VL-1.5 on exported JSONL (see ``14_export_paddleocr_vl_sft.py``).

**Training objective:** causal LM loss with **assistant tokens only** (standard SFT):
prompt positions (vision + user text + generation header) are masked with ``-100`` in
``labels``, matching common HF/TRL practice. Use ``--full-sequence-loss`` only for
debugging or ablations.

Optional **gradient accumulation** (``--gradient-accumulation-steps``) reduces optimizer
frequency and can improve stability; micro-batch size remains one image (typical for VL).

Outputs a PEFT adapter under ``--output-dir``; evaluate with
``15_baseline_paddleocr_vl15.py --adapter-path <that dir>/adapter``.

Usage:
    python scripts/14_export_paddleocr_vl_sft.py
    python scripts/16_train_paddleocr_vl_lora.py --epochs 1 --max-samples 500
    python scripts/16_train_paddleocr_vl_lora.py --gradient-accumulation-steps 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune PaddleOCR-VL-1.5 on paddleocr_vl15_sft JSONL export."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("data/paddleocr_vl15_sft"),
        help="Directory containing train.jsonl from script 14.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="PaddlePaddle/PaddleOCR-VL-1.5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/paddleocr_vl15_lora"),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
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
        default=1,
        help="Optimizer step every N forward passes (default 1).",
    )
    parser.add_argument(
        "--full-sequence-loss",
        action="store_true",
        help="Train on full sequence (no label masking). Not recommended for SFT.",
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
        images_kwargs=images_kwargs,
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
    """Run LoRA fine-tuning and save adapter."""
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
        from peft import LoraConfig, get_peft_model  # type: ignore
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Install: pip install 'transformers>=5' peft accelerate torch"
        ) from exc

    samples = load_train_samples(args.export_dir, args.max_samples)
    log.info("Training samples: %d", len(samples))

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=False,
    )

    wanted = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    found: set[str] = set()
    for name in wanted:
        for n, _ in model.named_modules():
            if n.endswith(name):
                found.add(name)
                break
    target_modules = sorted(found) or ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    sys.path.insert(0, str(Path(__file__).parent))
    from paddle_vl_shared import USER_TEXT_OCR_YORUBA  # noqa: E402

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.output_dir / "adapter"
    max_pixels = 1280 * 28 * 28
    device = next(model.parameters()).device
    grad_accum = max(1, int(args.gradient_accumulation_steps))

    for epoch in range(args.epochs):
        total_loss = 0.0
        micro_steps = 0
        opt_steps = 0
        accum_counter = 0
        opt.zero_grad()

        for rec in samples:
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
            assistant_text = (
                raw_asst if isinstance(raw_asst, str) else str(raw_asst)
            )
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": USER_TEXT_OCR_YORUBA},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
            ]
            inputs = processor.apply_chat_template(
                user_messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                images_kwargs={
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

            out = model(**inputs, labels=labels)
            loss = out.loss
            if loss is None or torch.isnan(loss):
                continue
            loss = loss / grad_accum
            loss.backward()
            accum_counter += 1
            total_loss += float(loss.item()) * grad_accum
            micro_steps += 1

            if accum_counter >= grad_accum:
                opt.step()
                opt.zero_grad()
                accum_counter = 0
                opt_steps += 1
                if opt_steps % 50 == 0:
                    log.info(
                        "epoch %d opt_step %d micro=%d mean_loss=%.4f",
                        epoch + 1,
                        opt_steps,
                        micro_steps,
                        total_loss / max(micro_steps, 1),
                    )

        if accum_counter > 0:
            opt.step()
            opt.zero_grad()
            opt_steps += 1

        log.info(
            "epoch %d done: micro_steps=%d opt_steps=%d mean_loss=%.4f",
            epoch + 1,
            micro_steps,
            opt_steps,
            total_loss / max(micro_steps, 1),
        )

    model.save_pretrained(adapter_dir)
    processor.save_pretrained(args.output_dir / "processor_snapshot")
    log.info("Saved LoRA adapter to %s", adapter_dir)


if __name__ == "__main__":
    main()
