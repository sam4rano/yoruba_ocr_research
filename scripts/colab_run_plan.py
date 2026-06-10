"""
Colab notebook run plan → pipeline ``SKIP_*`` environment variables.

Edit ``RUN_PLAN`` in ``yor_ocr.ipynb`` Step 1, then call ``apply_run_plan()``.

Sections
--------
A  Out-of-the-box: TrOCR zero-shot, PP-OCR EN pretrained, Surya v2 zero-shot
B  Fine-tuned: PaddleOCR-VL-1.5 (export → zero-shot → LoRA → eval) + Surya Foundation
C  PP-OCRv4 CRNN ablations (Tables 2–4)
D  Analysis + compile Table 1
Appendix  HF dataset upload + HF model uploads (VL LoRA + Surya)
"""

from __future__ import annotations

import os
from typing import Mapping

DEFAULT_RUN_PLAN: dict[str, bool] = {
    "A_baselines_ootb": True,
    "B_finetuned": True,
    "C_ppocr_ablations": True,
    "D_analysis_compile": True,
    "appendix_hf_dataset": False,
    "appendix_hf_models": False,
    "extra_qwen_zero_shot": False,
    "extra_trocr_finetune": False,
    "extra_ppocr_full_train": False,
}

DATA_DEFAULTS: dict[str, str] = {
    "USE_EXISTING_PROCESSED_DATA": "1",
    "ARCHIVE_METRICS_BEFORE_RUN": "1",
    "SKIP_CONSOLIDATE": "1",
    "RUN_RESPLIT": "0",
    "RESET_PROCESSED": "0",
    "VL15_LORA_EPOCHS": "1",
    "VL15_TRAIN_RESUME": "1",
    "VL15_GRAD_ACCUM": "4",
    "VL15_QUANTIZE_4BIT": "1",
    "SURYA_INFERENCE_BACKEND": "auto",
    "SKIP_HF_UPLOAD": "1",
    "SKIP_HF_DATASET_UPLOAD": "1",
}


def apply_run_plan(plan: Mapping[str, bool] | None = None) -> dict[str, str]:
    """
    Convert ``RUN_PLAN`` booleans to ``os.environ`` and return the full env dict.
    """
    p = {**DEFAULT_RUN_PLAN, **(plan or {})}

    # Back-compat: old key B_vl15_lora → B_finetuned
    if "B_vl15_lora" in p and "B_finetuned" not in (plan or {}):
        p["B_finetuned"] = p["B_vl15_lora"]

    env: dict[str, str] = dict(DATA_DEFAULTS)
    env["RUN_PLAN_A"] = "1" if p["A_baselines_ootb"] else "0"
    env["RUN_PLAN_B"] = "1" if p["B_finetuned"] else "0"
    env["RUN_PLAN_C"] = "1" if p["C_ppocr_ablations"] else "0"
    env["RUN_PLAN_D"] = "1" if p["D_analysis_compile"] else "0"
    env["RUN_PLAN_APPENDIX_HF"] = (
        "1" if p.get("appendix_hf_dataset") or p.get("appendix_hf_models") else "0"
    )

    # Section A — OOTB baselines (no training)
    env["SKIP_TROCR_ZERO_SHOT"] = "0" if p["A_baselines_ootb"] else "1"
    env["SKIP_SURYA"] = "0" if p["A_baselines_ootb"] else "1"
    env["SKIP_VL15_ZERO_SHOT"] = "1"
    env["SKIP_QWEN"] = "0" if p["extra_qwen_zero_shot"] else "1"

    # Section B — fine-tuned models
    env["SKIP_VL15_LORA"] = "0" if p["B_finetuned"] else "1"
    env["SKIP_VL15_ZERO_SHOT"] = "0" if p["B_finetuned"] else "1"
    env["SKIP_SURYA_FINETUNE"] = "0" if p["B_finetuned"] else "1"

    # Section C — PP-OCRv4 ablations
    env["SKIP_ABLATION"] = "0" if p["C_ppocr_ablations"] else "1"
    env["SKIP_PADDLE_FINETUNE"] = "1"
    env["SKIP_PADDLE_TRAIN"] = (
        "0" if p["extra_ppocr_full_train"] and not p["C_ppocr_ablations"] else "1"
    )

    # Section D — analysis + compile
    env["SKIP_ANALYSIS"] = "0" if p["D_analysis_compile"] else "1"

    # Appendix — Hugging Face releases
    env["SKIP_HF_DATASET_UPLOAD"] = "0" if p.get("appendix_hf_dataset") else "1"
    env["SKIP_HF_UPLOAD"] = "0" if p.get("appendix_hf_models") else "1"

    # Optional extras (appendix / legacy)
    env["SKIP_TROCR"] = "0" if p["extra_trocr_finetune"] else "1"

    for key, val in env.items():
        os.environ[key] = val
    return env


def print_run_summary(plan: Mapping[str, bool] | None = None) -> None:
    """Print a one-screen summary of what will run."""
    p = {**DEFAULT_RUN_PLAN, **(plan or {})}
    if "B_vl15_lora" in p and "B_finetuned" not in (plan or {}):
        p["B_finetuned"] = p["B_vl15_lora"]
    lines = [
        "",
        "═══ Run plan ═══",
        f"  A  OOTB baselines            : {'ON' if p['A_baselines_ootb'] else 'OFF'}",
        "       → TrOCR ZS, PP-OCR EN, Surya v2 ZS",
        f"  B  Fine-tuned models         : {'ON' if p['B_finetuned'] else 'OFF'}",
        "       → VL-1.5 export/ZS/LoRA/eval + Surya Foundation fine-tune",
        f"  C  PP-OCRv4 ablations        : {'ON' if p['C_ppocr_ablations'] else 'OFF'}",
        "       → data size / dict / aug (Tables 2–4; long GPU run)",
        f"  D  Analysis + compile        : {'ON' if p['D_analysis_compile'] else 'OFF'}",
        "─── Appendix (default OFF) ───",
        f"  + HF dataset upload         : {'ON' if p.get('appendix_hf_dataset') else 'off'}",
        f"  + HF model uploads (VL+Surya): {'ON' if p.get('appendix_hf_models') else 'off'}",
        "─── Optional extras ───",
        f"  + Qwen 2.5 VL zero-shot     : {'ON' if p['extra_qwen_zero_shot'] else 'off'}",
        f"  + TrOCR fine-tune           : {'ON' if p['extra_trocr_finetune'] else 'off'}",
        f"  + PP-OCR Phase 04 only      : {'ON' if p['extra_ppocr_full_train'] else 'off'}",
        "",
        "Order after setup: A → B → C → D → Appendix",
        "Note: Surya fine-tune (B) downgrades to surya 0.15.x — run A (v2 ZS) first.",
        "",
    ]
    print("\n".join(lines))
