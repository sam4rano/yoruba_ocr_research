"""
Colab notebook run plan → pipeline ``SKIP_*`` environment variables.

Edit ``RUN_PLAN`` in ``yor_ocr.ipynb`` Step 1, then call ``apply_run_plan()``.

Active model stack (3 models zero-shot + 1 fine-tuned):
  - Base PaddleOCR (English pretrained recognition) [zero-shot]
  - PaddleOCR-VL-1.6 [zero-shot]
  - GLM-OCR [zero-shot]
  - PaddleOCR-VL-1.6 fine-tuned [fine-tuned LM]

Sections
--------
A  Out-of-the-box zero-shot: PaddleOCR EN pretrained, PaddleOCR-VL-1.6, GLM-OCR
B  PaddleOCR-VL-1.6 LM fine-tuning
D  Analysis + compile Table 1
Appendix  HF dataset upload
"""

from __future__ import annotations

import os
from typing import Mapping

DEFAULT_RUN_PLAN: dict[str, bool] = {
    "A_baselines_ootb": True,
    "B_vl16_finetune": True,
    "D_analysis_compile": True,
    "appendix_hf_dataset": False,
}

DATA_DEFAULTS: dict[str, str] = {
    "USE_EXISTING_PROCESSED_DATA": "1",
    "ARCHIVE_METRICS_BEFORE_RUN": "1",
    "SKIP_CONSOLIDATE": "1",
    "RUN_RESPLIT": "0",
    "RESET_PROCESSED": "0",
    "HF_TRUST_REMOTE_CODE": "1",
    "SKIP_HF_UPLOAD": "1",
    "SKIP_HF_DATASET_UPLOAD": "1",
}


def apply_run_plan(plan: Mapping[str, bool] | None = None) -> dict[str, str]:
    """
    Convert ``RUN_PLAN`` booleans to ``os.environ`` and return the full env dict.
    """
    p = {**DEFAULT_RUN_PLAN, **(plan or {})}

    env: dict[str, str] = dict(DATA_DEFAULTS)
    env["RUN_PLAN_A"] = "1" if p["A_baselines_ootb"] else "0"
    env["RUN_PLAN_B"] = "1" if p["B_vl16_finetune"] else "0"
    env["RUN_PLAN_D"] = "1" if p["D_analysis_compile"] else "0"
    env["RUN_PLAN_APPENDIX_HF"] = "1" if p.get("appendix_hf_dataset") else "0"

    # Section A — OOTB zero-shot baselines (no training)
    env["SKIP_PPOCR_PRETRAINED"] = "0" if p["A_baselines_ootb"] else "1"
    env["SKIP_PADDLEOCRVL16_ZERO_SHOT"] = "0" if p["A_baselines_ootb"] else "1"
    env["SKIP_GLM_ZERO_SHOT"] = "0" if p["A_baselines_ootb"] else "1"

    # Section B — PaddleOCR-VL-1.6 fine-tuning
    env["SKIP_PADDLEOCRVL16_SFT_EXPORT"] = "0" if p["B_vl16_finetune"] else "1"
    env["SKIP_PADDLEOCRVL16_SFT_TRAIN"] = "0" if p["B_vl16_finetune"] else "1"
    env["SKIP_PADDLEOCRVL16_SFT_EVAL"] = "0" if p["B_vl16_finetune"] else "1"

    # Section D — analysis + compile
    env["SKIP_ANALYSIS"] = "0" if p["D_analysis_compile"] else "1"

    # Appendix — Hugging Face dataset release
    env["SKIP_HF_DATASET_UPLOAD"] = "0" if p.get("appendix_hf_dataset") else "1"

    for key, val in env.items():
        os.environ[key] = val
    return env


def print_run_summary(plan: Mapping[str, bool] | None = None) -> None:
    """Print a one-screen summary of what will run."""
    p = {**DEFAULT_RUN_PLAN, **(plan or {})}
    lines = [
        "",
        "═══ Run plan ═══",
        f"  A  OOTB zero-shot baselines   : {'ON' if p['A_baselines_ootb'] else 'OFF'}",
        "       → PaddleOCR EN pretrained, PaddleOCR-VL-1.6, GLM-OCR",
        f"  B  PaddleOCR-VL-1.6 fine-tune : {'ON' if p['B_vl16_finetune'] else 'OFF'}",
        "       → export SFT data + full LM fine-tuning (long GPU run)",
        f"  D  Analysis + compile         : {'ON' if p['D_analysis_compile'] else 'OFF'}",
        "─── Appendix (default OFF) ───",
        f"  + HF dataset upload           : {'ON' if p.get('appendix_hf_dataset') else 'off'}",
        "",
        "Order after setup: A → B → D → Appendix",
        "",
    ]
    print("\n".join(lines))
