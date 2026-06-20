"""
Shared helpers for PaddleOCR-VL-1.6 (Hugging Face) zero-shot evaluation.

Does not modify ``data/processed``; only normalises model outputs for metric computation.
"""

from __future__ import annotations

import re
import unicodedata

# Matches HF model card task prompt key "ocr" but specialised for Yorùbá verbatim transcription.
OCR_TASK_TAG = "ocr"
USER_TEXT_OCR_YORUBA = (
    "OCR: Transcribe the single line of text in this image exactly as printed. "
    "The language is Yorùbá. Preserve every tone mark and subdot (ẹ, ọ, ṣ, à, á, etc.). "
    "Output only the line text with no explanation or markdown."
)


def hf_trust_remote_code() -> bool:
    """
    Whether Hugging Face hub custom code should run for PaddleOCR-VL-1.6.

    Default ``True``: ``PaddlePaddle/PaddleOCR-VL-1.6`` requires hub modeling and
    processor code on current transformers. Set ``HF_TRUST_REMOTE_CODE=0`` to opt out.
    """
    import os

    v = os.environ.get("HF_TRUST_REMOTE_CODE", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def hf_trust_remote_code_model() -> bool:
    """Whether ``AutoModelForImageTextToText`` should run hub custom code."""
    return hf_trust_remote_code()


def hf_trust_remote_code_processor() -> bool:
    """Whether ``AutoProcessor`` should run hub custom code."""
    return hf_trust_remote_code()


def clean_vl_transcript(raw: str) -> str:
    """
    Strip common VLM artefacts (fenced code blocks, extra chatter) and NFC-normalise.

    Ground truth in this project is NFC; predictions are normalised the same way
    before CER/WER/DER.
    """
    s = (raw or "").strip()
    if "```" in s:
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    s = s.split("\n")[0] if s else s
    return unicodedata.normalize("NFC", s.strip())
