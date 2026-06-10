"""
Shared helpers for PaddleOCR-VL-1.5 (Hugging Face) export, training, and evaluation.

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


def hf_trust_remote_code_model() -> bool:
    """
    Whether ``AutoModelForImageTextToText`` should run hub custom code.

    Default ``False`` avoids a known config mismatch on some transformers builds
    (PaddleOCR#17666). Set ``HF_TRUST_REMOTE_CODE=1`` to force hub modeling code.
    """
    import os

    v = os.environ.get("HF_TRUST_REMOTE_CODE", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def hf_trust_remote_code_processor() -> bool:
    """
    Whether ``AutoProcessor`` should run hub custom code.

    Default ``True``: recent transformers require hub processor code for
    ``PaddlePaddle/PaddleOCR-VL-1.5`` (Colab fails without it).
    """
    import os

    v = os.environ.get("HF_TRUST_REMOTE_CODE", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


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
