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
