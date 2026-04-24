"""
Shared Yorùbá Unicode whitelist.

Both ``scripts/01_consolidate_data.py`` (filtering) and
``scripts/02b_data_quality_audit.py`` (reporting) rely on this set being
identical, so it lives in a single module instead of being duplicated.

Every codepoint here is either:

* ASCII printable (including space and the standard punctuation used in
  Yorùbá prose),
* a combining diacritic used by Yorùbá orthography (grave, acute,
  macron, dot-below),
* a precomposed Latin letter with a Yorùbá-relevant diacritic, or
* a General Punctuation symbol that appears in legitimate prose
  (en-dash, curly quotes).

Anything outside this set — Greek, Vietnamese circumflex+dot, heavy
checkmarks, copyright signs, or the U+2026 ellipsis used by
fill-in-the-blank questionnaire lines — is treated as noise.
"""

from __future__ import annotations

YORUBA_WHITELIST_CODEPOINTS: frozenset[int] = frozenset(
    {
        *range(0x0020, 0x007F),  # ASCII printable incl. space
        0x00A0,  # NBSP
        # Combining marks
        0x0300,  # combining grave
        0x0301,  # combining acute
        0x0304,  # combining macron
        0x0323,  # combining dot below
        # Latin-1 Supplement: vowels with grave/acute
        0x00C0,
        0x00C1,
        0x00C8,
        0x00C9,
        0x00CC,
        0x00CD,
        0x00D2,
        0x00D3,
        0x00D9,
        0x00DA,
        0x00E0,
        0x00E1,
        0x00E8,
        0x00E9,
        0x00EC,
        0x00ED,
        0x00F2,
        0x00F3,
        0x00F9,
        0x00FA,
        # Latin Extended-A: macron variants + n-acute
        0x0100,
        0x0101,
        0x0112,
        0x0113,
        0x012A,
        0x012B,
        0x014C,
        0x014D,
        0x016A,
        0x016B,
        0x0143,
        0x0144,
        0x01F8,
        0x01F9,
        # Latin Extended Additional: dot-below letters
        0x1E62,
        0x1E63,  # Ṣ / ṣ
        0x1EB8,
        0x1EB9,  # Ẹ / ẹ
        0x1ECC,
        0x1ECD,  # Ọ / ọ
        # General Punctuation present in legitimate Yorùbá prose
        0x2013,  # en dash
        0x2018,  # left single quotation mark
        0x2019,  # right single quotation mark
    }
)


def has_only_whitelisted_codepoints(text: str) -> bool:
    """Return ``True`` iff every codepoint in ``text`` is whitelisted."""
    return all(ord(ch) in YORUBA_WHITELIST_CODEPOINTS for ch in text)
