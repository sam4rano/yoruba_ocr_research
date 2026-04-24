"""
Profile the consolidated dataset for label / image quality issues.

This script is **read-only**. It runs against whatever is currently in
``data/processed/`` and answers a small set of questions the forensic
analysis raised:

* How heterogeneous are the labels (length distribution, empty/near-empty,
  very long multi-line content)?
* How heterogeneous are the images (especially heights — tall images are
  almost certainly multi-line content in disguise)?
* What Unicode blocks actually appear in the labels? Are there stray
  Greek / Turkish / Vietnamese glyphs that inflate the character
  dictionary?
* How many samples would be dropped by a given set of hygiene thresholds
  *before* we apply them for real in ``01_consolidate_data.py``?

Output:
    results/tables/data_quality.json   (default, override with --out-json)

Nothing in ``data/processed/`` is modified. The thresholds used for the
"would_drop" block default to the ones we plan to use in
``01_consolidate_data.py``'s hygiene mode but can be overridden on the
command line to explore alternatives.

Usage::

    python scripts/02b_data_quality_audit.py
    python scripts/02b_data_quality_audit.py --min-label-len 3 --max-label-len 80
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Yorùbá whitelist — imported from the shared module so audits and the
# hygiene filter can never drift apart.
# ---------------------------------------------------------------------------

import sys as _sys
from pathlib import Path as _Path

_SCRIPTS_DIR = _Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPTS_DIR))
from yoruba_charset import YORUBA_WHITELIST_CODEPOINTS  # noqa: E402

# ---------------------------------------------------------------------------
# Unicode-block classification (for histograms)
# ---------------------------------------------------------------------------


def _block_name(cp: int) -> str:
    """Return a coarse block label for a codepoint (for reporting only)."""
    if cp < 0x0080:
        return "ascii"
    if cp < 0x0100:
        return "latin-1-supplement"
    if cp < 0x0180:
        return "latin-extended-a"
    if cp < 0x0250:
        return "latin-extended-b"
    if 0x0300 <= cp < 0x0370:
        return "combining-diacritics"
    if 0x0370 <= cp < 0x0400:
        return "greek"
    if 0x0400 <= cp < 0x0500:
        return "cyrillic"
    if 0x1E00 <= cp < 0x1F00:
        return "latin-extended-additional"
    if 0x2000 <= cp < 0x2070:
        return "general-punctuation"
    if 0x2070 <= cp < 0x20D0:
        return "superscripts-and-currency"
    if 0x2100 <= cp < 0x2200:
        return "letterlike-symbols"
    if 0xFB00 <= cp < 0xFB50:
        return "alphabetic-presentation-forms"
    if cp < 0x0800:
        return "other-bmp-low"
    return "other"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_label_entries(data_dir: Path, split: str) -> list[tuple[Path, str]]:
    """Return ``(image_path, label_text_NFC)`` pairs for ``split``."""
    label_file = data_dir / "labels" / f"{split}.txt"
    if not label_file.exists():
        return []
    entries: list[tuple[Path, str]] = []
    with label_file.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            rel, text = parts
            entries.append((data_dir / rel, unicodedata.normalize("NFC", text)))
    return entries


def load_image_dimensions(paths: list[Path]) -> list[tuple[int, int]]:
    """Return ``(width, height)`` for each path; missing images are skipped.

    Pillow is only imported here so the rest of the script still runs if
    the user just wants label stats.
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        log.warning(
            "Pillow not importable; skipping image-dimension audit. "
            "Run inside the project venv to enable this section."
        )
        return []

    dims: list[tuple[int, int]] = []
    for p in paths:
        try:
            with Image.open(p) as im:
                dims.append(im.size)  # (w, h)
        except (OSError, FileNotFoundError):
            continue
    return dims


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def _quantiles(values: list[float], *, n: int = 10) -> dict[str, float]:
    """Return min / n-tiles / max for ``values`` (empty-safe)."""
    if not values:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    s = sorted(values)
    return {
        "min": float(s[0]),
        "p50": float(statistics.median(s)),
        "p90": float(s[int(0.90 * (len(s) - 1))]),
        "p95": float(s[int(0.95 * (len(s) - 1))]),
        "p99": float(s[int(0.99 * (len(s) - 1))]),
        "max": float(s[-1]),
    }


def profile_split(
    data_dir: Path,
    split: str,
    *,
    min_len: int,
    max_len: int,
    max_height: int,
) -> dict:
    """Return a quality profile for one split."""
    entries = load_label_entries(data_dir, split)
    n = len(entries)
    if n == 0:
        return {"n": 0}

    label_char_lens = [len(t) for _, t in entries]
    label_word_lens = [len(t.split()) for _, t in entries]

    # Codepoint / block histograms (over characters, not samples)
    block_counts: Counter[str] = Counter()
    offending_codepoints: Counter[int] = Counter()
    samples_with_offending: list[dict] = []
    for _, text in entries:
        sample_offenders: list[str] = []
        for ch in text:
            cp = ord(ch)
            block_counts[_block_name(cp)] += 1
            if cp not in YORUBA_WHITELIST_CODEPOINTS:
                offending_codepoints[cp] += 1
                sample_offenders.append(ch)
        if sample_offenders and len(samples_with_offending) < 20:
            samples_with_offending.append(
                {
                    "text": text,
                    "offenders": "".join(sorted(set(sample_offenders))),
                }
            )

    # Image dimensions
    dims = load_image_dimensions([p for p, _ in entries])
    widths = [w for w, _ in dims]
    heights = [h for _, h in dims]

    # Would-drop accounting (does not modify anything on disk)
    dropped_len_short = [t for _, t in entries if len(t) < min_len]
    dropped_len_long = [t for _, t in entries if len(t) > max_len]
    dropped_height = [h for _, h in dims if h > max_height]
    dropped_nonyor_samples = 0
    for _, text in entries:
        if any(ord(ch) not in YORUBA_WHITELIST_CODEPOINTS for ch in text):
            dropped_nonyor_samples += 1

    # Build reporting-friendly top-offender list with readable names
    top_offenders = []
    for cp, count in offending_codepoints.most_common(25):
        try:
            name = unicodedata.name(chr(cp))
        except ValueError:
            name = "<unnamed>"
        top_offenders.append(
            {
                "codepoint": f"U+{cp:04X}",
                "char": chr(cp) if cp >= 0x20 else "",
                "count": count,
                "name": name,
                "block": _block_name(cp),
            }
        )

    return {
        "n": n,
        "label_length_chars": _quantiles([float(x) for x in label_char_lens]),
        "label_length_words": _quantiles([float(x) for x in label_word_lens]),
        "image_width_px": _quantiles([float(x) for x in widths]),
        "image_height_px": _quantiles([float(x) for x in heights]),
        "n_with_image_dims": len(dims),
        "block_histogram": dict(sorted(block_counts.items(), key=lambda kv: -kv[1])),
        "top_offending_codepoints": top_offenders,
        "sample_offending_texts": samples_with_offending,
        "would_drop": {
            "label_too_short": {
                "threshold_min_len": min_len,
                "count": len(dropped_len_short),
                "examples": dropped_len_short[:5],
            },
            "label_too_long": {
                "threshold_max_len": max_len,
                "count": len(dropped_len_long),
                "examples": [t[:80] + "..." for t in dropped_len_long[:5]],
            },
            "image_too_tall": {
                "threshold_max_height_px": max_height,
                "count": len(dropped_height),
                "example_heights": dropped_height[:10],
            },
            "non_yoruba_codepoint": {
                "count": dropped_nonyor_samples,
            },
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Audit data/processed for hygiene issues (read-only)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Consolidated dataset root.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/tables/data_quality.json"),
        help="Where to write the audit report.",
    )
    parser.add_argument(
        "--min-label-len",
        type=int,
        default=3,
        help="Minimum accepted label length in characters (strict: < this is dropped).",
    )
    parser.add_argument(
        "--max-label-len",
        type=int,
        default=100,
        help="Maximum accepted label length in characters (strict: > this is dropped).",
    )
    parser.add_argument(
        "--max-image-height",
        type=int,
        default=180,
        help="Maximum accepted image height in pixels (> this is treated as multi-line noise).",
    )
    return parser.parse_args()


def main() -> None:
    """Profile each split and write the consolidated JSON report."""
    args = parse_args()
    if not args.data_dir.exists():
        raise SystemExit(f"data-dir does not exist: {args.data_dir}")

    per_split = {}
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split in SPLITS:
        log.info("Profiling %s ...", split)
        profile = profile_split(
            args.data_dir,
            split,
            min_len=args.min_label_len,
            max_len=args.max_label_len,
            max_height=args.max_image_height,
        )
        per_split[split] = profile
        if profile.get("n"):
            for key, payload in profile["would_drop"].items():
                totals["would_drop"][key] += int(payload.get("count", 0))
                totals["n"][split] = profile["n"]

    report = {
        "data_dir": str(args.data_dir),
        "thresholds": {
            "min_label_len": args.min_label_len,
            "max_label_len": args.max_label_len,
            "max_image_height": args.max_image_height,
        },
        "per_split": per_split,
        "totals": {k: dict(v) for k, v in totals.items()},
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log.info("Audit written to %s", args.out_json)

    # Compact human-readable summary
    for split, prof in per_split.items():
        if not prof.get("n"):
            continue
        wd = prof["would_drop"]
        log.info(
            "  %-5s n=%d  would_drop short=%d long=%d tall=%d nonyor=%d",
            split,
            prof["n"],
            wd["label_too_short"]["count"],
            wd["label_too_long"]["count"],
            wd["image_too_tall"]["count"],
            wd["non_yoruba_codepoint"]["count"],
        )


if __name__ == "__main__":
    main()
