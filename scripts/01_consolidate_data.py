"""
Consolidate all raw dataset exports into a single clean data/processed/ tree.

Each export under data/raw/yoruba_ocr_NNNNN[*] is an independent annotation
batch from the Yorùbá di Wúrà annotation platform. This script:

  1. Scans all export directories and reads train/val/test label files.
  2. Deduplicates entries by image filename (last export wins on conflict).
  3. Copies unique PNG crops to data/processed/images/{train,val,test}/.
  4. Writes merged label files to data/processed/labels/.
  5. Merges character dictionaries (union, sorted) to
     data/processed/dictionary/yoruba_char_dict.txt.
  6. Saves a JSON consolidation report to results/tables/.

Usage:
    python scripts/01_consolidate_data.py
    python scripts/01_consolidate_data.py --raw-dir data/raw \
        --output-dir data/processed --log-file results/tables/consolidation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
# Match "yoruba_ocr_NNNNN" and "yoruba_ocr_NNNNN (Unzipped Files)"
EXPORT_PATTERN = re.compile(r"^yoruba_ocr_(\d+)")

# Single source of truth for the Yorùbá Unicode whitelist — see
# scripts/yoruba_charset.py. We import the helper here instead of
# re-defining it so the hygiene filter and the audit tool can never
# drift apart.
# Support both "python scripts/01_consolidate_data.py" and module-style
# invocation from the repo root.
import sys as _sys

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPTS_DIR))
from yoruba_charset import has_only_whitelisted_codepoints  # noqa: E402


def find_export_dirs(raw_dir: Path) -> list[tuple[int, Path]]:
    """Return (export_id, path) pairs sorted ascending by export_id."""
    exports = []
    for item in raw_dir.iterdir():
        if not item.is_dir():
            continue
        m = EXPORT_PATTERN.match(item.name)
        if m:
            exports.append((int(m.group(1)), item))
    return sorted(exports, key=lambda x: x[0])


def read_label_file(label_path: Path) -> list[tuple[str, str]]:
    """
    Parse a PaddleOCR label file.

    Each valid line: relative_image_path<TAB>ground_truth_text
    Returns list of (relative_path, nfc_text) tuples.
    """
    entries = []
    with label_path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                log.warning("Skipping malformed line in %s: %r", label_path, line[:80])
                continue
            rel_path, text = parts
            entries.append((rel_path, unicodedata.normalize("NFC", text)))
    return entries


def is_valid_yoruba(text: str) -> bool:
    """Check if the text is clean, valid Yoruba and not English code-mixed or UI artifacts."""
    # 1. No English-only consonants (c, q, v, x, z)
    if re.search(r"[cqvxz]", text, re.IGNORECASE):
        return False

    # 2. No fill-in-the-blank dots or underscores
    if re.search(r"\.{3,}", text) or re.search(r"_{3,}", text):
        return False

    # 3. No hanging equals signs often used for translation matching '... = ...'
    if "=" in text:
        return False

    # 4. No English months or Roman numerals used in list numbering
    months_roman = r"\b(january|february|march|april|may|june|july|august|september|october|november|december|iv|vi|vii|viii|ix)\b"
    if re.search(months_roman, text, re.IGNORECASE):
        return False

    # 5. Generic bad English tokens found in the data or common stopwords
    bad_words = r"\b(chest|child|breeze|chair|cup|bicycle|village|zebra|fox|camel|the|and|of|to|in|is|was|for|on|are|with)\b"
    if re.search(bad_words, text, re.IGNORECASE):
        return False

    return True


def _get_image_height(path: Path) -> int | None:
    """Return image height in pixels, or ``None`` if Pillow is unavailable or
    the image cannot be opened."""
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            return int(im.size[1])
    except (OSError, FileNotFoundError):
        return None


def collect_registry(
    exports: list[tuple[int, Path]],
    *,
    hygiene: bool,
    min_label_len: int,
    max_label_len: int,
    max_image_height: int,
    strict_charset: bool,
) -> tuple[dict[str, dict], dict[str, int]]:
    """
    Build a registry keyed by image stem (filename without directory).

    When the same stem appears in multiple exports, the entry from the
    numerically highest export_id takes precedence — reflecting the most
    recent annotation decision.

    When ``hygiene=True`` (default), samples are additionally dropped for:

    * label length outside ``[min_label_len, max_label_len]``
    * image height ``> max_image_height`` (treated as multi-line noise)
    * any codepoint outside the Yorùbá whitelist (only when
      ``strict_charset=True``)

    Returns ``(registry, drop_counts)`` where ``drop_counts`` tracks the
    number of entries eliminated by each filter so the consolidation
    report can make the decisions auditable.
    """
    registry: dict[str, dict] = {}
    drops: dict[str, int] = {
        "missing_image": 0,
        "invalid_yoruba": 0,
        "label_too_short": 0,
        "label_too_long": 0,
        "image_too_tall": 0,
        "non_whitelisted_codepoint": 0,
    }

    # Cache image heights so duplicates across exports don't re-read the file.
    height_cache: dict[Path, int | None] = {}

    for export_id, export_dir in exports:
        for split in SPLITS:
            label_file = export_dir / "labels" / f"{split}.txt"
            if not label_file.exists():
                continue
            for rel_path, text in read_label_file(label_file):
                src_path = export_dir / rel_path
                if not src_path.exists():
                    drops["missing_image"] += 1
                    log.debug("Image missing: %s", src_path)
                    continue
                stem = Path(rel_path).name

                if not is_valid_yoruba(text):
                    drops["invalid_yoruba"] += 1
                    continue

                if hygiene:
                    if len(text) < min_label_len:
                        drops["label_too_short"] += 1
                        continue
                    if len(text) > max_label_len:
                        drops["label_too_long"] += 1
                        continue
                    if strict_charset and not has_only_whitelisted_codepoints(text):
                        drops["non_whitelisted_codepoint"] += 1
                        continue
                    if max_image_height > 0:
                        if src_path not in height_cache:
                            height_cache[src_path] = _get_image_height(src_path)
                        h = height_cache[src_path]
                        if h is not None and h > max_image_height:
                            drops["image_too_tall"] += 1
                            continue

                if stem not in registry or export_id > registry[stem]["export_id"]:
                    registry[stem] = {
                        "text": text,
                        "src_path": src_path,
                        "split": split,
                        "export_id": export_id,
                    }

    if drops["missing_image"]:
        log.warning(
            "Skipped %d entries with missing image files.", drops["missing_image"]
        )
    return registry, drops


def collect_char_dicts_from_registry(registry: dict[str, dict]) -> list[str]:
    """
    Build the character dictionary dynamically from all text in the consolidated dataset.

    This ensures that *every* character present in the labels (including any missing
    diacritics or special characters like 'ố' and 'Ở') is included in the dictionary.
    We explicitly remove the space character (' ') because PaddleOCR's CTCLabelEncode
    handles it internally when use_space_char=True is set.
    """
    chars: set[str] = set()
    for entry in registry.values():
        text = entry["text"]
        for ch in text:
            chars.add(ch)

    # Explicitly remove space to avoid index shifting with use_space_char=True
    if " " in chars:
        chars.remove(" ")

    return sorted(chars, key=lambda c: (ord(c[0]), c))


def copy_images_and_write_labels(
    registry: dict[str, dict],
    output_dir: Path,
) -> dict[str, int]:
    """
    Copy images to output_dir/images/{split}/ and write label txt files.

    Returns per-split sample counts.
    """
    counts: dict[str, int] = {s: 0 for s in SPLITS}
    label_handles: dict[str, object] = {}
    label_dir = output_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        label_handles[split] = (label_dir / f"{split}.txt").open("w", encoding="utf-8")

    try:
        for stem, entry in sorted(registry.items()):
            split = entry["split"]
            dst = output_dir / "images" / split / stem
            shutil.copy2(entry["src_path"], dst)
            rel = f"images/{split}/{stem}"
            label_handles[split].write(f"{rel}\t{entry['text']}\n")
            counts[split] += 1
    finally:
        for fh in label_handles.values():
            fh.close()  # type: ignore[union-attr]

    return counts


def write_char_dict(chars: list[str], output_dir: Path) -> Path:
    """Write the merged character dictionary and return its path."""
    dict_dir = output_dir / "dictionary"
    dict_dir.mkdir(parents=True, exist_ok=True)
    dict_path = dict_dir / "yoruba_char_dict.txt"
    with dict_path.open("w", encoding="utf-8") as fh:
        for ch in chars:
            fh.write(ch + "\n")
    return dict_path


def save_report(
    report: dict,
    log_file: Path,
) -> None:
    """Persist the consolidation report as JSON."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Consolidate all raw Yorùbá OCR exports into data/processed/."
    )
    parser.add_argument(
        "--raw-dir",
        "--raw_dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing all yoruba_ocr_NNNNN export folders.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        "--out_dir",
        dest="output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Destination directory for consolidated dataset.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("results/tables/consolidation_report.json"),
        help="JSON file to record consolidation statistics.",
    )
    parser.add_argument(
        "--no-hygiene",
        dest="hygiene",
        action="store_false",
        help="Disable length/height/charset filters (keeps legacy behaviour).",
    )
    parser.set_defaults(hygiene=True)
    parser.add_argument(
        "--min-label-len",
        type=int,
        default=3,
        help="Minimum label length in characters when hygiene is enabled.",
    )
    parser.add_argument(
        "--max-label-len",
        type=int,
        default=100,
        help="Maximum label length in characters when hygiene is enabled.",
    )
    parser.add_argument(
        "--max-image-height",
        type=int,
        default=180,
        help=(
            "Maximum accepted image height (px). Taller images are treated as "
            "multi-line noise. Set to 0 to disable the height filter."
        ),
    )
    parser.add_argument(
        "--no-strict-charset",
        dest="strict_charset",
        action="store_false",
        help="Disable the Yorùbá Unicode whitelist (keeps annotation noise).",
    )
    parser.set_defaults(strict_charset=True)
    return parser.parse_args()


def main() -> None:
    """Run the full consolidation pipeline."""
    args = parse_args()

    log.info("Scanning exports in %s ...", args.raw_dir)
    exports = find_export_dirs(args.raw_dir)
    log.info("Found %d export directories.", len(exports))

    log.info(
        "Building image registry (hygiene=%s, min=%d, max=%d, max_h=%d, strict_charset=%s) ...",
        args.hygiene,
        args.min_label_len,
        args.max_label_len,
        args.max_image_height,
        args.strict_charset,
    )
    registry, drops = collect_registry(
        exports,
        hygiene=args.hygiene,
        min_label_len=args.min_label_len,
        max_label_len=args.max_label_len,
        max_image_height=args.max_image_height,
        strict_charset=args.strict_charset,
    )
    log.info("Unique images after deduplication: %d", len(registry))
    for k, v in drops.items():
        if v:
            log.info("  dropped %s: %d", k, v)

    log.info("Collecting character dictionaries ...")
    chars = collect_char_dicts_from_registry(registry)
    log.info("Merged character set size: %d", len(chars))

    log.info("Copying images and writing label files ...")
    counts = copy_images_and_write_labels(registry, args.output_dir)

    dict_path = write_char_dict(chars, args.output_dir)
    log.info("Character dictionary written to %s", dict_path)

    report = {
        "stage": "consolidate",
        "n_exports": len(exports),
        "export_ids": [eid for eid, _ in exports],
        "unique_images_total": len(registry),
        "split_counts": counts,
        "char_dict_size": len(chars),
        "output_dir": str(args.output_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hygiene": {
            "enabled": args.hygiene,
            "min_label_len": args.min_label_len,
            "max_label_len": args.max_label_len,
            "max_image_height": args.max_image_height,
            "strict_charset": args.strict_charset,
            "drop_counts": drops,
        },
    }
    save_report(report, args.log_file)

    log.info(
        "Done. Train: %d  Val: %d  Test: %d",
        counts["train"],
        counts["val"],
        counts["test"],
    )
    log.info("Report saved to %s", args.log_file)


if __name__ == "__main__":
    main()
