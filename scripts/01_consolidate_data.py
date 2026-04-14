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


def collect_registry(
    exports: list[tuple[int, Path]],
) -> dict[str, dict]:
    """
    Build a registry keyed by image stem (filename without directory).

    When the same stem appears in multiple exports, the entry from the
    numerically highest export_id takes precedence — reflecting the most
    recent annotation decision.

    Returns:
        {
          "0a785293_line0001.png": {
              "text": "Ẹ̀KỌ́ KẸTÀDÍNLÓGÚN",
              "src_path": Path(...),
              "split": "train",
              "export_id": 85456,
          },
          ...
        }
    """
    registry: dict[str, dict] = {}
    skipped_missing = 0

    for export_id, export_dir in exports:
        for split in SPLITS:
            label_file = export_dir / "labels" / f"{split}.txt"
            if not label_file.exists():
                continue
            for rel_path, text in read_label_file(label_file):
                src_path = export_dir / rel_path
                if not src_path.exists():
                    skipped_missing += 1
                    log.debug("Image missing: %s", src_path)
                    continue
                stem = Path(rel_path).name
                if stem not in registry or export_id > registry[stem]["export_id"]:
                    registry[stem] = {
                        "text": text,
                        "src_path": src_path,
                        "split": split,
                        "export_id": export_id,
                    }

    if skipped_missing:
        log.warning("Skipped %d entries with missing image files.", skipped_missing)
    return registry


def collect_char_dicts(exports: list[tuple[int, Path]]) -> list[str]:
    """
    Merge all per-export character dictionaries into a sorted union list.

    Each dict file has one character per line (may include a leading space char).
    """
    chars: set[str] = set()
    for _, export_dir in exports:
        for dict_file in (export_dir / "dictionary").glob("rec_char_dict*.txt"):
            with dict_file.open(encoding="utf-8") as fh:
                for line in fh:
                    ch = line.rstrip("\n")
                    if ch:  # preserve space char by checking length, not truthiness
                        chars.add(ch)
                    elif line == "\n":
                        # A line containing only newline = the space character entry
                        chars.add(" ")
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
        label_handles[split] = (label_dir / f"{split}.txt").open(
            "w", encoding="utf-8"
        )

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
    return parser.parse_args()


def main() -> None:
    """Run the full consolidation pipeline."""
    args = parse_args()

    log.info("Scanning exports in %s ...", args.raw_dir)
    exports = find_export_dirs(args.raw_dir)
    log.info("Found %d export directories.", len(exports))

    log.info("Building image registry ...")
    registry = collect_registry(exports)
    log.info("Unique images after deduplication: %d", len(registry))

    log.info("Collecting character dictionaries ...")
    chars = collect_char_dicts(exports)
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
