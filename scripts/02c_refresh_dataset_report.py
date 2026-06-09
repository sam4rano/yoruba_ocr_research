"""
Refresh split counts in consolidation_report.json from on-disk label files.

Use when ``SKIP_CONSOLIDATE=1`` but you need accurate counts in
``research_approach.md`` and dataset cards.

Usage:
    python scripts/02c_refresh_dataset_report.py
    python scripts/02c_refresh_dataset_report.py --data-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Update consolidation_report.json split counts from label files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Processed dataset root.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/tables/consolidation_report.json"),
        help="Consolidation report JSON to update.",
    )
    return parser.parse_args()


def count_split(data_dir: Path, split: str) -> int:
    """Return non-empty line count for one label file."""
    path = data_dir / "labels" / f"{split}.txt"
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def char_dict_size(data_dir: Path) -> int:
    """Return grapheme dictionary line count."""
    path = data_dir / "dictionary" / "yoruba_char_dict.txt"
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    """Refresh report split counts from disk."""
    args = parse_args()
    split_counts = {split: count_split(args.data_dir, split) for split in SPLITS}
    total = sum(split_counts.values())
    char_n = char_dict_size(args.data_dir)

    report: dict = {}
    if args.report.is_file():
        report = json.loads(args.report.read_text(encoding="utf-8"))

    report.update(
        {
            "unique_images_total": total,
            "split_counts": split_counts,
            "char_dict_size": char_n,
            "output_dir": str(args.data_dir),
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "split_source": "on_disk_labels",
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info(
        "Updated %s — train=%d val=%d test=%d (dict=%d chars)",
        args.report,
        split_counts["train"],
        split_counts["val"],
        split_counts["test"],
        char_n,
    )


if __name__ == "__main__":
    main()
