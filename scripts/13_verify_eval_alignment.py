"""
Compare metrics.csv ``n`` column to the current dataset label files.

If stored evaluation used a different test/val size than ``load_test_pairs`` sees
today, tables in the paper can disagree with the repo — this script surfaces that.

Writes ``results/tables/eval_alignment_report.json`` with expected vs reported counts.

Usage:
    python scripts/13_verify_eval_alignment.py
    python scripts/13_verify_eval_alignment.py --strict  # exit 1 on any mismatch
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Verify metrics.csv sample counts match label files on disk."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Consolidated dataset root (images + labels/).",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("results/tables/metrics.csv"),
        help="Aggregated metrics CSV from eval scripts.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/tables/eval_alignment_report.json"),
        help="Where to write the alignment report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any split shows n != expected_pairs.",
    )
    return parser.parse_args()


def expected_pair_count(data_dir: Path, split: str) -> int:
    """Return the number of (image, gt) pairs with existing files on disk."""
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_utils import load_test_pairs  # noqa: E402

    return len(load_test_pairs(data_dir, split))


def load_metrics_rows(csv_path: Path) -> list[dict]:
    """Load all rows from metrics.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {csv_path}")
    import csv

    with csv_path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    """Compare reported ``n`` to label-file pair counts per split."""
    args = parse_args()
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "metrics_csv": str(args.metrics_csv.resolve()),
        "splits": {},
        "mismatches": [],
    }

    rows = load_metrics_rows(args.metrics_csv)
    splits_in_metrics = sorted({r.get("split", "test") for r in rows})

    for split in splits_in_metrics:
        if split not in ("train", "val", "test"):
            continue
        try:
            exp = expected_pair_count(args.data_dir, split)
        except FileNotFoundError as exc:
            log.warning("%s", exc)
            report["splits"][split] = {"expected_pairs": None, "error": str(exc)}
            continue
        report["splits"][split] = {"expected_pairs": exp}

    for row in rows:
        split = row.get("split", "test")
        if (
            split not in report["splits"]
            or report["splits"][split].get("expected_pairs") is None
        ):
            continue
        exp = report["splits"][split]["expected_pairs"]
        try:
            n = int(row.get("n", "") or 0)
        except ValueError:
            continue
        if n != exp:
            report["mismatches"].append(
                {
                    "model": row.get("model"),
                    "split": split,
                    "reported_n": n,
                    "expected_pairs": exp,
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Wrote %s", args.output_json)

    if not report["mismatches"]:
        log.info("Alignment OK — all reported n values match current label files.")
        sys.exit(0)

    for m in report["mismatches"][:10]:
        log.warning(
            "Mismatch: model=%s split=%s reported_n=%s expected=%s",
            m["model"],
            m["split"],
            m["reported_n"],
            m["expected_pairs"],
        )
    if len(report["mismatches"]) > 10:
        log.warning("... and %d more (see JSON).", len(report["mismatches"]) - 10)

    log.warning(
        "Re-run evaluation scripts on the current processed split, or document the "
        "exact export used for published numbers."
    )
    sys.exit(1 if args.strict else 0)


if __name__ == "__main__":
    main()
