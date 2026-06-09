"""
Archive and reset evaluation artifacts between Colab re-runs.

Copies ``metrics.csv``, Table 1 outputs, per-model JSONL logs, and meta JSON
to ``results/tables/archive/pre_run_<timestamp>/``, then resets ``metrics.csv``
to header-only and removes working JSONL, meta, and compiled table snapshots
so partial re-runs cannot mix stale and fresh inference logs.

Usage:
    python scripts/metrics_lifecycle.py archive
    python scripts/metrics_lifecycle.py archive --tables-dir results/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

METRICS_HEADER = [
    "model",
    "split",
    "n",
    "cer",
    "wer",
    "der",
    "der_n",
    "der_insertion_rate",
    "phantom",
    "meta_path",
    "timestamp",
]

TABLE_SNAPSHOTS = (
    "metrics.csv",
    "metrics_summary.csv",
    "table1_main_comparison.csv",
    "table1_main_comparison.md",
    "bootstrap_metric_cis.csv",
    "bootstrap_pairwise_comparison.csv",
    "stratified_der_by_density.csv",
    "stratified_der_by_book.csv",
    "der_universe_ablation.csv",
    "eval_alignment_report.json",
    "checkpoint_audit.json",
)


def checkpoint_status_for_row(row: dict, tables_dir: Path) -> str:
    """
    Return checkpoint audit status for one metrics.csv row.

    Mirrors ``12_diagnose_hypotheses.py checkpoints`` logic:
    ``ok`` | ``phantom`` | ``stale`` | ``no_meta``.
    """
    model = row.get("model", "")
    split = row.get("split", "")
    csv_phantom = (row.get("phantom") or "").strip().lower()
    meta_path = tables_dir / "meta" / f"{model}_{split}.json"

    if not meta_path.is_file():
        return "phantom" if csv_phantom == "true" else "no_meta"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "no_meta"

    if str(meta.get("phantom", "")).lower() == "true":
        return "phantom"

    ckpt_path = (meta.get("artifacts") or {}).get("checkpoint_pdparams")
    if ckpt_path and not Path(ckpt_path).is_file():
        return "stale"

    return "ok"


def archive_eval_artifacts(tables_dir: Path) -> Path:
    """
    Copy current metrics/tables/jsonl/meta to archive and reset metrics.csv.

    Returns the archive directory path.
    """
    tables_dir = tables_dir.resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = tables_dir / "archive" / f"pre_run_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)

    for name in TABLE_SNAPSHOTS:
        src = tables_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)

    jsonl_archived: list[Path] = []
    for pattern in ("*_test.jsonl", "*_val.jsonl", "*_train.jsonl"):
        for src in sorted(tables_dir.glob(pattern)):
            shutil.copy2(src, dest / src.name)
            jsonl_archived.append(src)

    meta_src = tables_dir / "meta"
    meta_archived: list[Path] = []
    if meta_src.is_dir():
        meta_dest = dest / "meta"
        meta_dest.mkdir(parents=True, exist_ok=True)
        for src in meta_src.glob("*.json"):
            shutil.copy2(src, meta_dest / src.name)
            meta_archived.append(src)

    tables_removed: list[Path] = []
    for name in TABLE_SNAPSHOTS:
        src = tables_dir / name
        if src.is_file() and name != "metrics.csv":
            src.unlink()
            tables_removed.append(src)
    if tables_removed:
        log.info(
            "Removed %d compiled table snapshot(s) from %s (re-run compile/analysis after eval).",
            len(tables_removed),
            tables_dir,
        )

    metrics = tables_dir / "metrics.csv"
    if metrics.is_file():
        shutil.copy2(metrics, dest / "metrics.csv")
        with metrics.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=METRICS_HEADER)
            writer.writeheader()
        log.info("Reset %s to header-only.", metrics)

    for path in jsonl_archived:
        path.unlink()
    if jsonl_archived:
        log.info("Removed %d JSONL log(s) from %s (copied to archive).", len(jsonl_archived), tables_dir)

    for path in meta_archived:
        path.unlink()
    if meta_archived:
        log.info("Removed %d meta JSON file(s) from %s.", len(meta_archived), meta_src)

    marker = tables_dir / ".last_metrics_archive.txt"
    marker.write_text(str(dest) + "\n", encoding="utf-8")
    log.info("Archived eval artifacts to %s", dest)
    return dest


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Archive and reset metrics between runs.")
    parser.add_argument(
        "command",
        choices=("archive",),
        help="Currently supported: archive.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results/tables"),
        help="Results tables directory.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if args.command == "archive":
        dest = archive_eval_artifacts(args.tables_dir)
        print(dest)


if __name__ == "__main__":
    main()
