"""Check whether a model has complete, failure-free evidence for one split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def is_complete(model: str, split: str, tables_dir: Path, data_dir: Path) -> bool:
    """Return True when metrics, JSONL, metadata, and dataset counts align."""
    metrics_path = tables_dir / "metrics.csv"
    label_path = data_dir / "labels" / f"{split}.txt"
    jsonl_path = tables_dir / f"{model}_{split}.jsonl"
    meta_path = tables_dir / "meta" / f"{model}_{split}.json"
    if not all(path.is_file() for path in (metrics_path, label_path, jsonl_path, meta_path)):
        return False

    expected = 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2 and (data_dir / parts[0]).is_file():
            expected += 1
    jsonl_count = sum(1 for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if expected == 0 or jsonl_count != expected:
        return False

    with metrics_path.open(encoding="utf-8", newline="") as fh:
        rows = [
            row for row in csv.DictReader(fh)
            if row.get("model") == model and row.get("split") == split
        ]
    if not rows:
        return False
    try:
        if int(rows[-1].get("n") or 0) != expected:
            return False
    except (TypeError, ValueError):
        return False
    if str(rows[-1].get("phantom", "")).lower() in {"true", "1"}:
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    provenance = meta.get("provenance") or {}
    if str(meta.get("phantom", "")).lower() in {"true", "1"}:
        return False
    try:
        return int(provenance.get("failure_count", 0) or 0) == 0
    except (TypeError, ValueError):
        return False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Check completed OCR model evidence.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    """Exit zero only when the requested row is complete and citable."""
    args = parse_args()
    complete = is_complete(args.model, args.split, args.tables_dir, args.data_dir)
    print(f"{args.model}/{args.split}: {'complete' if complete else 'pending'}")
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
