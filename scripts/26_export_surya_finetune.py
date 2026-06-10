"""
Export Yorùbá line crops for Surya Foundation fine-tuning (``image`` + ``text``).

Surya's ``finetune_ocr.py`` (datalab-to/surya v0.15.x) expects a Hugging Face
dataset with at least a ``train`` split and columns ``image`` (PIL/bytes) and
``text`` (UTF-8 NFC string). By default we merge ``train`` + ``val`` for
supervised fine-tuning and hold out ``test`` for evaluation only.

Usage:
    python scripts/26_export_surya_finetune.py
    python scripts/26_export_surya_finetune.py --push --repo-id USER/yoruba-ocr-surya-train
"""

from __future__ import annotations

import argparse
import json
import logging
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_EXPORT_DIR = Path("data/hf_surya_finetune")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Export train+val line crops for Surya OCR fine-tuning."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Consolidated dataset root.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help="Local ``save_to_disk`` target.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Optional Hub dataset repo for ``--push``.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload to Hugging Face (requires HF_TOKEN and --repo-id).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update Hub repo as private.",
    )
    parser.add_argument(
        "--train-splits",
        type=str,
        default="train,val",
        help="Comma-separated label splits merged into HF ``train``.",
    )
    return parser.parse_args()


def read_label_split(data_dir: Path, split: str) -> list[tuple[Path, str]]:
    """Parse ``labels/{split}.txt`` into absolute image path + NFC text."""
    label_file = data_dir / "labels" / f"{split}.txt"
    if not label_file.is_file():
        raise FileNotFoundError(f"Missing label file: {label_file}")
    rows: list[tuple[Path, str]] = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        rel_path, text = parts
        img_path = data_dir / rel_path
        rows.append((img_path, unicodedata.normalize("NFC", text)))
    return rows


def build_rows(data_dir: Path, train_splits: list[str]) -> list[dict]:
    """Load image/text pairs from one or more PaddleOCR label splits."""
    rows: list[dict] = []
    for split in train_splits:
        for img_path, text in read_label_split(data_dir, split):
            if not img_path.is_file():
                raise FileNotFoundError(f"Missing image for {split}: {img_path}")
            rows.append({"image": str(img_path), "text": text, "source_split": split})
    return rows


def export_dataset(rows: list[dict], export_dir: Path):
    """Write a HF Dataset with ``train`` split to ``export_dir``."""
    from datasets import Dataset, DatasetDict, Image as HFImage  # type: ignore

    ds = Dataset.from_list(rows)
    ds = ds.cast_column("image", HFImage())
    dsd = DatasetDict({"train": ds})
    export_dir.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(export_dir))
    log.info("Saved %d train rows to %s", len(rows), export_dir)
    return dsd


def write_manifest(export_dir: Path, n_rows: int, train_splits: list[str]) -> Path:
    """Record export metadata under ``results/tables/``."""
    manifest = {
        "export_dir": str(export_dir),
        "n_train": n_rows,
        "train_splits": train_splits,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = Path("results/tables/surya_finetune_export.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def push_to_hub(dsd, repo_id: str, private: bool) -> None:
    """Push ``DatasetDict`` to the Hub."""
    dsd.push_to_hub(repo_id, private=private)
    log.info("Pushed Surya fine-tune dataset to %s", repo_id)


def main() -> None:
    """Export merged train split for Surya fine-tuning."""
    args = parse_args()
    train_splits = [s.strip() for s in args.train_splits.split(",") if s.strip()]
    rows = build_rows(args.data_dir, train_splits)
    if not rows:
        raise RuntimeError("No training rows exported.")

    try:
        from datasets import DatasetDict  # type: ignore
    except ImportError as exc:
        raise ImportError("Run: pip install datasets") from exc

    dsd = export_dataset(rows, args.export_dir)
    manifest = write_manifest(args.export_dir, len(rows), train_splits)
    log.info("Manifest: %s", manifest)

    if args.push:
        if not args.repo_id:
            raise ValueError("--push requires --repo-id")
        push_to_hub(dsd, args.repo_id, args.private)


if __name__ == "__main__":
    main()
