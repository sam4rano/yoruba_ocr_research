"""
Build and upload the Yorùbá OCR line-crop benchmark to the Hugging Face Hub.

Creates a ``DatasetDict`` with PNG line crops, NFC-normalised transcriptions,
and book-level metadata, writes a dataset card (README.md), and optionally
pushes to Hugging Face.

Usage:
    python scripts/publish_hf_dataset.py --dry-run
    python scripts/publish_hf_dataset.py --repo-id USER/yoruba-ocr-line-crops --push
    python scripts/publish_hf_dataset.py --export-dir data/hf_export --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
EXPORT_PATTERN = re.compile(r"^yoruba_ocr_(\d+)")

DATASET_AUTHORS = (
    ("Moses Oyedele", "Yorùbá di Wúrà source material (Books 1–6)"),
    ("Damilare Oyedele", "Yorùbá di Wúrà source material (Books 1–6)"),
    ("Samuel Oyerinde", "Annotation platform, dataset consolidation, and OCR benchmark"),
)

DEFAULT_LICENSE = "cc-by-4.0"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Export data/processed to Hugging Face Datasets and upload."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Consolidated dataset root (images + labels).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw export root for dataset_metadata.csv lookup.",
    )
    parser.add_argument(
        "--consolidation-report",
        type=Path,
        default=Path("results/tables/consolidation_report.json"),
        help="Optional consolidation JSON for dataset card stats.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hub dataset repo, e.g. Sam4rano/yoruba-ocr-line-crops.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update the Hub repo as private.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload to Hugging Face (requires HF_TOKEN and --repo-id).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build locally and write dataset card; do not upload.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional local folder for README + manifest (no Hub upload).",
    )
    parser.add_argument(
        "--license",
        default=DEFAULT_LICENSE,
        help="SPDX license id for dataset card YAML (default: cc-by-4.0).",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload Yorùbá OCR line-crop benchmark",
        help="Hub commit message when --push is set.",
    )
    parser.add_argument(
        "--upload-mode",
        choices=("auto", "push_to_hub", "large_folder"),
        default="auto",
        help=(
            "Hub upload strategy: datasets push_to_hub, resumable large_folder, "
            "or auto (large_folder when >2000 examples)."
        ),
    )
    return parser.parse_args()


def load_metadata_index(raw_dir: Path) -> dict[str, dict[str, str]]:
    """
    Build ``{image_basename: metadata}`` from raw export CSVs.

    Later exports overwrite earlier ones on filename collision, matching
    consolidation policy in ``consolidate_data.py``.
    """
    index: dict[str, dict[str, str]] = {}

    # 1. Fallback to consolidated splits/dataset_metadata.csv if it exists
    splits_meta = Path("data/splits/dataset_metadata.csv")
    if splits_meta.is_file():
        try:
            with splits_meta.open(encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    image = (row.get("image") or "").strip()
                    if image:
                        index[image] = {
                            "source": (row.get("source") or "").strip(),
                            "year": (row.get("year") or "").strip(),
                            "dialect": (row.get("dialect") or "").strip(),
                            "corrected": (row.get("corrected") or "").strip(),
                            "parent_image": (row.get("parent_image") or "").strip(),
                            "line_index": (row.get("line_index") or "").strip(),
                        }
            log.info("Loaded metadata for %d line images from consolidated splits/dataset_metadata.csv", len(index))
            return index
        except Exception as e:
            log.warning("Failed to load consolidated metadata from %s: %s", splits_meta, e)

    # 2. Fall back to scanning raw exports
    if not raw_dir.is_dir():
        log.warning("Raw dir missing: %s — metadata columns will be empty.", raw_dir)
        return index

    for item in sorted(raw_dir.iterdir()):
        if not item.is_dir() or not EXPORT_PATTERN.match(item.name):
            continue
        meta_path = item / "metadata" / "dataset_metadata.csv"
        if not meta_path.is_file():
            alt = item / "metadata" / "dataset_metadata(1).csv"
            meta_path = alt if alt.is_file() else meta_path
        if not meta_path.is_file():
            continue
        with meta_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                image = (row.get("image") or "").strip()
                if image:
                    index[image] = {
                        "source": (row.get("source") or "").strip(),
                        "year": (row.get("year") or "").strip(),
                        "dialect": (row.get("dialect") or "").strip(),
                        "corrected": (row.get("corrected") or "").strip(),
                        "parent_image": (row.get("parent_image") or "").strip(),
                        "line_index": (row.get("line_index") or "").strip(),
                    }
    log.info("Indexed metadata for %d line images.", len(index))
    return index


def read_label_split(data_dir: Path, split: str) -> list[tuple[str, str]]:
    """Parse ``labels/{split}.txt`` into (relative_image_path, nfc_text) pairs."""
    label_path = data_dir / "labels" / f"{split}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    rows: list[tuple[str, str]] = []
    with label_path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                log.warning("Skipping malformed line in %s", label_path)
                continue
            rel_path, text = parts
            rows.append((rel_path, unicodedata.normalize("NFC", text)))
    return rows


def build_rows(
    data_dir: Path,
    split: str,
    meta_index: dict[str, dict[str, str]],
) -> list[dict]:
    """Materialise one split as HF-ready dict rows."""
    rows: list[dict] = []
    for rel_path, text in read_label_split(data_dir, split):
        img_path = (data_dir / rel_path).resolve()
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing image for {split}: {img_path}")
        image_id = Path(rel_path).name
        meta = meta_index.get(image_id, {})
        line_index_raw = meta.get("line_index", "")
        try:
            line_index = int(line_index_raw) if line_index_raw != "" else -1
        except ValueError:
            line_index = -1
        rows.append(
            {
                "image": str(img_path),
                "text": text,
                "image_id": image_id,
                "split": split,
                "source": meta.get("source", ""),
                "year": meta.get("year", ""),
                "dialect": meta.get("dialect", ""),
                "corrected": meta.get("corrected", ""),
                "parent_image": meta.get("parent_image", ""),
                "line_index": line_index,
            }
        )
    return rows


def load_consolidation_stats(report_path: Path) -> dict:
    """Load consolidation report JSON if present."""
    if not report_path.is_file():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def char_dict_size(data_dir: Path) -> int:
    """Return character dictionary line count."""
    dict_path = data_dir / "dictionary" / "yoruba_char_dict.txt"
    if not dict_path.is_file():
        return 0
    return sum(1 for line in dict_path.read_text(encoding="utf-8").splitlines() if line.strip())


def render_license_text() -> str:
    """Return CC BY 4.0 attribution notice for the Hub LICENSE file."""
    author_names = ", ".join(name for name, _ in DATASET_AUTHORS)
    return f"""Creative Commons Attribution 4.0 International (CC BY 4.0)

Copyright (c) {datetime.now(timezone.utc).year} {author_names}

This dataset — line-crop images, transcriptions, metadata, and packaging — is
licensed under the Creative Commons Attribution 4.0 International License.

You are free to share and adapt the material for any purpose, including
commercial use, provided that you give appropriate credit, indicate if changes
were made, and do not apply legal terms or technological measures that legally
restrict others from doing anything the license permits.

Full license text: https://creativecommons.org/licenses/by/4.0/legalcode
Human-readable summary: https://creativecommons.org/licenses/by/4.0/

Suggested attribution:
  Yorùbá OCR Line Crops benchmark. Moses Oyedele, Damilare Oyedele, and
  Samuel Oyerinde. Hugging Face Datasets. CC BY 4.0.
"""


def render_dataset_card(
    *,
    repo_id: str,
    split_counts: dict[str, int],
    consolidation: dict,
    char_dict_n: int,
    license_id: str,
) -> str:
    """Return README.md with Hugging Face YAML front matter."""
    total = sum(split_counts.values())
    n_exports = consolidation.get("n_exports", "33")
    authors_md = "\n".join(
        f"- **{name}** — {role}" for name, role in DATASET_AUTHORS
    )
    license_label = "CC BY 4.0" if license_id == "cc-by-4.0" else license_id
    yaml_splits = "\n".join(
        f"  - name: {('validation' if split == 'val' else split)}\n"
        f"    num_examples: {split_counts[split]}"
        for split in SPLITS
        if split in split_counts
    )
    card = f"""---
language:
- yo
license: {license_id}
pretty_name: Yorùbá OCR Line Crops
size_categories:
- 1K<n<10K
task_categories:
- image-to-text
tags:
- ocr
- yoruba
- yorùbá
- low-resource
- diacritics
- african-languages
- line-level
dataset_info:
  features:
    - name: image
      dtype: image
    - name: text
      dtype: string
    - name: image_id
      dtype: string
    - name: split
      dtype: string
    - name: source
      dtype: string
    - name: year
      dtype: string
    - name: dialect
      dtype: string
    - name: corrected
      dtype: string
    - name: parent_image
      dtype: string
    - name: line_index
      dtype: int32
  splits:
{yaml_splits}
  download_size: unknown
  dataset_size: unknown
configs:
- config_name: default
  data_files:
    - split: train
      path: train-*
    - split: validation
      path: validation-*
    - split: test
      path: test-*
---

# Yorùbá OCR Line Crops

Line-level optical character recognition benchmark for **Standard Yorùbá** with
full tonal and sub-dot diacritics. Each example is a PNG crop from a printed
line of text plus a human-corrected UTF-8 NFC transcription.

## Authors

{authors_md}

## Source

- **Printed material:** *Yorùbá di Wúrà* graded reader series (Books 1–6, 2021)
  by Moses Oyedele and Damilare Oyedele.
- **Annotation & release:** Benchmark curated by Samuel Oyerinde via the
  [Yorùbá OCR Hub](https://yoruba-ocr.vercel.app); OCR-assisted hypotheses
  reviewed and corrected by fluent annotators.
- **Consolidation:** {n_exports} independent export batches merged and deduplicated
  (see project script ``consolidate_data.py``).

## Splits

| Split | Lines |
|-------|------:|
| train | {split_counts.get("train", 0)} |
| validation | {split_counts.get("val", 0)} |
| test | {split_counts.get("test", 0)} |
| **Total** | **{total}** |

Splits follow an 80/10/10 line-level partition (seed 42) over unique crops after
hygiene filters (label length 3–100, Yorùbá charset whitelist, invalid-entry removal).

## Fields

| Column | Description |
|--------|-------------|
| `image` | PNG line crop |
| `text` | Ground-truth transcription (NFC) |
| `image_id` | Stable filename (``{{hash}}_line{{NNNN}}.png``) |
| `source` | Book volume label (e.g. *Yorùbá di Wúrà Book Five*) |
| `year` | Publication year (2021) |
| `dialect` | ``Standard Yorùbá`` |
| `corrected` | Human correction flag (`yes` for all entries) |
| `parent_image` | Source page scan filename |
| `line_index` | Line index on parent page (-1 if unknown) |

Character dictionary size in the consolidated release: **{char_dict_n}** graphemes
(excluding space). See ``dictionary/yoruba_char_dict.txt`` in the repository.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
example = ds["train"][0]
print(example["text"])
example["image"].show()
```

PaddleOCR-format label files (``image<TAB>text``) live in the
[GitHub repository](https://github.com/sam4rano/yoruba_ocr_research) under
``data/processed/labels/``.

## Metrics

Benchmark models with **CER**, **WER**, and **DER** (Diacritic Error Rate) as
defined in the Deep Learning Indaba 2026 paper accompanying this release.
Evaluation scripts: ``scripts/evaluate_paddleocr_recognition.py``.

## License and attribution

This dataset is released under the
[Creative Commons Attribution 4.0 International License ({license_label})](https://creativecommons.org/licenses/by/4.0/).

You may share and adapt the data provided you credit **Moses Oyedele**,
**Damilare Oyedele**, and **Samuel Oyerinde**, link to the license, and note
any changes. See ``LICENSE`` in this repository for the full notice.

If you use this dataset, cite the Yorùbá OCR benchmark paper (Deep Learning
Indaba 2026) and the *Yorùbá di Wúrà* series [CITE: oyedele2020ydw].

## Repository

- Code & tables: [sam4rano/yoruba_ocr_research](https://github.com/sam4rano/yoruba_ocr_research)
- Hub dataset: `{repo_id}`
"""
    return card


def build_dataset_dict(data_dir: Path, raw_dir: Path):
    """Build a Hugging Face ``DatasetDict`` from processed data."""
    from datasets import Dataset, DatasetDict, Features, Image, Value

    meta_index = load_metadata_index(raw_dir)
    features = Features(
        {
            "image": Image(),
            "text": Value("string"),
            "image_id": Value("string"),
            "split": Value("string"),
            "source": Value("string"),
            "year": Value("string"),
            "dialect": Value("string"),
            "corrected": Value("string"),
            "parent_image": Value("string"),
            "line_index": Value("int32"),
        }
    )
    split_map = {"train": "train", "val": "validation", "test": "test"}
    out: dict[str, Dataset] = {}
    counts: dict[str, int] = {}
    for split in SPLITS:
        rows = build_rows(data_dir, split, meta_index)
        counts[split] = len(rows)
        hf_split = split_map[split]
        out[hf_split] = Dataset.from_list(rows, features=features)
        log.info("Built split %s (%s): %d examples", split, hf_split, len(rows))
    return DatasetDict(out), counts


def write_local_export(
    export_dir: Path,
    readme: str,
    data_dir: Path,
    manifest: dict,
    license_text: str,
) -> None:
    """Write README, manifest, LICENSE, and character dictionary to a local folder."""
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "README.md").write_text(readme, encoding="utf-8")
    (export_dir / "LICENSE").write_text(license_text, encoding="utf-8")
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dict_src = data_dir / "dictionary" / "yoruba_char_dict.txt"
    if dict_src.is_file():
        dict_dst = export_dir / "yoruba_char_dict.txt"
        dict_dst.write_text(dict_src.read_text(encoding="utf-8"), encoding="utf-8")
    log.info("Local export written to %s", export_dir)


def push_to_hub(
    dataset_dict,
    repo_id: str,
    readme: str,
    license_text: str,
    *,
    private: bool,
    commit_message: str,
    data_dir: Path,
    upload_mode: str = "auto",
) -> None:
    """Create Hub repo, push dataset shards, README, LICENSE, and dictionary."""
    import tempfile

    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    n_examples = sum(len(dataset_dict[split]) for split in dataset_dict)
    mode = upload_mode
    if mode == "auto" and n_examples > 2000:
        mode = "large_folder"

    if mode == "large_folder":
        with tempfile.TemporaryDirectory(prefix="yoruba_hf_upload_") as tmp:
            staging = Path(tmp) / "dataset"
            log.info(
                "Saving DatasetDict to %s for resumable upload (%d examples)…",
                staging,
                n_examples,
            )
            dataset_dict.save_to_disk(staging)
            log.info("Uploading via upload_large_folder to %s …", repo_id)
            api.upload_large_folder(
                folder_path=str(staging),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )
    else:
        log.info("Pushing DatasetDict to %s via push_to_hub …", repo_id)
        dataset_dict.push_to_hub(
            repo_id,
            private=private,
            commit_message=commit_message,
            max_shard_size="500MB",
        )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card (README.md)",
    )
    api.upload_file(
        path_or_fileobj=license_text.encode("utf-8"),
        path_in_repo="LICENSE",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add CC BY 4.0 LICENSE",
    )
    dict_path = data_dir / "dictionary" / "yoruba_char_dict.txt"
    if dict_path.is_file():
        api.upload_file(
            path_or_fileobj=str(dict_path),
            path_in_repo="dictionary/yoruba_char_dict.txt",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add Yorùbá character dictionary",
        )
    log.info("Upload complete: https://huggingface.co/datasets/%s", repo_id)


def main() -> None:
    """Build dataset, write card, and optionally push to Hugging Face."""
    args = parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"Data dir not found: {args.data_dir}")

    username = None
    try:
        from huggingface_hub import HfApi

        who = HfApi().whoami()
        username = who.get("name") or who.get("fullname")
    except Exception:  # noqa: BLE001
        pass

    default_repo = f"{username}/yoruba-ocr-line-crops" if username else "USER/yoruba-ocr-line-crops"
    repo_id = args.repo_id or default_repo

    dataset_dict, split_counts = build_dataset_dict(args.data_dir, args.raw_dir)
    consolidation = load_consolidation_stats(args.consolidation_report)
    char_dict_n = char_dict_size(args.data_dir)
    readme = render_dataset_card(
        repo_id=repo_id,
        split_counts=split_counts,
        consolidation=consolidation,
        char_dict_n=char_dict_n,
        license_id=args.license,
    )
    license_text = render_license_text()

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id,
        "license": args.license,
        "authors": [name for name, _ in DATASET_AUTHORS],
        "split_counts": split_counts,
        "total_examples": sum(split_counts.values()),
        "char_dict_size": char_dict_n,
        "consolidation_report": str(args.consolidation_report),
    }
    report_path = Path("results/tables/hf_dataset_upload.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote upload manifest to %s", report_path)

    export_dir = args.export_dir or Path("data/hf_export")
    if args.dry_run or args.export_dir:
        write_local_export(export_dir, readme, args.data_dir, manifest, license_text)
        log.info("Dry run — card at %s/README.md", export_dir)

    if args.push:
        if args.dry_run:
            log.warning("--push ignored together with --dry-run")
        elif not args.repo_id and username is None:
            raise SystemExit("Set --repo-id or log in with HF_TOKEN before --push.")
        else:
            push_to_hub(
                dataset_dict,
                repo_id,
                readme,
                license_text,
                private=args.private,
                commit_message=args.commit_message,
                data_dir=args.data_dir,
                upload_mode=args.upload_mode,
            )
    elif not args.dry_run and not args.export_dir:
        log.info("Built dataset in memory. Pass --dry-run, --export-dir, or --push.")


if __name__ == "__main__":
    main()
