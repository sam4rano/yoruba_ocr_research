"""
Export consolidated PaddleOCR-format data to JSONL for PaddleOCR-VL-1.5 SFT / LoRA.

**Does not read or write ``data/processed`` labels in place** — only creates new files
under ``--out-dir`` (default ``data/paddleocr_vl15_sft``).

Each line is one sample with OpenAI-style messages + image path (repo-relative when
``--repo-root`` contains ``data-dir``, else resolved absolute) for portability across machines.

Usage:
    python scripts/14_export_paddleocr_vl_sft.py
    python scripts/14_export_paddleocr_vl_sft.py --data-dir data/processed --out-dir data/paddleocr_vl15_sft
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Export Yorùbá line crops to JSONL for PaddleOCR-VL-1.5 fine-tuning."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Consolidated dataset root (read-only).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/paddleocr_vl15_sft"),
        help="Output directory for JSONL + manifest (created if missing).",
    )
    parser.add_argument(
        "--dict-path",
        type=Path,
        default=None,
        help="Optional: record dictionary path in manifest (default: data-dir/dictionary/yoruba_char_dict.txt if present).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root; image paths in JSONL are relative to this when under it (default: cwd).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to export.",
    )
    return parser.parse_args()


def portable_path(repo_root: Path, path: Path) -> str:
    """Return posix path relative to repo_root when possible (for Colab / other clones)."""
    root = repo_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def export_split(
    repo_root: Path,
    data_dir: Path,
    out_dir: Path,
    split: str,
    user_prompt: str,
) -> dict:
    """
    Write ``{split}.jsonl`` and return stats (exported count, skipped, paths).

    Skips rows with missing images or malformed lines; nothing is deleted.
    """
    label_file = data_dir / "labels" / f"{split}.txt"
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{split}.jsonl"
    exported = 0
    skipped: list[dict] = []
    ids_order: list[str] = []

    with label_file.open(encoding="utf-8") as fh_in, out_jsonl.open(
        "w", encoding="utf-8"
    ) as fh_out:
        for line_no, line in enumerate(fh_in, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                skipped.append({"line": line_no, "reason": "not_two_tab_fields"})
                continue
            rel_path, text = parts
            img_path = data_dir / rel_path
            if not img_path.is_file():
                skipped.append(
                    {"line": line_no, "reason": "missing_image", "path": rel_path}
                )
                continue
            gt = unicodedata.normalize("NFC", text)
            sample_id = img_path.stem
            img_ref = portable_path(repo_root, img_path)
            record = {
                "id": sample_id,
                "source_relative": rel_path.replace("\\", "/"),
                "source_image": img_ref,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_ref},
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                    {"role": "assistant", "content": gt},
                ],
            }
            fh_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1
            ids_order.append(sample_id)

    return {
        "split": split,
        "jsonl": portable_path(repo_root, out_jsonl),
        "exported": exported,
        "skipped": skipped,
        "ids_sha256": hashlib.sha256("\n".join(ids_order).encode("utf-8")).hexdigest(),
    }


def main() -> None:
    """Export selected splits and write a manifest JSON."""
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent))
    from paddle_vl_shared import USER_TEXT_OCR_YORUBA  # noqa: E402

    repo_root = args.repo_root
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dict_path = args.dict_path
    if dict_path is None:
        cand = args.data_dir / "dictionary" / "yoruba_char_dict.txt"
        dict_path = cand if cand.is_file() else None

    manifest: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": portable_path(repo_root, repo_root.resolve()),
        "data_dir": portable_path(repo_root, args.data_dir.resolve()),
        "out_dir": portable_path(repo_root, args.out_dir.resolve()),
        "splits": {},
        "dictionary_path": (
            portable_path(repo_root, dict_path.resolve()) if dict_path else None
        ),
    }

    for sp in args.splits:
        log.info("Exporting split '%s' ...", sp)
        manifest["splits"][sp] = export_split(
            repo_root, args.data_dir, args.out_dir, sp, USER_TEXT_OCR_YORUBA
        )
        log.info(
            "  wrote %d samples (%d skipped)",
            manifest["splits"][sp]["exported"],
            len(manifest["splits"][sp]["skipped"]),
        )

    man_path = args.out_dir / "manifest.json"
    man_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Manifest: %s", man_path)


if __name__ == "__main__":
    main()
