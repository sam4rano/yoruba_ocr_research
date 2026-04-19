"""
Export the consolidated datasets into an LLM/VLM conversational JSONL format (e.g. for Qwen-VL).

Produces a dataset where the system instructs the AI to extract Yoruba text from the image.
Image paths in JSONL are repo-relative when under ``--repo-root`` (default: cwd) so clones
and Colab runs resolve files without machine-specific absolute prefixes.

Usage:
    python scripts/08_export_qwen_finetune.py
    python scripts/08_export_qwen_finetune.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def portable_path(repo_root: Path, path: Path) -> str:
    """Return posix path relative to repo_root when possible."""
    root = repo_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def export_qwen(
    data_dir: Path, out_dir: Path, split: str, repo_root: Path
) -> None:
    label_file = data_dir / "labels" / f"{split}.txt"
    if not label_file.exists():
        log.warning(f"Label file not found: {label_file}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{split}.jsonl"

    count = 0
    with label_file.open(encoding="utf-8") as fh, out_jsonl.open("w", encoding="utf-8") as out:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            
            rel_path, text = parts
            src_img = data_dir / rel_path
            
            if not src_img.exists():
                continue

            img_ref = portable_path(repo_root, src_img)
            entry = {
                "id": src_img.stem,
                "conversations": [
                    {
                        "from": "user",
                        "value": (
                            f"Picture 1: <img>{img_ref}</img>\n"
                            "Extract the Yorùbá text from this image exactly as written."
                        ),
                    },
                    {
                        "from": "assistant",
                        "value": text
                    }
                ]
            }
            
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    log.info(f"Exported {count} pairs for '{split}' to {out_jsonl}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Export to Qwen conversational JSONL format.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/qwen_finetune"))
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root for portable image paths in JSONL (default: cwd).",
    )
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        export_qwen(args.data_dir, args.out_dir, split, args.repo_root)

    log.info("Qwen VLM export complete! Ready for use with LLaMA-Factory / Swift.")

if __name__ == "__main__":
    main()
