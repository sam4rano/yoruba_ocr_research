"""
Export the consolidated datasets into Tesseract's `tesstrain` format.

Tesseract fine-tuning requires each image file to have a matching 
text file with the `.gt.txt` extension containing exactly the ground truth text.

Usage:
    python scripts/08_export_tesseract_finetune.py
"""

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

def export_tesseract(data_dir: Path, out_dir: Path, split: str) -> None:
    label_file = data_dir / "labels" / f"{split}.txt"
    if not label_file.exists():
        log.warning(f"Label file not found: {label_file}")
        return

    dst_dir = out_dir / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with label_file.open(encoding="utf-8") as fh:
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

            # Copy image
            dst_img = dst_dir / src_img.name
            shutil.copy2(src_img, dst_img)
            
            # Write .gt.txt
            gt_text_path = dst_dir / src_img.with_suffix(".gt.txt").name
            gt_text_path.write_text(text, encoding="utf-8")
            
            count += 1

    log.info(f"Exported {count} pairs for '{split}' to {dst_dir}")

def main():
    parser = argparse.ArgumentParser(description="Export to Tesseract tesstrain format.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/tesseract_finetune"))
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        export_tesseract(args.data_dir, args.out_dir, split)

    log.info("Tesseract export complete! Ready for use with `tesstrain`.")

if __name__ == "__main__":
    main()
