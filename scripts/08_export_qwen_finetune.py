"""
Export the consolidated datasets into an LLM/VLM conversational JSONL format (e.g. for Qwen-VL).

Produces a dataset where the system instructs the AI to extract Yoruba text from the image.

Usage:
    python scripts/08_export_qwen_finetune.py
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

def export_qwen(data_dir: Path, out_dir: Path, split: str) -> None:
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

            entry = {
                "id": src_img.stem,
                "conversations": [
                    {
                        "from": "user",
                        "value": f"Picture 1: <img>{src_img.absolute()}</img>\nExtract the Yorùbá text from this image exactly as written."
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

def main():
    parser = argparse.ArgumentParser(description="Export to Qwen conversational JSONL format.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/qwen_finetune"))
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        export_qwen(args.data_dir, args.out_dir, split)

    log.info("Qwen VLM export complete! Ready for use with LLaMA-Factory / Swift.")

if __name__ == "__main__":
    main()
