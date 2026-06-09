"""
Redact browser chrome (URLs, status bar) from the annotation-interface figure.

Usage:
    python scripts/anonymize_interface_figure.py
    python scripts/anonymize_interface_figure.py --input path/to/figure.jpeg
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from PIL import Image, ImageFilter, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_INPUT = Path(
    "FormattingGuidelines-IJCAI-ECAI-26/figures/ocr_extraction_interface.jpeg"
)

# Fractions of image height covering mobile Safari top/bottom chrome.
TOP_FRAC = 0.12
BOTTOM_FRAC = 0.875


def anonymize_interface_figure(input_path: Path, *, backup: bool = True) -> Path:
    """
    Blur and grey out top/bottom browser bars that expose deployment URLs.

    Saves over ``input_path`` after optionally backing up the original.
    """
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    backup_path = input_path.with_name(f"{input_path.stem}_original{input_path.suffix}")
    if backup and not backup_path.exists():
        shutil.copy2(input_path, backup_path)
        log.info("Backup written to %s", backup_path)

    im = Image.open(input_path).convert("RGB")
    w, h = im.size
    regions = [
        (0, 0, w, int(h * TOP_FRAC)),
        (0, int(h * BOTTOM_FRAC), w, h),
    ]

    out = im.copy()
    fill = (235, 235, 235)
    for x0, y0, x1, y1 in regions:
        crop = im.crop((x0, y0, x1, y1))
        blurred = crop.filter(ImageFilter.GaussianBlur(radius=24))
        out.paste(blurred, (x0, y0))
        draw = ImageDraw.Draw(out)
        draw.rectangle([x0, y0, x1, y1], fill=fill)

    out.save(input_path, quality=93, optimize=True)
    log.info("Anonymized %s (%dx%d)", input_path, w, h)
    return input_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Redact browser URL bars from the annotation-interface figure."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to ocr_extraction_interface.jpeg",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write *_original.jpeg backup.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    anonymize_interface_figure(args.input, backup=not args.no_backup)


if __name__ == "__main__":
    main()
