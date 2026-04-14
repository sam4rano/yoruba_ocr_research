"""
Tesseract OCR baseline evaluation on the Yorùbá OCR test set.

Runs Tesseract with three language configurations:
  - eng         : English-only (direct zero-shot baseline)
  - yor         : Yoruba language pack (if installed)
  - eng+yor     : Combined English + Yoruba (if yor pack is installed)

Each configuration is evaluated independently, with CER/WER/DER computed
against the human-corrected ground truth. Results are appended to
results/tables/metrics.csv under model labels such as
"tesseract_eng", "tesseract_yor", "tesseract_eng+yor".

Install requirements:
    # macOS
    brew install tesseract tesseract-lang
    pip install pytesseract Pillow

    # Ubuntu
    apt-get install tesseract-ocr tesseract-ocr-yor
    pip install pytesseract Pillow

Usage:
    python scripts/07_baseline_tesseract.py
    python scripts/07_baseline_tesseract.py \
        --data-dir data/processed \
        --split test \
        --langs eng yor eng+yor \
        --psm 7
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import unicodedata
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def tesseract_is_installed() -> bool:
    """Return True if the tesseract binary is reachable on PATH."""
    result = subprocess.run(
        ["tesseract", "--version"],
        capture_output=True,
    )
    return result.returncode == 0


def yor_pack_available() -> bool:
    """Return True if the Yoruba (yor) language data is installed."""
    result = subprocess.run(
        ["tesseract", "--list-langs"],
        capture_output=True,
        text=True,
    )
    return "yor" in result.stdout or "yor" in result.stderr


def run_tesseract(
    img_path: Path,
    lang: str,
    psm: int,
) -> str:
    """
    Run Tesseract recognition on a single line-crop image.

    psm 7 = treat image as a single text line (correct for our crops).
    Returns the recognised text string, stripped of trailing whitespace/newlines.
    """
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        raise ImportError(
            "pytesseract and Pillow are required. "
            "Run: pip install pytesseract Pillow"
        )
    img = Image.open(img_path)
    raw = pytesseract.image_to_string(
        img,
        lang=lang,
        config=f"--psm {psm} --oem 3",
    )
    return unicodedata.normalize("NFC", raw.strip())


def evaluate_lang(
    lang: str,
    pairs: list[tuple[Path, str]],
    psm: int,
) -> list[tuple[str, str]]:
    """
    Run Tesseract inference for a given language config on all image pairs.

    Returns list of (prediction, ground_truth) strings.
    Skips images that cause Tesseract to error, substituting an empty string.
    """
    results = []
    failed = 0
    for img_path, gt in pairs:
        try:
            pred = run_tesseract(img_path, lang, psm)
        except Exception as exc:  # noqa: BLE001
            log.debug("Tesseract failed on %s: %s", img_path.name, exc)
            pred = ""
            failed += 1
        results.append((pred, gt))
    if failed:
        log.warning("Tesseract failed on %d images (lang=%s).", failed, lang)
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tesseract OCR baseline for Yorùbá OCR benchmark."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Consolidated dataset root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["eng"],
        help="Tesseract language strings to evaluate (e.g. eng yor eng+yor).",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=7,
        help="Tesseract page segmentation mode (7 = single text line).",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/tables/metrics.csv"),
        help="Shared results table — appended on each run.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Tesseract evaluation across all requested language configurations."""
    args = parse_args()

    if not tesseract_is_installed():
        log.error(
            "Tesseract not found on PATH.\n"
            "  macOS:  brew install tesseract tesseract-lang\n"
            "  Ubuntu: apt-get install tesseract-ocr tesseract-ocr-yor"
        )
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_utils import load_test_pairs, aggregate_metrics, save_results  # type: ignore  # noqa: E402

    pairs = load_test_pairs(args.data_dir, args.split)
    log.info("Loaded %d labelled examples.", len(pairs))

    if not yor_pack_available():
        log.warning(
            "Yoruba (yor) language pack not found. "
            "Install with: brew install tesseract-lang  "
            "or: apt-get install tesseract-ocr-yor\n"
            "Skipping yor and eng+yor configs."
        )
        args.langs = [l for l in args.langs if "yor" not in l]  # noqa: E741

    for lang in args.langs:
        log.info("Evaluating Tesseract lang=%s ...", lang)
        pred_pairs = evaluate_lang(lang, pairs, args.psm)
        metrics = aggregate_metrics(pred_pairs)
        model_name = f"tesseract_{lang}"
        log.info(
            "%s — CER: %.4f  WER: %.4f  DER: %.4f  (n=%d)",
            model_name,
            metrics["cer"],
            metrics["wer"],
            metrics["der"],
            metrics["n"],
        )
        per_sample_log = (
            Path("results/tables") / f"{model_name}_{args.split}.jsonl"
        )
        save_results(
            metrics,
            model_name=model_name,
            split=args.split,
            csv_path=args.results_csv,
            jsonl_path=per_sample_log,
        )
        log.info("Results for %s appended to %s", model_name, args.results_csv)


if __name__ == "__main__":
    main()
