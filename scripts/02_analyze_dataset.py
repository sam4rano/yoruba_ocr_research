"""
Exploratory dataset analysis for the consolidated Yorùbá OCR data.

Computes and saves:
  - Per-split sample counts and text length distributions
  - Character frequency table (top N chars)
  - Diacritic character distribution (tone marks + subdot)
  - Vocabulary coverage vs the character dictionary
  - With --plot: results/tables/figures/text_length_distribution.png and
    char_frequency_top.png

Usage:
    python scripts/02_analyze_dataset.py
    python scripts/02_analyze_dataset.py --data-dir data/processed \
        --output-dir results/tables --plot
"""

from __future__ import annotations

import argparse
import json
import logging
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Yorùbá-specific combining diacritics present in the character dictionary
COMBINING_NAMES = {
    "\u0300": "combining_grave",      # ̀  (low tone)
    "\u0301": "combining_acute",      # ́  (high tone)
    "\u0323": "combining_dot_below",  # ̣  (subdot for ẹ/ọ/ṣ)
}

# Precomposed special characters (diacritic-bearing)
DIACRITIC_CHARS = set("ẹọṣẸỌṢàáèéìíòóùúÀÁÈÉÌÍÒÓÙÚńṢ")


def read_labels(label_path: Path) -> list[str]:
    """Return list of ground-truth text strings from a PaddleOCR label file."""
    texts = []
    with label_path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                texts.append(unicodedata.normalize("NFC", parts[1]))
    return texts


def text_length_stats(texts: list[str]) -> dict:
    """Compute min/max/mean/median text length statistics."""
    if not texts:
        return {}
    lengths = [len(t) for t in texts]
    lengths.sort()
    n = len(lengths)
    return {
        "count": n,
        "min": lengths[0],
        "max": lengths[-1],
        "mean": round(sum(lengths) / n, 2),
        "median": lengths[n // 2],
        "p95": lengths[int(n * 0.95)],
    }


def char_frequency(texts: list[str]) -> Counter:
    """Count character occurrences across all texts (NFC-normalised)."""
    counter: Counter = Counter()
    for text in texts:
        counter.update(text)
    return counter


def diacritic_stats(texts: list[str]) -> dict:
    """
    Count combining diacritics and precomposed diacritic-bearing characters.

    Decomposes each text to NFD to reliably isolate combining characters,
    then counts their occurrences.
    """
    combining_counts: Counter = Counter()
    precomposed_counts: Counter = Counter()

    for text in texts:
        nfd = unicodedata.normalize("NFD", text)
        for ch in nfd:
            if unicodedata.combining(ch):
                name = COMBINING_NAMES.get(ch, f"U+{ord(ch):04X}")
                combining_counts[name] += 1
        for ch in text:
            if ch in DIACRITIC_CHARS:
                precomposed_counts[ch] += 1

    return {
        "combining_diacritics": dict(combining_counts.most_common()),
        "precomposed_diacritic_chars": dict(precomposed_counts.most_common()),
        "total_combining": sum(combining_counts.values()),
        "total_precomposed_diacritic": sum(precomposed_counts.values()),
    }


def vocab_coverage(chars_in_data: set[str], dict_path: Path) -> dict:
    """Check how many data characters are covered by the character dictionary."""
    if not dict_path.exists():
        return {"warning": "dict not found"}
    with dict_path.open(encoding="utf-8") as fh:
        dict_chars = {line.rstrip("\n") for line in fh if line.rstrip("\n")}
    unseen = chars_in_data - dict_chars
    return {
        "dict_size": len(dict_chars),
        "unique_chars_in_data": len(chars_in_data),
        "chars_in_dict": len(chars_in_data & dict_chars),
        "chars_not_in_dict": sorted(unseen),
        "coverage_pct": round(
            100 * len(chars_in_data & dict_chars) / max(len(chars_in_data), 1), 2
        ),
    }


def save_plots(per_split: dict, output_dir: Path) -> None:
    """Save matplotlib figures for text length and character frequency."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        log.warning("matplotlib not installed; skipping plots.")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Text length histogram
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, split in zip(axes, ("train", "val", "test")):
        lengths = [len(t) for t in per_split.get(split, [])]
        if lengths:
            ax.hist(lengths, bins=30, edgecolor="black", alpha=0.8)
        ax.set_title(f"{split} (n={len(lengths)})")
        ax.set_xlabel("characters per line")
        ax.set_ylabel("frequency")
    fig.suptitle("Yorùbá OCR — Text Length Distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "text_length_distribution.png", dpi=120)
    plt.close(fig)
    log.info("Saved text length plot.")

    # Top characters (pool all splits)
    all_texts = [t for texts in per_split.values() for t in texts]
    freq = char_frequency(all_texts)
    top_n = 30
    common = freq.most_common(top_n)
    if common:
        chars = [c if c != " " else "(space)" for c, _ in common]
        counts = [n for _, n in common]
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.bar(range(len(chars)), counts, color="steelblue", edgecolor="black")
        ax2.set_xticks(range(len(chars)))
        ax2.set_xticklabels(chars, fontsize=8, rotation=45, ha="right")
        ax2.set_ylabel("count")
        ax2.set_title(f"Yorùbá OCR — Top {top_n} characters (all splits)")
        fig2.tight_layout()
        fig2.savefig(fig_dir / "char_frequency_top.png", dpi=120)
        plt.close(fig2)
        log.info("Saved character frequency plot.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyse the consolidated Yorùbá OCR dataset."
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        type=Path,
        default=Path("data/processed"),
        help="Processed dataset root (output of 01_consolidate_data.py).",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory to write analysis JSON and optional plots.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save matplotlib plots.",
    )
    return parser.parse_args()


def main() -> None:
    """Run full dataset analysis and persist results."""
    args = parse_args()
    dict_path = args.data_dir / "dictionary" / "yoruba_char_dict.txt"

    per_split: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        label_file = args.data_dir / "labels" / f"{split}.txt"
        if label_file.exists():
            per_split[split] = read_labels(label_file)
            log.info("Loaded %d %s examples.", len(per_split[split]), split)
        else:
            log.warning("Label file not found: %s", label_file)
            per_split[split] = []

    all_texts = [t for texts in per_split.values() for t in texts]

    char_freq = char_frequency(all_texts)
    all_chars_in_data = set(char_freq.keys())

    report = {
        "stage": "analyze",
        "split_counts": {s: len(t) for s, t in per_split.items()},
        "total_samples": len(all_texts),
        "text_length": {s: text_length_stats(t) for s, t in per_split.items()},
        "top_50_chars": dict(char_freq.most_common(50)),
        "unique_char_count": len(char_freq),
        "diacritic_stats": diacritic_stats(all_texts),
        "vocab_coverage": vocab_coverage(all_chars_in_data, dict_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "dataset_analysis.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    log.info("Analysis saved to %s", out_path)

    # Summary to terminal
    dc = report["diacritic_stats"]
    cov = report["vocab_coverage"]
    log.info(
        "Totals — train: %d  val: %d  test: %d",
        report["split_counts"].get("train", 0),
        report["split_counts"].get("val", 0),
        report["split_counts"].get("test", 0),
    )
    log.info(
        "Diacritics — combining: %d  precomposed: %d",
        dc["total_combining"],
        dc["total_precomposed_diacritic"],
    )
    log.info(
        "Dict coverage: %s%%  (unseen chars: %s)",
        cov.get("coverage_pct", "n/a"),
        cov.get("chars_not_in_dict", []),
    )

    if args.plot:
        save_plots(per_split, args.output_dir)


if __name__ == "__main__":
    main()
