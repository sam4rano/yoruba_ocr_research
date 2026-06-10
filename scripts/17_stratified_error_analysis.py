"""
Stratified error analysis and minimal-pair evaluation subset.

Joins per-sample JSONL logs with test-split metadata (book source, diacritic
density) and reports:

  * corpus CER/WER/DER on lines touching diacritic minimal-pair word types;
  * DER/CER by diacritic-density quartile;
  * DER/CER by book source (Yorùbá di Wúrà volume);
  * diacritic error taxonomy (exact, substitution, deletion-heavy, insertion,
    total tone drop).

Usage:
    python scripts/17_stratified_error_analysis.py
    python scripts/17_stratified_error_analysis.py \\
        --jsonl results/tables/paddleocr_vl15_lora_finetuned_test.jsonl \\
        --jsonl results/tables/trocr_large_printed_zero_shot_test.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

EXPORT_PATTERN = re.compile(r"^yoruba_ocr_(\d+)")
WORD_SPLIT = re.compile(r"[^\w\u00C0-\u024F\u1E00-\u1EFF\u0300-\u036f\u0323]+", re.UNICODE)

DEFAULT_MODEL_JSONLS = [
    "paddleocr_vl15_lora_finetuned_test.jsonl",
    "paddleocr_vl15_zero_shot_test.jsonl",
    "baseline_english_pretrained_test.jsonl",
    "ablation_data_size_100pct_test_test.jsonl",
    "surya_v2_zero_shot_test.jsonl",
    "surya_finetuned_test.jsonl",
    "trocr_large_printed_zero_shot_test.jsonl",
]

DENSITY_LABELS = ("q1_low", "q2", "q3", "q4_high")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Stratified OCR error analysis on test JSONL logs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Processed dataset root (labels + images).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw export root for dataset_metadata.csv lookup.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Split to analyse.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        action="append",
        default=None,
        help="Per-sample eval JSONL (repeatable). Defaults to main Table 1 models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory for CSV/JSON outputs.",
    )
    return parser.parse_args()


import sys

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from evaluate_utils import aggregate_metrics, load_test_pairs, normalize_yoruba_text  # noqa: E402


def load_metadata_index(raw_dir: Path) -> dict[str, dict[str, str]]:
    """
    Build ``{image_basename: {source, year, dialect, ...}}`` from raw exports.

    Later exports overwrite earlier ones on filename collision, matching
    consolidation policy.
    """
    index: dict[str, dict[str, str]] = {}
    if not raw_dir.exists():
        log.warning("Raw dir missing: %s — book stratification will be empty.", raw_dir)
        return index

    for item in sorted(raw_dir.iterdir()):
        if not item.is_dir() or not EXPORT_PATTERN.match(item.name):
            continue
        meta_path = item / "metadata" / "dataset_metadata.csv"
        if not meta_path.exists():
            alt = item / "metadata" / "dataset_metadata(1).csv"
            meta_path = alt if alt.exists() else meta_path
        if not meta_path.exists():
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
                        "parent_image": (row.get("parent_image") or "").strip(),
                    }
    log.info("Indexed metadata for %d line images.", len(index))
    return index


def diacritic_skeleton(text: str) -> str:
    """Strip combining marks; lowercase Latin letters for grouping."""
    nfd = unicodedata.normalize("NFD", text)
    base = "".join(c for c in nfd if not unicodedata.combining(c))
    return base.lower()


def extract_gt_diacs(text: str) -> list[str]:
    """Return combining diacritic codepoints in NFD order."""
    return [
        c
        for c in unicodedata.normalize("NFD", text)
        if unicodedata.combining(c)
    ]


def diacritic_density(text: str) -> float:
    """Combining diacritics per NFC character (0 when empty)."""
    gt = normalize_yoruba_text(text)
    if not gt:
        return 0.0
    n_diacs = len(extract_gt_diacs(gt))
    return n_diacs / len(gt)


def build_minimal_pair_index(gt_texts: list[str]) -> tuple[set[str], dict[str, list[str]]]:
    """
    Identify word skeletons with multiple surface forms in the split.

    Returns:
        skeletons: set of skeletons that participate in minimal pairs;
        examples: skeleton -> sorted unique surface forms (for reporting).
    """
    skeleton_forms: dict[str, set[str]] = defaultdict(set)
    for text in gt_texts:
        for token in WORD_SPLIT.split(normalize_yoruba_text(text)):
            if len(token) < 2:
                continue
            if not extract_gt_diacs(token):
                continue
            skeleton_forms[diacritic_skeleton(token)].add(token)

    examples: dict[str, list[str]] = {}
    skeletons: set[str] = set()
    for sk, forms in skeleton_forms.items():
        if len(forms) >= 2:
            skeletons.add(sk)
            examples[sk] = sorted(forms)[:6]

    return skeletons, examples


def line_has_minimal_pair(text: str, mp_skeletons: set[str]) -> bool:
    """True if any token in ``text`` belongs to a minimal-pair skeleton group."""
    for token in WORD_SPLIT.split(normalize_yoruba_text(text)):
        if diacritic_skeleton(token) in mp_skeletons:
            return True
    return False


def classify_diac_error(pred: str, gt: str) -> str:
    """
    Assign a diacritic-centric error category for GT with ≥1 combining mark.

    Categories are mutually exclusive heuristics for stratified reporting.
    """
    pred_n = normalize_yoruba_text(pred)
    gt_n = normalize_yoruba_text(gt)
    pred_diacs = extract_gt_diacs(pred_n)
    gt_diacs = extract_gt_diacs(gt_n)

    if not gt_diacs:
        return "no_gt_diacritics"
    if not pred_diacs:
        return "total_tone_drop"
    if pred_diacs == gt_diacs:
        return "exact_diacritics"
    if len(pred_diacs) < len(gt_diacs):
        return "deletion_heavy"
    if len(pred_diacs) > len(gt_diacs):
        return "insertion_heavy"
    return "substitution"


def density_quartile_label(value: float, edges: list[float]) -> str:
    """Map a density value to q1–q4 using precomputed split edges."""
    if value <= edges[0]:
        return DENSITY_LABELS[0]
    if value <= edges[1]:
        return DENSITY_LABELS[1]
    if value <= edges[2]:
        return DENSITY_LABELS[2]
    return DENSITY_LABELS[3]


def compute_quartile_edges(densities: list[float]) -> list[float]:
    """Return three internal quartile boundaries (inclusive lower buckets)."""
    if not densities:
        return [0.0, 0.0, 0.0]
    sorted_vals = sorted(densities)
    n = len(sorted_vals)

    def _q(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return sorted_vals[idx]

    return [_q(0.25), _q(0.50), _q(0.75)]


def aggregate_subset(
    rows: list[dict],
    *,
    predicate,
) -> dict:
    """Corpus metrics over rows matching ``predicate(row)``."""
    subset = [(r["pred"], r["gt"]) for r in rows if predicate(r)]
    if not subset:
        return {"n": 0, "cer": None, "wer": None, "der": None, "der_n": 0}
    return aggregate_metrics(subset)


def load_jsonl_rows(path: Path) -> list[dict]:
    """Load per-sample JSONL records."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def model_label_from_jsonl(path: Path) -> str:
    """Derive a stable model key from a JSONL filename."""
    name = path.name
    for suffix in ("_test.jsonl", "_val.jsonl", ".jsonl"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def analyse_model(
    model_key: str,
    rows: list[dict],
    sample_meta: list[dict],
    mp_skeletons: set[str],
    density_edges: list[float],
) -> dict:
    """Run all stratifications for one model."""
    enriched = []
    for row, meta in zip(rows, sample_meta, strict=True):
        enriched.append({**row, **meta})

    full = aggregate_subset(enriched, predicate=lambda _: True)
    mp = aggregate_subset(
        enriched,
        predicate=lambda r: r.get("is_minimal_pair_line"),
    )

    by_density: dict[str, dict] = {}
    for label in DENSITY_LABELS:
        by_density[label] = aggregate_subset(
            enriched,
            predicate=lambda r, lab=label: r.get("density_quartile") == lab,
        )

    by_book: dict[str, dict] = {}
    book_counts: Counter[str] = Counter()
    for r in enriched:
        book_counts[r.get("book_source") or "unknown"] += 1
    for book in sorted(book_counts):
        by_book[book] = aggregate_subset(
            enriched,
            predicate=lambda r, b=book: (r.get("book_source") or "unknown") == b,
        )

    taxonomy = Counter(
        classify_diac_error(r["pred"], r["gt"])
        for r in enriched
        if extract_gt_diacs(r["gt"])
    )

    return {
        "model": model_key,
        "full_test": full,
        "minimal_pair_lines": mp,
        "by_density_quartile": by_density,
        "by_book_source": by_book,
        "error_taxonomy": dict(taxonomy),
        "n_minimal_pair_lines": sum(1 for r in enriched if r.get("is_minimal_pair_line")),
    }


def write_density_csv(path: Path, results: list[dict]) -> None:
    """Write long-form density stratification table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "density_quartile", "n", "cer_pct", "wer_pct", "der_pct", "der_n"],
        )
        writer.writeheader()
        for res in results:
            for quartile, metrics in res["by_density_quartile"].items():
                writer.writerow(
                    {
                        "model": res["model"],
                        "density_quartile": quartile,
                        "n": metrics.get("n", 0),
                        "cer_pct": _pct(metrics.get("cer")),
                        "wer_pct": _pct(metrics.get("wer")),
                        "der_pct": _pct(metrics.get("der")),
                        "der_n": metrics.get("der_n", 0),
                    }
                )


def write_book_csv(path: Path, results: list[dict]) -> None:
    """Write long-form book-source stratification table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "book_source", "n", "cer_pct", "wer_pct", "der_pct", "der_n"],
        )
        writer.writeheader()
        for res in results:
            for book, metrics in sorted(res["by_book_source"].items()):
                writer.writerow(
                    {
                        "model": res["model"],
                        "book_source": book,
                        "n": metrics.get("n", 0),
                        "cer_pct": _pct(metrics.get("cer")),
                        "wer_pct": _pct(metrics.get("wer")),
                        "der_pct": _pct(metrics.get("der")),
                        "der_n": metrics.get("der_n", 0),
                    }
                )


def write_taxonomy_csv(path: Path, results: list[dict]) -> None:
    """Write error taxonomy counts per model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    categories = sorted(
        {cat for res in results for cat in res["error_taxonomy"]}
    )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "category", "count"])
        writer.writeheader()
        for res in results:
            for cat in categories:
                count = res["error_taxonomy"].get(cat, 0)
                if count:
                    writer.writerow(
                        {"model": res["model"], "category": cat, "count": count}
                    )


def write_minimal_pair_csv(path: Path, results: list[dict]) -> None:
    """Write minimal-pair subset metrics per model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "n_mp_lines", "cer_pct", "wer_pct", "der_pct", "der_n"],
        )
        writer.writeheader()
        for res in results:
            mp = res["minimal_pair_lines"]
            writer.writerow(
                {
                    "model": res["model"],
                    "n_mp_lines": res["n_minimal_pair_lines"],
                    "cer_pct": _pct(mp.get("cer")),
                    "wer_pct": _pct(mp.get("wer")),
                    "der_pct": _pct(mp.get("der")),
                    "der_n": mp.get("der_n", 0),
                }
            )


def _pct(rate: float | None) -> str:
    """Format a rate as a percentage string with one decimal."""
    if rate is None:
        return ""
    return f"{rate * 100:.1f}"


def main() -> None:
    """Entry point."""
    args = parse_args()

    jsonl_paths = args.jsonl
    if not jsonl_paths:
        jsonl_paths = [args.output_dir / name for name in DEFAULT_MODEL_JSONLS]

    jsonl_paths = [p for p in jsonl_paths if p.exists()]
    if not jsonl_paths:
        raise FileNotFoundError("No JSONL logs found — run evaluation first.")

    pairs = load_test_pairs(args.data_dir, args.split)
    gt_texts = [gt for _, gt in pairs]
    meta_index = load_metadata_index(args.raw_dir)

    mp_skeletons, mp_examples = build_minimal_pair_index(gt_texts)
    log.info(
        "Minimal-pair skeleton groups on %s split: %d (e.g. %s)",
        args.split,
        len(mp_skeletons),
        list(mp_examples.items())[:3],
    )

    densities = [diacritic_density(gt) for gt in gt_texts]
    density_edges = compute_quartile_edges(
        [d for d in densities if d > 0] or densities
    )

    sample_meta: list[dict] = []
    for (img_path, gt), density in zip(pairs, densities, strict=True):
        basename = img_path.name
        raw = meta_index.get(basename, {})
        book = raw.get("source") or "unknown"
        sample_meta.append(
            {
                "image": str(img_path),
                "book_source": book,
                "diacritic_density": round(density, 4),
                "density_quartile": density_quartile_label(density, density_edges),
                "is_minimal_pair_line": line_has_minimal_pair(gt, mp_skeletons),
            }
        )

    model_results = []
    for jsonl_path in jsonl_paths:
        rows = load_jsonl_rows(jsonl_path)
        if len(rows) != len(sample_meta):
            raise ValueError(
                f"{jsonl_path.name}: expected {len(sample_meta)} rows, got {len(rows)}"
            )
        key = model_label_from_jsonl(jsonl_path)
        model_results.append(
            analyse_model(key, rows, sample_meta, mp_skeletons, density_edges)
        )
        log.info("Analysed %s", key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    mp_vocab_path = args.output_dir / "minimal_pair_vocabulary.json"
    mp_vocab_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "n_skeleton_groups": len(mp_skeletons),
                "n_lines_with_mp_token": sum(
                    1 for m in sample_meta if m["is_minimal_pair_line"]
                ),
                "density_quartile_edges": density_edges,
                "examples": mp_examples,
                "timestamp": timestamp,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_path = args.output_dir / "stratified_error_analysis.json"
    summary_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "n_samples": len(sample_meta),
                "density_quartile_edges": density_edges,
                "minimal_pair": {
                    "n_skeleton_groups": len(mp_skeletons),
                    "n_lines": sum(1 for m in sample_meta if m["is_minimal_pair_line"]),
                },
                "models": model_results,
                "timestamp": timestamp,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_minimal_pair_csv(args.output_dir / "minimal_pair_subset.csv", model_results)
    write_density_csv(args.output_dir / "stratified_der_by_density.csv", model_results)
    write_book_csv(args.output_dir / "stratified_der_by_book.csv", model_results)
    write_taxonomy_csv(args.output_dir / "error_taxonomy.csv", model_results)

    log.info("Wrote stratified analysis to %s", args.output_dir)


if __name__ == "__main__":
    main()
