"""
Compile all evaluation results into the paper's final comparison tables.

Reads results/tables/metrics.csv (written by all evaluation scripts)
and produces:

  Table 1 — Main Comparison
    Baseline models vs fine-tuned PaddleOCR, on the test split.
    Columns: Model | CER ↓ | WER ↓ | DER ↓

  Table 2 — Ablation: Data Size
    Performance at 25/50/75/100% of training data.

  Table 3 — Ablation: Character Dictionary
    Yorùbá dict vs English dict.

  Table 4 — Ablation: Data Augmentation
    With RecAug vs without.

Each table is written as a Markdown file (for copy-paste into the paper
sections) and a CSV (for the results/ archive). Numbers are rounded to
1 decimal place (as percentages) per the experiment-reporting skill.

Usage:
    python scripts/11_compile_results.py
    python scripts/11_compile_results.py \
        --results-csv results/tables/metrics.csv \
        --output-dir results/tables
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Display names for the model labels used across all eval scripts
MODEL_DISPLAY = {
    "baseline_english_pretrained":      "PaddleOCR PP-OCRv4 (EN pretrained)",
    "tesseract_eng":                    "Tesseract (eng)",
    "tesseract_yor":                    "Tesseract (yor)",
    "tesseract_eng+yor":                "Tesseract (eng+yor)",
    "qwen25_vl_zero_shot":              "Qwen 2.5 VL (zero-shot)",
    "finetuned_paddleocr_v1":           "PaddleOCR PP-OCRv4 (fine-tuned)",
    # Ablation data size
    "ablation_data_size_025pct_test":   "Fine-tuned 25% data",
    "ablation_data_size_050pct_test":   "Fine-tuned 50% data",
    "ablation_data_size_075pct_test":   "Fine-tuned 75% data",
    "ablation_data_size_100pct_test":   "Fine-tuned 100% data",
    # Ablation dictionary
    "ablation_dict_yoruba_dict_test":   "Fine-tuned + Yorùbá dict",
    "ablation_dict_english_dict_test":  "Fine-tuned + English dict",
    # Ablation augmentation
    "ablation_aug_with_aug_test":       "Fine-tuned + RecAug",
    "ablation_aug_no_aug_test":         "Fine-tuned − RecAug",
}

# Ordered model rows for Table 1
TABLE1_ORDER = [
    "baseline_english_pretrained",
    "tesseract_eng",
    "tesseract_yor",
    "tesseract_eng+yor",
    "qwen25_vl_zero_shot",
    "finetuned_paddleocr_v1",
]

# Ablation groupings for Tables 2–4
ABLATION_GROUPS = {
    "data_size": [
        "ablation_data_size_025pct_test",
        "ablation_data_size_050pct_test",
        "ablation_data_size_075pct_test",
        "ablation_data_size_100pct_test",
    ],
    "dictionary": [
        "ablation_dict_english_dict_test",
        "ablation_dict_yoruba_dict_test",
    ],
    "augmentation": [
        "ablation_aug_no_aug_test",
        "ablation_aug_with_aug_test",
    ],
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(csv_path: Path) -> dict[str, dict]:
    """
    Load results CSV into a dict keyed by model label.

    Deduplicates by (model, split), keeping the **last** row in file order for each key.
    Does **not** rewrite metrics.csv (append-only history is preserved).

    When the same ``model`` appears under multiple splits, **test** is preferred for
    compilation, then **val**, then **train**.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {csv_path}\n"
            "Run the evaluation scripts first."
        )
    all_rows: list[dict] = []
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            all_rows.append(row)

    seen: dict[tuple[str, str], dict] = {}
    for row in all_rows:
        key = (row["model"], row.get("split", "test"))
        seen[key] = row

    models = sorted({r["model"] for r in all_rows})
    records: dict[str, dict] = {}
    for model in models:
        chosen: dict | None = None
        for split in ("test", "val", "train"):
            if (model, split) in seen:
                chosen = seen[(model, split)]
                break
        if chosen is None:
            for (m, _s), row in seen.items():
                if m == model:
                    chosen = row
                    break
        if chosen is not None:
            records[model] = chosen
    return records


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def pct(val: str | None) -> str:
    """
    Convert a stored metric rate to a percentage-style display (value × 100, 1 decimal).

    CER/WER may exceed 1.0 when insertions dominate; do not cap, or the table
    collapses to 100.0 for every such model.
    """
    if val is None or val == "":
        return "—"
    try:
        v = float(val)
        return f"{v * 100:.1f}"
    except ValueError:
        return "—"


def best_column(rows: list[dict], col: str) -> str | None:
    """Return the model label with the lowest value in col (lower = better)."""
    valid = [(r["model"], float(r[col])) for r in rows if r.get(col) not in (None, "")]
    if not valid:
        return None
    return min(valid, key=lambda x: x[1])[0]


def format_cell(val: str | None, is_best: bool) -> str:
    """Bold the best value in a column."""
    s = pct(val)
    return f"**{s}**" if is_best and s != "—" else s


def render_markdown_table(
    rows: list[dict],
    model_order: list[str],
) -> str:
    """
    Render a Markdown results table in the paper's standard format.

    | Model | CER ↓ | WER ↓ | DER ↓ |
    Bold the best (lowest) value per metric column.
    """
    present = [m for m in model_order if m in rows]
    data_rows = [rows[m] for m in present]

    best_cer = best_column(data_rows, "cer")
    best_wer = best_column(data_rows, "wer")
    best_der = best_column(data_rows, "der")

    lines = [
        "| Model | CER ↓ | WER ↓ | DER ↓ |",
        "|-------|------:|------:|------:|",
    ]
    for model_key in present:
        r = rows[model_key]
        display = MODEL_DISPLAY.get(model_key, model_key)
        cer = format_cell(r.get("cer"), r["model"] == best_cer)
        wer = format_cell(r.get("wer"), r["model"] == best_wer)
        der = format_cell(r.get("der"), r["model"] == best_der)
        lines.append(f"| {display} | {cer} | {wer} | {der} |")

    return "\n".join(lines)


def write_csv_table(
    rows: dict[str, dict],
    model_order: list[str],
    out_path: Path,
) -> None:
    """Write a subset of results rows as a clean CSV file."""
    present = [m for m in model_order if m in rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model_label", "display_name", "cer_pct", "wer_pct", "der_pct", "n"],
        )
        writer.writeheader()
        for key in present:
            r = rows[key]
            writer.writerow(
                {
                    "model_label": key,
                    "display_name": MODEL_DISPLAY.get(key, key),
                    "cer_pct": pct(r.get("cer")),
                    "wer_pct": pct(r.get("wer")),
                    "der_pct": pct(r.get("der")),
                    "n": r.get("n", ""),
                }
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compile all evaluation results into paper-ready tables."
    )
    parser.add_argument(
        "--results-csv",
        "--results_csv",
        "--metrics_csv",
        dest="results_csv",
        type=Path,
        default=Path("results/tables/metrics.csv"),
        help="Master results CSV written by evaluation scripts.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        "--out_dir",
        dest="output_dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory to write compiled Markdown and CSV tables.",
    )
    return parser.parse_args()


def main() -> None:
    """Load results and render all tables."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(args.results_csv)
    log.info("Loaded %d model result rows.", len(rows))

    missing = [m for m in TABLE1_ORDER if m not in rows]
    if missing:
        log.warning(
            "Table 1: no metrics row for: %s (run the corresponding eval or expect "
            "— in tables).",
            ", ".join(missing),
        )

    # --- Table 1: Main Comparison ---
    table1_md = render_markdown_table(rows, TABLE1_ORDER)
    (args.output_dir / "table1_main_comparison.md").write_text(
        "# Table 1 — Main Model Comparison (test split)\n\n" + table1_md + "\n",
        encoding="utf-8",
    )
    table1_csv = args.output_dir / "table1_main_comparison.csv"
    write_csv_table(rows, TABLE1_ORDER, table1_csv)
    # Alias for notebooks / older docs that expect this filename
    summary_alias = args.output_dir / "metrics_summary.csv"
    summary_alias.write_text(table1_csv.read_text(encoding="utf-8"), encoding="utf-8")
    log.info("Table 1 written (%s + metrics_summary.csv).", table1_csv.name)

    # --- Tables 2–4: Ablation Studies ---
    ablation_titles = {
        "data_size":    "Table 2 — Ablation: Training Data Size (test split)",
        "dictionary":   "Table 3 — Ablation: Character Dictionary (test split)",
        "augmentation": "Table 4 — Ablation: Data Augmentation (test split)",
    }
    for abl_id, model_order in ABLATION_GROUPS.items():
        title = ablation_titles[abl_id]
        present = [m for m in model_order if m in rows]
        if not present:
            log.warning("No results found for ablation '%s'. Skipping.", abl_id)
            continue
        md = render_markdown_table(rows, model_order)
        md_path = args.output_dir / f"ablation_{abl_id}.md"
        md_path.write_text(f"# {title}\n\n{md}\n", encoding="utf-8")
        write_csv_table(rows, model_order, args.output_dir / f"ablation_{abl_id}.csv")
        log.info("Ablation table '%s' written.", abl_id)

    log.info("All tables in %s", args.output_dir)

    # Print Table 1 to terminal for a quick sanity check
    print("\n" + "=" * 60)
    print("Table 1 — Main Model Comparison")
    print("=" * 60)
    print(table1_md)
    print()


if __name__ == "__main__":
    main()
