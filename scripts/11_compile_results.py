"""
Compile all evaluation results into the paper's final comparison tables.

Reads results/tables/metrics.csv (written by all evaluation scripts)
and produces:

  Table 1 — Main Comparison (test split)
    Rows are ordered: OCR baselines → zero-shot multimodal models → **primary
    supervised result (PaddleOCR-VL-1.5 LoRA)** → classical PP-OCRv4 CRNN
    fine-tune as comparison. Columns: Model | CER ↓ | WER ↓ | DER ↓

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
import json
import logging
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from metrics_lifecycle import checkpoint_status_for_row  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Display names for the model labels used across all eval scripts
MODEL_DISPLAY = {
    "baseline_english_pretrained": "PaddleOCR PP-OCRv4 (EN pretrained)",
    "paddleocr_vl15_zero_shot": "PaddleOCR-VL-1.5 (zero-shot)",
    "qwen25_vl3_zero_shot": "Qwen 2.5 VL-3B (zero-shot)",
    "qwen25_vl_zero_shot": "Qwen 2.5 VL-7B (zero-shot, archived pilot)",
    "trocr_large_printed_zero_shot": "TrOCR-large-printed (zero-shot)",
    "surya_v2_zero_shot": "Surya v2 (zero-shot, recognition-only)",
    "trocr_large_printed_finetuned": "TrOCR-large-printed (fine-tuned)",
    "surya_finetuned": "Surya (Foundation fine-tuned)",
    "finetuned_paddleocr_v1": "PaddleOCR PP-OCRv4 (CRNN fine-tuned — comparison)",
    "paddleocr_vl15_lora_finetuned": "PaddleOCR-VL-1.5 (LoRA fine-tuned — main supervised)",
    # Ablation data size (PP-OCRv4 CRNN — not VL-1.5)
    "ablation_data_size_025pct_test": "PP-OCRv4 fine-tuned — 25% data",
    "ablation_data_size_050pct_test": "PP-OCRv4 fine-tuned — 50% data",
    "ablation_data_size_075pct_test": "PP-OCRv4 fine-tuned — 75% data",
    "ablation_data_size_100pct_test": "PP-OCRv4 fine-tuned — 100% data",
    # Ablation dictionary
    "ablation_dict_yoruba_dict_test": "PP-OCRv4 + Yorùbá dict",
    "ablation_dict_english_dict_test": "PP-OCRv4 + English dict",
    # Ablation augmentation
    "ablation_aug_with_aug_test": "PP-OCRv4 + RecAug",
    "ablation_aug_no_aug_test": "PP-OCRv4 − RecAug",
}

# Ordered model rows for Table 1 (supervised VL LoRA before CRNN fine-tune for narrative)
TABLE1_ORDER = [
    "baseline_english_pretrained",
    "trocr_large_printed_zero_shot",
    "surya_v2_zero_shot",
    "paddleocr_vl15_zero_shot",
    "qwen25_vl3_zero_shot",
    "paddleocr_vl15_lora_finetuned",
    "surya_finetuned",
    "trocr_large_printed_finetuned",
    "finetuned_paddleocr_v1",
]

# When the canonical eval name is absent, borrow from an equivalent run.
TABLE1_ALIASES: dict[str, str] = {
    "finetuned_paddleocr_v1": "ablation_data_size_100pct_test",
    "qwen25_vl3_zero_shot": "qwen25_vl_zero_shot",
}

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
            f"Results file not found: {csv_path}\n" "Run the evaluation scripts first."
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
            if not chosen.get("median_cer") or not chosen.get("micro_cer"):
                split = chosen.get("split", "test")
                jsonl_path = csv_path.parent / f"{model}_{split}.jsonl"
                if jsonl_path.is_file():
                    try:
                        pairs = []
                        with jsonl_path.open("r", encoding="utf-8") as fh:
                            for line in fh:
                                line = line.strip()
                                if not line:
                                    continue
                                item = json.loads(line)
                                pred = item.get("pred", "")
                                gt = item.get("gt", "")
                                pairs.append((pred, gt))
                        if pairs:
                            from evaluate_utils import aggregate_metrics
                            agg = aggregate_metrics(pairs)
                            chosen["median_cer"] = str(agg["median_cer"]) if agg["median_cer"] is not None else ""
                            chosen["median_wer"] = str(agg["median_wer"]) if agg["median_wer"] is not None else ""
                            chosen["micro_cer"] = str(agg["micro_cer"]) if agg["micro_cer"] is not None else ""
                            chosen["micro_wer"] = str(agg["micro_wer"]) if agg["micro_wer"] is not None else ""
                    except Exception as e:
                        log.warning("Could not re-calculate metrics from jsonl for %s: %s", jsonl_path, e)
            records[model] = chosen
    return records


def prepare_table1_records(
    records: dict[str, dict],
    tables_dir: Path,
) -> dict[str, dict]:
    """Build Table 1 rows: apply aliases; skip phantom and stale checkpoints."""
    out: dict[str, dict] = {}
    for model_key in TABLE1_ORDER:
        row = records.get(model_key)
        alias_used = None
        if row is None and model_key in TABLE1_ALIASES:
            alias = TABLE1_ALIASES[model_key]
            row = records.get(alias)
            if row is not None:
                alias_used = alias
        if row is None:
            log.warning(
                "Table 1: no usable metrics row for: %s (run eval or check aliases).",
                model_key,
            )
            continue
        phantom = (row.get("phantom") or "").strip().lower()
        if phantom == "true":
            log.warning(
                "Table 1: excluding %s (phantom checkpoint).",
                model_key,
            )
            continue
        status = checkpoint_status_for_row(
            row, tables_dir, allow_stale_with_jsonl=True,
        )
        if status == "stale":
            log.warning(
                "Table 1: excluding %s (stale checkpoint%s).",
                model_key,
                f"; alias={alias_used}" if alias_used else "",
            )
            continue
        if alias_used:
            log.info(
                "Table 1: using %s metrics for %s.",
                alias_used,
                model_key,
            )
        out[model_key] = row
    return out


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
    rows: dict[str, dict],
    model_order: list[str],
) -> str:
    """
    Render a Markdown results table in the paper's standard format.

    | Model | CER ↓ | Median CER ↓ | Micro CER ↓ | WER ↓ | DER ↓ |
    Bold the best (lowest) value per metric column.
    """
    present = [m for m in model_order if m in rows]
    data_rows = [rows[m] for m in present]

    best_cer = best_column(data_rows, "cer")
    best_median_cer = best_column(data_rows, "median_cer")
    best_micro_cer = best_column(data_rows, "micro_cer")
    best_wer = best_column(data_rows, "wer")
    best_der = best_column(data_rows, "der")

    lines = [
        "| Model | CER ↓ | Median CER ↓ | Micro CER ↓ | WER ↓ | DER ↓ |",
        "|-------|------:|-------------:|------------:|------:|------:|",
    ]
    for model_key in present:
        r = rows[model_key]
        display = MODEL_DISPLAY.get(model_key, model_key)
        cer = format_cell(r.get("cer"), r["model"] == best_cer)
        median_cer = format_cell(r.get("median_cer"), r["model"] == best_median_cer)
        micro_cer = format_cell(r.get("micro_cer"), r["model"] == best_micro_cer)
        wer = format_cell(r.get("wer"), r["model"] == best_wer)
        der = format_cell(r.get("der"), r["model"] == best_der)
        lines.append(f"| {display} | {cer} | {median_cer} | {micro_cer} | {wer} | {der} |")

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
            fieldnames=[
                "model_label",
                "display_name",
                "cer_pct",
                "median_cer_pct",
                "micro_cer_pct",
                "wer_pct",
                "der_pct",
                "n",
                "der_n",
            ],
        )
        writer.writeheader()
        for key in present:
            r = rows[key]
            writer.writerow(
                {
                    "model_label": key,
                    "display_name": MODEL_DISPLAY.get(key, key),
                    "cer_pct": pct(r.get("cer")),
                    "median_cer_pct": pct(r.get("median_cer")),
                    "micro_cer_pct": pct(r.get("micro_cer")),
                    "wer_pct": pct(r.get("wer")),
                    "der_pct": pct(r.get("der")),
                    "n": r.get("n", ""),
                    "der_n": r.get("der_n", ""),
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

    table1_rows = prepare_table1_records(rows, args.results_csv.parent)
    missing = [m for m in TABLE1_ORDER if m not in table1_rows]
    if missing:
        log.warning(
            "Table 1: no usable metrics row for: %s (run eval or check aliases).",
            ", ".join(missing),
        )

    # --- Table 1: Main Comparison ---
    table1_md = render_markdown_table(table1_rows, TABLE1_ORDER)
    (args.output_dir / "table1_main_comparison.md").write_text(
        "# Table 1 — Main Model Comparison (test split)\n\n" + table1_md + "\n",
        encoding="utf-8",
    )
    table1_csv = args.output_dir / "table1_main_comparison.csv"
    write_csv_table(table1_rows, TABLE1_ORDER, table1_csv)
    # Alias for notebooks / older docs that expect this filename
    summary_alias = args.output_dir / "metrics_summary.csv"
    summary_alias.write_text(table1_csv.read_text(encoding="utf-8"), encoding="utf-8")
    log.info("Table 1 written (%s + metrics_summary.csv).", table1_csv.name)

    # --- Tables 2–4: Ablation Studies ---
    ablation_titles = {
        "data_size": "Table 2 — Ablation: Training Data Size (test split)",
        "dictionary": "Table 3 — Ablation: Character Dictionary (test split)",
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
