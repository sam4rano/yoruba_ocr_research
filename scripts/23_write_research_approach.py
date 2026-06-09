"""
Write research_approach.md documenting a Colab/local experiment run.

Aggregates git provenance, pipeline toggles, dataset counts, model rows in
metrics.csv, and paths to every paper table artifact under results/tables/.

Usage:
    python scripts/23_write_research_approach.py
    python scripts/23_write_research_approach.py --output research_approach.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

TABLE_ARTIFACTS = [
    ("Table 1 — main comparison", "table1_main_comparison.csv"),
    ("Table 1 (markdown)", "table1_main_comparison.md"),
    ("Metrics master log", "metrics.csv"),
    ("Metrics summary", "metrics_summary.csv"),
    ("Bootstrap CIs", "bootstrap_metric_cis.csv"),
    ("Bootstrap pairwise", "bootstrap_pairwise_comparison.csv"),
    ("Stratified DER by density", "stratified_der_by_density.csv"),
    ("Stratified DER by book", "stratified_der_by_book.csv"),
    ("Minimal-pair subset", "minimal_pair_subset.csv"),
    ("Error taxonomy", "error_taxonomy.csv"),
    ("DER universe ablation", "der_universe_ablation.csv"),
    ("DER zero-diac insertion", "der_zero_diac_insertion.csv"),
    ("Ablation data size", "ablation_data_size.csv"),
    ("Ablation dictionary", "ablation_dictionary.csv"),
    ("Ablation augmentation", "ablation_augmentation.csv"),
    ("Eval alignment report", "eval_alignment_report.json"),
    ("Checkpoint audit", "checkpoint_audit.json"),
    ("HF dataset upload manifest", "hf_dataset_upload.json"),
    ("Consolidation report", "consolidation_report.json"),
    ("Data quality audit", "data_quality.json"),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Write research_approach.md for a pipeline run.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research_approach.md"),
        help="Output markdown path (default: repo root research_approach.md).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory containing metrics and table CSVs.",
    )
    return parser.parse_args()


def git_head() -> str:
    """Return short git commit hash or placeholder."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_json(path: Path) -> dict | None:
    """Load JSON file if it exists."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def toggle_lines() -> list[str]:
    """Collect pipeline toggle env vars set for this run."""
    keys = sorted(
        k for k in os.environ
        if k.startswith(("SKIP_", "RUN_", "RESET_", "VL15_", "TROCR_", "QWEN_", "USE_EXISTING"))
    )
    lines = []
    for key in keys:
        lines.append(f"- `{key}` = `{os.environ.get(key, '')}`")
    return lines or ["- (no toggle env vars recorded)"]


def metrics_table(results_dir: Path) -> list[str]:
    """Build markdown lines for metrics_summary if present."""
    summary = results_dir / "metrics_summary.csv"
    if not summary.is_file():
        return ["_metrics_summary.csv not found — run Step 10 compile first._"]

    import csv

    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    if not rows:
        return ["_metrics_summary.csv is empty._"]

    cols = ["display_name", "cer_pct", "wer_pct", "der_pct", "n"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return lines


def artifact_lines(results_dir: Path) -> list[str]:
    """List expected table files and whether each exists."""
    lines = []
    for label, name in TABLE_ARTIFACTS:
        path = results_dir / name
        flag = "yes" if path.is_file() else "missing"
        lines.append(f"- **{label}:** `{path}` ({flag})")
    return lines


def main() -> None:
    """Write research_approach.md."""
    args = parse_args()
    results_dir = args.results_dir
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    consolidation = load_json(results_dir / "consolidation_report.json") or {}
    split_counts = consolidation.get("split_counts", {})
    n_test = split_counts.get("test", "?")

    body = [
        "# Yorùbá OCR — Research Approach & Run Log",
        "",
        f"**Generated (UTC):** {now}  ",
        f"**Git commit:** `{git_head()}`  ",
        f"**Project root:** `{Path.cwd()}`  ",
        "",
        "## Workflow",
        "",
        "1. Code synced from GitHub into the Drive repo folder (`git fetch` + `reset --hard`).",
        "2. `data/` on Drive is **not** in git — uploads persist across pulls.",
        "3. Models evaluated on `data/processed/` test split; metrics append to `results/tables/metrics.csv`.",
        "4. `scripts/11_compile_results.py` builds Table 1 (+ ablation tables 2–4 when Phase 04/08 ran).",
        "5. Analysis scripts 17–19 add bootstrap CIs, stratified DER, DER-universe ablation.",
        "6. Timestamped backup under `My Drive/yoruba_ocr_backups/` (Phase 99).",
        "",
        "## Dataset (this run)",
        "",
        f"- Unique line crops (consolidation): **{consolidation.get('unique_images_total', '?')}**",
        f"- Split counts: train={split_counts.get('train', '?')}, "
        f"val={split_counts.get('val', '?')}, test={split_counts.get('test', '?')}",
        f"- Character dict size: {consolidation.get('char_dict_size', '?')}",
        f"- Resplit enabled: {consolidation.get('resplit', {}).get('enabled', 'unknown')}",
        "",
        "## Pipeline toggles",
        "",
        *toggle_lines(),
        "",
        "## Primary supervised model",
        "",
        "PaddleOCR-VL-1.5 LoRA (`paddleocr_vl15_lora_finetuned`): export (14) → zero-shot eval (15) → "
        "LoRA train (16) → adapter eval (15). Classical comparison: PP-OCRv4 CRNN when Phase 04 enabled.",
        "",
        "## Table 1 — headline metrics (test split)",
        "",
        f"_Test lines n={n_test} (from consolidation_report; Table 1 uses eval-time n in metrics_summary)._",
        "",
        *metrics_table(results_dir),
        "",
        "## Paper artifacts on disk",
        "",
        *artifact_lines(results_dir),
        "",
        "## Metrics conventions",
        "",
        "- **CER / WER / DER** — NFC-normalised; corpus-level rates in `metrics.csv`.",
        "- **DER** — edit distance on combining diacritics only (see `docs/metrics_conventions.md`).",
        "- **phantom** — `true` rows used a re-initialised Paddle CTC head; do not cite.",
        "",
        "## Models in benchmark (script map)",
        "",
        "| Model key | Script |",
        "| --- | --- |",
        "| Tesseract (eng/yor/eng+yor) | `07_baseline_tesseract.py` |",
        "| PaddleOCR EN pretrained | `05_evaluate.py` |",
        "| PP-OCRv4 CRNN fine-tuned | `04_train` + `05_evaluate.py` |",
        "| PaddleOCR-VL-1.5 zero-shot / LoRA | `15_baseline_paddleocr_vl15.py`, `16_train_paddleocr_vl_lora.py` |",
        "| Qwen 2.5-VL zero-shot | `09_baseline_qwen.py` |",
        "| TrOCR-large-printed | `21_train_trocr.py`, `22_evaluate_trocr.py` (Phase 21) |",
        "| Surya v2 (zero-shot) | `20_baseline_surya_v2.py` (Phase 20) |",
        "",
    ]

    args.output.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
