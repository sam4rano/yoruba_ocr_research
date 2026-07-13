"""
Write research_approach.md documenting a Colab/local experiment run.

Aggregates git provenance, pipeline toggles, dataset counts, model rows in
metrics.csv, and paths to every paper table artifact under results/tables/.

Usage:
    python scripts/write_research_approach.py
    python scripts/write_research_approach.py --output research_approach.md
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
    ("Figure 1 — main comparison plot", "figures/model_metrics_comparison.png"),
    ("Figure 2 — bootstrap intervals plot", "figures/bootstrap_confidence_intervals.png"),
    ("Figure 3 — stratified density plot", "figures/stratified_der_by_density.png"),
    ("Figure 4 — error taxonomy plot", "figures/error_taxonomy_distribution.png"),
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
        if k.startswith(
            (
                "SKIP_",
                "RUN_",
                "RESET_",
                "PADDLEOCRVL16_",
                "GLM_",
                "USE_EXISTING",
            )
        )
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

    resplit_val = "unknown"
    if "resplit" in consolidation:
        resplit_val = str(consolidation["resplit"].get("enabled", "unknown"))
    elif "split_source" in consolidation:
        resplit_val = "false" if consolidation["split_source"] == "on_disk_labels" else "true"

    body = [
        "# Yorùbá OCR — Research Approach & Run Log",
        "",
        f"**Generated (UTC):** {now}  ",
        f"**Git commit:** `{git_head()}`  ",
        f"**Project root:** `{Path.cwd()}`  ",
        "",
        "## Workflow",
        "",
        "1. Record the clean repository commit before starting the final run.",
        "2. Keep the frozen `data/processed/` split unchanged across all model rows.",
        "3. Evaluate each model with resumable per-sample logs; failed samples block publication by default.",
        "4. `scripts/compile_results.py` builds Table 1 from traceable zero-shot and supervised rows.",
        "5. The stratified-error, DER-universe, and bootstrap scripts add robustness analyses.",
        "6. Phase 99 optionally writes a timestamped backup to `DRIVE_BACKUP_ROOT`.",
        "",
        "## Dataset (this run)",
        "",
        f"- Unique line crops (consolidation): **{consolidation.get('unique_images_total', '?')}**",
        f"- Split counts: train={split_counts.get('train', '?')}, "
        f"val={split_counts.get('val', '?')}, test={split_counts.get('test', '?')}",
        f"- Character dict size: {consolidation.get('char_dict_size', '?')}",
        f"- Resplit enabled: {resplit_val}",
        "",
        "## Pipeline toggles",
        "",
        *toggle_lines(),
        "",
        "## Benchmark Architecture",
        "",
        "The benchmark evaluates OCR on Yorùbá line crops with PaddleOCR English-pretrained recognition, PaddleOCR-VL-1.6 zero-shot, GLM-OCR zero-shot, and PaddleOCR-VL-1.6 supervised adaptation. The default SFT scope updates `lm_head`; broader scopes are separate experiments and are recorded in metadata.",
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
        "| PaddleOCR EN pretrained | `scripts/evaluate_paddleocr_en_pretrained.py` |",
        "| PaddleOCR-VL-1.6 zero-shot | `scripts/eval_paddleocrvl16.py` |",
        "| GLM-OCR zero-shot | `scripts/eval_glm_ocr.py` |",
        "| PaddleOCR-VL-1.6 SFT | `scripts/train_paddleocrvl16_sft.py` (train), `scripts/eval_paddleocrvl16.py` (eval) |",
        "",
    ]

    args.output.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
