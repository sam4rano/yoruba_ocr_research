"""
Archive and reset evaluation artifacts between Colab re-runs.

Copies ``metrics.csv``, Table 1 outputs, per-model JSONL logs, and meta JSON
to ``results/tables/archive/pre_run_<timestamp>/``, then resets ``metrics.csv``
to header-only and removes working JSONL, meta, and compiled table snapshots
so partial re-runs cannot mix stale and fresh inference logs.

Usage:
    python scripts/metrics_lifecycle.py archive
    python scripts/metrics_lifecycle.py reset
    python scripts/metrics_lifecycle.py archive --tables-dir results/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

METRICS_HEADER = [
    "model",
    "split",
    "n",
    "cer",
    "wer",
    "der",
    "der_n",
    "der_insertion_rate",
    "median_cer",
    "median_wer",
    "micro_cer",
    "micro_wer",
    "phantom",
    "meta_path",
    "timestamp",
]

TABLE_SNAPSHOTS = (
    "metrics.csv",
    "bootstrap_metric_cis.json",
    "consolidation_report.json",
    "data_quality.json",
    "metrics_summary.csv",
    "table1_main_comparison.csv",
    "table1_main_comparison.md",
    "bootstrap_metric_cis.csv",
    "bootstrap_pairwise_comparison.csv",
    "stratified_der_by_density.csv",
    "stratified_der_by_book.csv",
    "der_universe_ablation.csv",
    "eval_alignment_report.json",
    "checkpoint_audit.json",
)

GENERATED_ARTIFACTS = (
    *TABLE_SNAPSHOTS,
    "paddleocr_en_pretrained_test.jsonl",
    "paddleocrvl16_zero_shot_test.jsonl",
    "glm_ocr_zero_shot_test.jsonl",
    "paddleocrvl16_sft_test.jsonl",
    "minimal_pair_subset.csv",
    "minimal_pair_vocabulary.json",
    "stratified_by_linguistic_features.csv",
    "stratified_error_analysis.json",
    "error_taxonomy.csv",
    "der_universe_ablation.json",
    "der_zero_diac_insertion.csv",
    "colab_smoke_test.json",
    "config_generation.json",
    "e2e_network_check.json",
    "hf_dataset_upload.json",
    ".DS_Store",
)


def checkpoint_status_for_row(
    row: dict,
    tables_dir: Path,
    *,
    allow_stale_with_jsonl: bool = False,
) -> str:
    """
    Return checkpoint audit status for one metrics.csv row.

    Mirrors ``diagnose_experiment.py checkpoints`` logic:
    ``ok`` | ``phantom`` | ``stale`` | ``no_meta``.

    Non-Paddle models (phantom="n/a") never produce ``.pdparams`` files;
    they are always ``ok`` if their meta exists and is not phantom.

    When ``allow_stale_with_jsonl=True``, a row whose checkpoint is
    missing on disk is downgraded from ``stale`` to ``ok`` if the
    per-sample JSONL log still exists — the inference evidence is
    preserved even though the model weights were deleted (common on
    ephemeral Colab disk).
    """
    model = row.get("model", "")
    split = row.get("split", "")
    csv_phantom = (row.get("phantom") or "").strip().lower()
    meta_path = tables_dir / "meta" / f"{model}_{split}.json"

    # Non-Paddle/PyTorch VLM models have phantom="n/a".
    # They never produce .pdparams files, so skip checkpoint verification.
    if csv_phantom == "n/a":
        if not meta_path.is_file():
            return "no_meta"
        return "ok"

    if not meta_path.is_file():
        return "phantom" if csv_phantom == "true" else "no_meta"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "no_meta"

    if str(meta.get("phantom", "")).lower() == "true":
        return "phantom"

    # Non-Paddle model kinds also use phantom="n/a" in meta.
    if str(meta.get("phantom", "")).lower() == "n/a":
        return "ok"

    ckpt_path = (meta.get("artifacts") or {}).get("checkpoint_pdparams")
    if ckpt_path and not Path(ckpt_path).is_file():
        if allow_stale_with_jsonl:
            jsonl_path = tables_dir / f"{model}_{split}.jsonl"
            if jsonl_path.is_file():
                return "ok"
        return "stale"

    return "ok"


def archive_eval_artifacts(tables_dir: Path) -> Path:
    """
    Copy current metrics/tables/jsonl/meta to archive and reset metrics.csv.

    Returns the archive directory path.
    """
    tables_dir = tables_dir.resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = tables_dir / "archive" / f"pre_run_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)

    for name in TABLE_SNAPSHOTS:
        src = tables_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)

    jsonl_archived: list[Path] = []
    for pattern in (
        "*_test.jsonl",
        "*_val.jsonl",
        "*_train.jsonl",
        "*.jsonl.partial",
    ):
        for src in sorted(tables_dir.glob(pattern)):
            shutil.copy2(src, dest / src.name)
            jsonl_archived.append(src)

    meta_src = tables_dir / "meta"
    meta_archived: list[Path] = []
    if meta_src.is_dir():
        meta_dest = dest / "meta"
        meta_dest.mkdir(parents=True, exist_ok=True)
        for src in meta_src.glob("*.json"):
            shutil.copy2(src, meta_dest / src.name)
            meta_archived.append(src)

    tables_removed: list[Path] = []
    for name in TABLE_SNAPSHOTS:
        src = tables_dir / name
        if src.is_file() and name != "metrics.csv":
            src.unlink()
            tables_removed.append(src)
    if tables_removed:
        log.info(
            "Removed %d compiled table snapshot(s) from %s (re-run compile/analysis after eval).",
            len(tables_removed),
            tables_dir,
        )

    metrics = tables_dir / "metrics.csv"
    if metrics.is_file():
        shutil.copy2(metrics, dest / "metrics.csv")
        with metrics.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=METRICS_HEADER)
            writer.writeheader()
        log.info("Reset %s to header-only.", metrics)

    for path in jsonl_archived:
        path.unlink()
    if jsonl_archived:
        log.info("Removed %d JSONL log(s) from %s (copied to archive).", len(jsonl_archived), tables_dir)

    for path in meta_archived:
        path.unlink()
    if meta_archived:
        log.info("Removed %d meta JSON file(s) from %s.", len(meta_archived), meta_src)

    marker = tables_dir / ".last_metrics_archive.txt"
    marker.write_text(str(dest) + "\n", encoding="utf-8")
    log.info("Archived eval artifacts to %s", dest)
    return dest


def reset_generated_artifacts(tables_dir: Path) -> None:
    """
    Remove generated reports/logs and reset metrics.csv to a schema-only file.

    This is intentionally destructive and is meant for the "fresh paper run"
    case where stale metrics should not survive into regenerated outputs.
    Source data, raw exports, configs, and scripts are not touched.
    """
    tables_dir = tables_dir.resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)

    removed = 0
    for name in GENERATED_ARTIFACTS:
        path = tables_dir / name
        if path.is_file() and name != "metrics.csv":
            path.unlink()
            removed += 1

    for pattern in (
        "*_test.jsonl",
        "*_val.jsonl",
        "*_train.jsonl",
        "*.jsonl.partial",
    ):
        for path in tables_dir.glob(pattern):
            path.unlink()
            removed += 1

    metrics = tables_dir / "metrics.csv"
    with metrics.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRICS_HEADER)
        writer.writeheader()

    meta_dir = tables_dir / "meta"
    if meta_dir.is_dir():
        for path in meta_dir.glob("*.json"):
            path.unlink()
            removed += 1

    fig_dir = tables_dir / "figures"
    if fig_dir.is_dir():
        for extension in ("*.png", "*.pdf", "*.svg"):
            for path in fig_dir.glob(extension):
                path.unlink()
                removed += 1

    for path in (tables_dir.parent / ".DS_Store", tables_dir / ".DS_Store"):
        if path.is_file():
            path.unlink()
            removed += 1

    marker = tables_dir / ".last_metrics_archive.txt"
    if marker.is_file():
        marker.unlink()

    log.info("Reset %s to header-only and removed %d generated artifact(s).", metrics, removed)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Archive and reset metrics between runs.")
    parser.add_argument(
        "command",
        choices=("archive", "reset"),
        help="archive copies current outputs first; reset removes stale outputs in place.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results/tables"),
        help="Results tables directory.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if args.command == "archive":
        dest = archive_eval_artifacts(args.tables_dir)
        print(dest)
    elif args.command == "reset":
        reset_generated_artifacts(args.tables_dir)


if __name__ == "__main__":
    main()
