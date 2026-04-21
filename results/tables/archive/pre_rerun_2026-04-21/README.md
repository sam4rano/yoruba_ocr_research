# Pre-rerun snapshot — 2026-04-21

Snapshot of every per-run artifact under `results/tables/` immediately before the
full clean re-execution following the P1–P6 forensic fixes and the Tier A/B/C
codebase cleanup.

## What is in here

* **Per-sample prediction JSONLs** from the prior pipeline (PP-OCR CRNN
  fine-tune, the English pretrained baseline, ablations, and the April 21
  Tesseract runs). Schema matches what `scripts/evaluate_utils.py::save_results`
  wrote at the time; older rows predate the `phantom` / `der_n` /
  `der_insertion_rate` columns and should not be mixed with rerun rows.
* **Compiled tables** produced by `scripts/11_compile_results.py` and friends:
  `metrics_summary.csv`, `table1_main_comparison.{csv,md}`,
  `ablation_{augmentation,data_size,dictionary}.{csv,md}`. These were derived
  from `metrics.csv` which itself is preserved one level up under
  `archive/pre_integrity/` (the P1 integrity cut-over) and
  `archive/pre_der_conditional/` (the P6 corpus-level DER cut-over).
* **Diagnostics**: `diagnose_data_inventory.json`,
  `diagnose_sample_for_review.jsonl`, `eval_alignment_report.json`,
  `train_run.json`.
* **Baseline provenance**: `meta/tesseract_*.json` — the first runs to carry
  the new `phantom="n/a"` provenance after P4/P5 landed.

## What is NOT here

These files stayed in `results/tables/` because they describe the current
pipeline state (not a historical run) and will be overwritten cleanly on the
next run:

* `consolidation_report.json`, `dataset_analysis.json`, `data_quality.json`
* `config_generation.json`, `checkpoint_audit.json`

## How the rerun starts fresh

* `results/tables/metrics.csv` is intentionally absent (P1 integrity cut-over
  lives in `archive/pre_integrity/metrics.csv`). The first new eval will
  recreate it with the updated schema
  (`phantom`, `der`, `der_n`, `der_insertion_rate`, `prompt_sha256`,
  `adapter_sha256`, ...).
* `results/tables/meta/` is empty; new `meta.json` sidecars will be written
  per-run.
* All prediction JSONLs and compiled tables listed above will be regenerated
  by phases 04 through 09.
