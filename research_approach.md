# Yorùbá OCR — Research Approach & Run Log

**Generated (UTC):** 2026-06-20T10:05:54Z  
**Git commit:** `7d3d608`  
**Project root:** `/Users/mac/Desktop/yoruba_ocr_research`  

## Workflow

1. Code synced from GitHub into the Drive repo folder (`git fetch` + `reset --hard`).
2. `data/` on Drive is **not** in git — uploads persist across pulls.
3. Models evaluated on `data/processed/` test split; metrics append to `results/tables/metrics.csv`.
4. `scripts/11_compile_results.py` builds Table 1 (containing zero-shot models and fine-tuned PaddleOCR-VL-1.6).
5. Analysis scripts 17–19 add bootstrap CIs, stratified DER, DER-universe ablation.
6. Timestamped backup under `My Drive/yoruba_ocr_backups/` (Phase 99).

## Dataset (this run)

- Unique line crops (consolidation): **2945**
- Split counts: train=2367, val=252, test=326
- Character dict size: 99
- Resplit enabled: false

## Pipeline toggles

- (no toggle env vars recorded)

## Benchmark Architecture

The benchmark evaluates OCR on Yorùbá line crops across a 3-model zero-shot stack (Base PaddleOCR PP-OCRv4 EN pretrained, GLM-OCR, and PaddleOCR-VL-1.6) alongside direct fine-tuning of PaddleOCR-VL-1.6.

## Table 1 — headline metrics (test split)

_Test lines n=326 (from consolidation_report; Table 1 uses eval-time n in metrics_summary)._

| display_name | cer_pct | wer_pct | der_pct | n |
| --- | --- | --- | --- | --- |
| PaddleOCR PP-OCRv4 (EN pretrained) | 100.4 | 106.1 | 100.0 | 326 |

## Paper artifacts on disk

- **Table 1 — main comparison:** `results/tables/table1_main_comparison.csv` (yes)
- **Table 1 (markdown):** `results/tables/table1_main_comparison.md` (yes)
- **Metrics master log:** `results/tables/metrics.csv` (yes)
- **Metrics summary:** `results/tables/metrics_summary.csv` (yes)
- **Bootstrap CIs:** `results/tables/bootstrap_metric_cis.csv` (yes)
- **Bootstrap pairwise:** `results/tables/bootstrap_pairwise_comparison.csv` (yes)
- **Stratified DER by density:** `results/tables/stratified_der_by_density.csv` (yes)
- **Stratified DER by book:** `results/tables/stratified_der_by_book.csv` (yes)
- **Minimal-pair subset:** `results/tables/minimal_pair_subset.csv` (yes)
- **Error taxonomy:** `results/tables/error_taxonomy.csv` (yes)
- **DER universe ablation:** `results/tables/der_universe_ablation.csv` (yes)
- **DER zero-diac insertion:** `results/tables/der_zero_diac_insertion.csv` (yes)
- **Figure 1 — main comparison plot:** `results/tables/figures/model_metrics_comparison.png` (yes)
- **Figure 2 — bootstrap intervals plot:** `results/tables/figures/bootstrap_confidence_intervals.png` (yes)
- **Figure 3 — stratified density plot:** `results/tables/figures/stratified_der_by_density.png` (yes)
- **Figure 4 — error taxonomy plot:** `results/tables/figures/error_taxonomy_distribution.png` (yes)
- **Eval alignment report:** `results/tables/eval_alignment_report.json` (yes)
- **Checkpoint audit:** `results/tables/checkpoint_audit.json` (yes)
- **HF dataset upload manifest:** `results/tables/hf_dataset_upload.json` (yes)
- **Consolidation report:** `results/tables/consolidation_report.json` (yes)
- **Data quality audit:** `results/tables/data_quality.json` (yes)

## Metrics conventions

- **CER / WER / DER** — NFC-normalised; corpus-level rates in `metrics.csv`.
- **DER** — edit distance on combining diacritics only (see `docs/metrics_conventions.md`).
- **phantom** — `true` rows used a re-initialised Paddle CTC head; do not cite.

## Models in benchmark (script map)

| Model key | Script |
| --- | --- |
| PaddleOCR EN pretrained | `05_evaluate.py` |
| PaddleOCR-VL-1.6 zero-shot | `15_baseline_paddleocr_vl16.py` |
| GLM-OCR zero-shot | `16_baseline_glm_ocr.py` |
| PaddleOCR-VL-1.6 fine-tuned | `scripts/16_train_paddleocr_vl.py` (train), `scripts/15_baseline_paddleocr_vl16.py` (eval) |

