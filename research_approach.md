# Yorùbá OCR — Research Approach & Run Log

**Generated (UTC):** 2026-06-09T17:51:44Z  
**Git commit:** `5418dd8`  
**Project root:** `/Users/mac/Desktop/yoruba_ocr_research`  

## Workflow

1. Code synced from GitHub into the Drive repo folder (`git fetch` + `reset --hard`).
2. `data/` on Drive is **not** in git — uploads persist across pulls.
3. Models evaluated on `data/processed/` test split; metrics append to `results/tables/metrics.csv`.
4. `scripts/11_compile_results.py` builds Table 1 (+ ablation tables 2–4 when Phase 04/08 ran).
5. Analysis scripts 17–19 add bootstrap CIs, stratified DER, DER-universe ablation.
6. Timestamped backup under `My Drive/yoruba_ocr_backups/` (Phase 99).

## Dataset (this run)

- Unique line crops (consolidation): **2945**
- Split counts: train=2367, val=252, test=326
- Character dict size: 99
- Resplit enabled: unknown

## Pipeline toggles

- (no toggle env vars recorded)

## Primary supervised model

PaddleOCR-VL-1.5 LoRA (`paddleocr_vl15_lora_finetuned`): export (14) → zero-shot eval (15) → LoRA train (16) → adapter eval (15). Classical comparison: PP-OCRv4 CRNN when Phase 04 enabled.

## Table 1 — headline metrics (test split)

_Test lines n=326 (from consolidation_report; Table 1 uses eval-time n in metrics_summary)._

| display_name | cer_pct | wer_pct | der_pct | n |
| --- | --- | --- | --- | --- |
| Tesseract (eng) | 120.3 | 153.5 | 98.5 | 326 |
| Tesseract (yor) | 124.4 | 163.7 | 87.7 | 326 |
| Tesseract (eng+yor) | 122.6 | 160.0 | 93.9 | 326 |
| PaddleOCR-VL-1.5 (zero-shot) | 543.3 | 840.9 | 200.9 | 326 |
| Qwen 2.5 VL (zero-shot) | 253.5 | 329.5 | 119.6 | 326 |
| PaddleOCR-VL-1.5 (LoRA fine-tuned — main supervised) | 96.5 | 122.6 | 66.4 | 326 |

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
- **Ablation data size:** `results/tables/ablation_data_size.csv` (yes)
- **Ablation dictionary:** `results/tables/ablation_dictionary.csv` (missing)
- **Ablation augmentation:** `results/tables/ablation_augmentation.csv` (missing)
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
| Tesseract (eng/yor/eng+yor) | `07_baseline_tesseract.py` |
| PaddleOCR EN pretrained | `05_evaluate.py` |
| PP-OCRv4 CRNN fine-tuned | `04_train` + `05_evaluate.py` |
| PaddleOCR-VL-1.5 zero-shot / LoRA | `15_baseline_paddleocr_vl15.py`, `16_train_paddleocr_vl_lora.py` |
| Qwen 2.5-VL zero-shot | `09_baseline_qwen.py` |
| TrOCR-large-printed | `21_train_trocr.py`, `22_evaluate_trocr.py` |
| Surya v2 (local only) | `20_baseline_surya_v2.py` |

