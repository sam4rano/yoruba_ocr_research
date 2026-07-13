# Pipeline shell orchestration

Phased bash drivers live here so you can run one stage at a time (reproducibility, HPC, Colab) or the full chain.

**Benchmark Architecture:** We evaluate Yorùbá line crops with Base PaddleOCR English-pretrained recognition, PaddleOCR-VL-1.6 zero-shot, GLM-OCR zero-shot, and PaddleOCR-VL-1.6 SFT. Optional PaddleOCR recognition fine-tuning is a classical comparison, not part of the default run. Numeric prefixes are reserved for orchestration phases; Python entry points use descriptive names.

| Script | Phase |
|--------|--------|
| `phase_01_consolidate.sh` | Merge `data/raw` → `data/processed` |
| `phase_02_analyze.sh` | EDA + plots → `results/tables/figures/` |
| `phase_03_config.sh` | `generate_paddleocr_config.py` + pretrained download |
| `phase_04_train_paddleocr_recognition.sh` | optional PaddleOCR recognition fine-tuning via `train_paddleocr_recognition.py` |
| `phase_05_eval_paddleocr_recognition.sh` | PaddleOCR EN pretrained + optional recognition fine-tune eval |
| `phase_15_eval_paddleocrvl16_zero_shot.sh` | PaddleOCR-VL-1.6 zero-shot eval |
| `phase_18_eval_glm_ocr_zero_shot.sh` | GLM-OCR zero-shot eval |
| `phase_14_export_paddleocrvl16_sft.sh` | Export SFT JSONL for PaddleOCR-VL-1.6 |
| `phase_16_train_paddleocrvl16_sft.sh` | PaddleOCR-VL-1.6 supervised adaptation (`lm_head` by default) |
| `phase_17_eval_paddleocrvl16_sft.sh` | Evaluate PaddleOCR-VL-1.6 SFT checkpoint |
| `phase_09_compile.sh` | `compile_results.py` → table Markdown/CSV |
| `phase_12_diagnose.sh` | Data vs eval vs setup diagnostics (`diagnose_experiment.py`) |
| `phase_13_verify_eval.sh` | `metrics.csv` ``n`` vs label files (`verify_eval_alignment.py`) |
| `phase_99_backup.sh` | Copy `results/` to `DRIVE_BACKUP_ROOT` |
| `run_all.sh` | Runs phases in order (override with `PHASES="..."`) |

### Default vs VLM baselines

The VLM/SFT phases require GPU and Hugging Face downloads. They are not enabled by default in `run_all.sh` to avoid CPU-only failures and accidental long jobs.

To run the complete benchmark:
1. Standard pipeline (Consolidate + Pretrained Paddle + Compile):
   ```bash
   bash scripts/shell/run_all.sh
   ```
2. Run VLM zero-shot baselines:
   ```bash
   bash scripts/shell/phase_15_eval_paddleocrvl16_zero_shot.sh
   bash scripts/shell/phase_18_eval_glm_ocr_zero_shot.sh
   ```
3. Re-run compile:
   ```bash
   bash scripts/shell/phase_09_compile.sh
   ```
4. Optional PaddleOCR-VL-1.6 SFT:
   ```bash
   bash scripts/shell/phase_14_export_paddleocrvl16_sft.sh
   bash scripts/shell/phase_16_train_paddleocrvl16_sft.sh
   bash scripts/shell/phase_17_eval_paddleocrvl16_sft.sh
   bash scripts/shell/phase_09_compile.sh
   ```

## Usage

From the repository root:

```bash
bash scripts/shell/phase_02_analyze.sh
```

## Environment variables (common)

| Variable | Purpose |
|----------|---------|
| `PROJECT_ROOT` | Repo root (default: inferred from script location) |
| `PYTHON` | Interpreter (default: `python3`) |
| `PROCESSED_DIR` | Dataset root (default: `data/processed`) |
| `RESULTS_TABLES_DIR` | Metrics/JSONL/meta evidence directory (default: `results/tables`) |
| `SKIP_CONSOLIDATE` | `1` = skip phase 01 |
| `CONFIG_FORCE_GPU` | `1` → `--force-gpu` in config generation |
| `CONFIG_CPU` | `1` → `--cpu` in config generation (macOS CPU Paddle) |
| `SKIP_PADDLE_TRAIN` | `0` enables optional phase 04 training (default `1`) |
| `TRAIN_CPU` | `1` → `--cpu` on `train_paddleocr_recognition.py` |
| `TRAIN_GPUS` | e.g. `0` (default) |
| `EVAL_USE_GPU` | `1` → `--use-gpu` on Paddle eval |
| `SKIP_PADDLEOCRVL16_ZERO_SHOT`| `1` to skip PaddleOCR-VL-1.6 zero-shot (default `0` when phase 15 runs) |
| `SKIP_GLM_ZERO_SHOT` | `1` to skip GLM-OCR (default `0` when phase 18 runs) |
| `SKIP_PADDLEOCRVL16_SFT_EXPORT` | `1` to skip phase 14 SFT export |
| `SKIP_PADDLEOCRVL16_SFT_TRAIN` | `1` to skip phase 16 SFT training |
| `SKIP_PADDLEOCRVL16_SFT_EVAL` | `1` to skip phase 17 SFT eval |
| `PADDLEOCRVL16_SFT_EPOCHS` | Epochs for phase 16 SFT (default `5`) |
| `PADDLEOCRVL16_SFT_TRAIN_SCOPE` | Trainable scope for phase 16 (default `lm_head`; use `non_vision` only on larger GPUs) |
| `PADDLEOCRVL16_BATCH_SIZE` / `GLM_BATCH_SIZE` | Zero-shot inference batch size (default `4`; failed batches retry at size 1) |
| `PADDLEOCRVL16_EVAL_BATCH_SIZE` | SFT checkpoint evaluation batch size (default `4`) |
| `PADDLEOCRVL16_NO_RESUME` / `GLM_NO_RESUME` | `1` discards compatible partial inference checkpoints |
| `PADDLEOCRVL16_ALLOW_FAILURES` / `GLM_ALLOW_FAILURES` | `1` explicitly permits publishing metrics with failed samples as empty predictions |
| `SKIP_COMPLETED_EVAL` | `1` skips model evaluation only when metrics, JSONL, metadata, and current split counts all agree |
| `DRIVE_BACKUP_ROOT` | Parent directory for timestamped backup (phase 99) |
| `BACKUP_EXPERIMENTS` | `1` (default) includes `experiments/` in backup |
| `GIT_SNAPSHOT` | `1` + phase 99 runs optional `git commit` on `results/tables` |
| `VERIFY_STRICT` | `1` + phase 13 exits non-zero if `n` ≠ current label pair count |

## Repo vs Drive

- **Repo:** Running phases from `PROJECT_ROOT` writes directly into `results/` and `experiments/` in the working tree.
- **Drive (or any second path):** Set `DRIVE_BACKUP_ROOT` and run phase `99` (included in `run_all.sh`) to copy artifacts. The last backup path is stored in `results/tables/.last_drive_backup_path.txt`.

**Strict alignment (before submission):** `VERIFY_STRICT=1 bash scripts/shell/phase_13_verify_eval.sh` — fails if any stored `n` ≠ current pair count.
