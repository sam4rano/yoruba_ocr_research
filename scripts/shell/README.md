# Pipeline shell orchestration

Phased bash drivers live here so you can run one stage at a time (reproducibility, HPC, Colab) or the full chain.

**Benchmark Architecture:** We evaluate zero-shot OCR on Yorùbá line crops across three main model classes: Base PaddleOCR (PP-OCRv4 EN pretrained + CRNN fine-tune ablations), PaddleOCR-VL-1.6, and GLM-OCR.

| Script | Phase |
|--------|--------|
| `phase_01_consolidate.sh` | Merge `data/raw` → `data/processed` |
| `phase_02_analyze.sh` | EDA + plots → `results/tables/figures/` |
| `phase_03_config.sh` | `03_generate_config.py` + pretrained download |
| `phase_04_train.sh` | `04_train_paddleocr.py` |
| `phase_05_eval_paddle.sh` | English pretrained + fine-tuned Paddle eval |
| `phase_15_eval_vl16.sh` | PaddleOCR-VL-1.6 zero-shot eval |
| `phase_glm_ocr.sh` | GLM-OCR zero-shot eval |
| `phase_08_ablation.sh` | Ablation study (off unless `SKIP_ABLATION=0`) |
| `phase_09_compile.sh` | `11_compile_results.py` → table Markdown/CSV |
| `phase_12_diagnose.sh` | Data vs eval vs setup diagnostics (`12_diagnose_hypotheses.py`) |
| `phase_13_verify_eval.sh` | `metrics.csv` ``n`` vs label files (`13_verify_eval_alignment.py`) |
| `phase_99_backup.sh` | Copy `results/` to `DRIVE_BACKUP_ROOT` |
| `run_all.sh` | Runs phases in order (override with `PHASES="..."`) |

### Default vs VLM baselines

The baseline VLM/recognition models (VL-1.6, GLM-OCR) require GPU or specialized inference environments. Therefore, they are not enabled by default in `run_all.sh` to avoid CPU-only run failures.

To run the complete benchmark:
1. Standard pipeline (Consolidate + Pretrained Paddle + Ablations + Compile):
   ```bash
   bash scripts/shell/run_all.sh
   ```
2. Run VLM / zero-shot recognition baselines:
   ```bash
   bash scripts/shell/phase_15_eval_vl16.sh
   bash scripts/shell/phase_glm_ocr.sh
   ```
3. Re-run compile:
   ```bash
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
| `SKIP_CONSOLIDATE` | `1` = skip phase 01 |
| `CONFIG_FORCE_GPU` | `1` → `--force-gpu` in config generation |
| `CONFIG_CPU` | `1` → `--cpu` in config generation (macOS CPU Paddle) |
| `TRAIN_CPU` | `1` → `--cpu` on `04_train_paddleocr.py` |
| `TRAIN_GPUS` | e.g. `0` (default) |
| `EVAL_USE_GPU` | `1` → `--use-gpu` on Paddle eval |
| `SKIP_VL16_ZERO_SHOT`| `1` to skip VL-1.6 (default `0` when phase_15 run) |
| `SKIP_GLM_ZERO_SHOT` | `1` to skip GLM-OCR (default `0` when phase_glm run) |
| `SKIP_ABLATION` | `0` to run ablations (default `1`) |
| `DRIVE_BACKUP_ROOT` | Parent directory for timestamped backup (phase 99) |
| `BACKUP_EXPERIMENTS` | `1` (default) includes `experiments/` in backup |
| `GIT_SNAPSHOT` | `1` + phase 99 runs optional `git commit` on `results/tables` |
| `VERIFY_STRICT` | `1` + phase 13 exits non-zero if `n` ≠ current label pair count |

## Repo vs Drive

- **Repo:** Running phases from `PROJECT_ROOT` writes directly into `results/` and `experiments/` in the working tree.
- **Drive (or any second path):** Set `DRIVE_BACKUP_ROOT` and run phase `99` (included in `run_all.sh`) to copy artifacts. The last backup path is stored in `results/tables/.last_drive_backup_path.txt`.

**Strict alignment (before submission):** `VERIFY_STRICT=1 bash scripts/shell/phase_13_verify_eval.sh` — fails if any stored `n` ≠ current pair count.
