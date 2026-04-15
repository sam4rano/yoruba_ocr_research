# Pipeline shell orchestration

Phased bash drivers live here so you can run one stage at a time (reproducibility, HPC, Colab) or the full chain.

| Script | Phase |
|--------|--------|
| `phase_01_consolidate.sh` | Merge `data/raw` → `data/processed` |
| `phase_02_analyze.sh` | EDA + plots → `results/tables/figures/` |
| `phase_03_config.sh` | `03_generate_config.py` + pretrained download |
| `phase_04_train.sh` | `04_train_paddleocr.py` |
| `phase_05_eval_paddle.sh` | English pretrained + fine-tuned Paddle eval |
| `phase_06_tesseract.sh` | Tesseract baselines |
| `phase_07_qwen.sh` | Qwen VL zero-shot (off unless `SKIP_QWEN=0`) |
| `phase_08_ablation.sh` | Ablation study (off unless `SKIP_ABLATION=0`) |
| `phase_09_compile.sh` | `11_compile_results.py` → table Markdown/CSV |
| `phase_99_backup.sh` | Copy `results/` (+ optional `experiments/`) to `DRIVE_BACKUP_ROOT` |
| `run_all.sh` | Runs phases in order (override with `PHASES="..."`) |

## Usage

From the repository root:

```bash
bash scripts/shell/phase_02_analyze.sh
```

Full pipeline (example Colab T4):

```bash
export PROJECT_ROOT="/content/drive/MyDrive/yoruba_ocr_research"
cd "$PROJECT_ROOT"
export SKIP_CONSOLIDATE=1
export CONFIG_FORCE_GPU=1
export TRAIN_CPU=0
export EVAL_USE_GPU=1
export DRIVE_BACKUP_ROOT="/content/drive/MyDrive/yoruba_ocr_results_backup"
export SKIP_QWEN=1
export SKIP_ABLATION=1
bash scripts/shell/run_all.sh
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
| `SKIP_TESSERACT` | `1` = skip Tesseract |
| `SKIP_QWEN` | `0` to run Qwen (default `1`) |
| `SKIP_ABLATION` | `0` to run ablations (default `1`) |
| `DRIVE_BACKUP_ROOT` | Parent directory for timestamped backup (phase 99) |
| `BACKUP_EXPERIMENTS` | `1` (default) includes `experiments/` in backup (large) |
| `GIT_SNAPSHOT` | `1` + phase 99 runs optional `git commit` on `results/tables` |

## Repo vs Drive

- **Repo:** Running phases from `PROJECT_ROOT` writes directly into `results/` and `experiments/` in the working tree. Commit what your policy allows (often `results/tables/*` only; large checkpoints via Drive or Git LFS).
- **Drive (or any second path):** Set `DRIVE_BACKUP_ROOT` and run phase `99` (included in `run_all.sh`) to copy artifacts. The last backup path is stored in `results/tables/.last_drive_backup_path.txt`.

See also `docs/colab_pro_t4.md`.
