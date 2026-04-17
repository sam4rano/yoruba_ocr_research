# Pipeline shell orchestration

Phased bash drivers live here so you can run one stage at a time (reproducibility, HPC, Colab) or the full chain.

**Primary fine-tuned system:** PaddleOCR-VL-1.5 LoRA (phases **14 → 16 → 15** with adapter). Phases **04–05** train/evaluate **PP-OCRv4** as a classical CRNN comparison; phase **08** ablations vary that stack only—not the VL model.

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
| `phase_09_compile.sh` | `11_compile_results.py` → table Markdown/CSV + alignment check |
| `phase_14_export_vl15.sh` | Export JSONL for PaddleOCR-VL-1.5 SFT (`data/paddleocr_vl15_sft/`) |
| `phase_15_eval_vl15.sh` | VL-1.5 zero-shot eval (off unless `SKIP_VL15_EVAL=0`) |
| `phase_16_train_vl15_lora.sh` | LoRA fine-tune VL-1.5 on phase-14 export |
| `phase_12_diagnose.sh` | Data vs eval vs setup diagnostics (`12_diagnose_hypotheses.py`) |
| `phase_13_verify_eval.sh` | `metrics.csv` ``n`` vs label files (`13_verify_eval_alignment.py`) |
| `phase_99_backup.sh` | Copy `results/` (+ optional `experiments/`) to `DRIVE_BACKUP_ROOT` |
| `run_all.sh` | Runs phases in order (override with `PHASES="..."`; supports `12`–`16`) |

### Default vs paper-complete (why VL phases are not in the default `run_all`)

The **primary supervised result** in the manuscript is PaddleOCR-VL-1.5 LoRA, but **default `PHASES`** in `run_all.sh` is still `01 … 08 09 99` (no **14–16**). That is intentional portability: phase **15** expects a GPU and `SKIP_VL15_EVAL=0`, phase **16** is a long HF job, and many environments only need PP-OCRv4 / Tesseract / compile. So “default run” ≠ “every Table 1 row populated” unless you add VL phases yourself.

**Paper-complete Table 1 including VL rows** (after the usual data + PP-OCRv4 path):

1. `bash scripts/shell/phase_14_export_vl15.sh`
2. `export SKIP_VL15_EVAL=0` and `bash scripts/shell/phase_15_eval_vl15.sh` (zero-shot `paddleocr_vl15_zero_shot`)
3. `bash scripts/shell/phase_16_train_vl15_lora.sh` (LoRA — long)
4. `python scripts/15_baseline_paddleocr_vl15.py --adapter-path experiments/paddleocr_vl15_lora/adapter` (main supervised `paddleocr_vl15_lora_finetuned`)
5. `bash scripts/shell/phase_09_compile.sh`

You can merge (1) and (2) into a custom `PHASES=...` that includes `14` and `15`, but (3)–(4) stay separate because `run_all` does not run Python eval with `--adapter-path` and 16 is usually scheduled as its own job.

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
| `SKIP_VL15_EVAL` | `0` to run PaddleOCR-VL-1.5 eval (default `1`) |
| `PADDLE_VL15_EXPORT_DIR` | Override export dir for phases 14/16 (default `data/paddleocr_vl15_sft`) |
| `SKIP_ABLATION` | `0` to run ablations (default `1`) |
| `DRIVE_BACKUP_ROOT` | Parent directory for timestamped backup (phase 99) |
| `BACKUP_EXPERIMENTS` | `1` (default) includes `experiments/` in backup (large) |
| `GIT_SNAPSHOT` | `1` + phase 99 runs optional `git commit` on `results/tables` |
| `VERIFY_STRICT` | `1` + phase 13 exits non-zero if `n` ≠ current label pair count |

## Repo vs Drive

- **Repo:** Running phases from `PROJECT_ROOT` writes directly into `results/` and `experiments/` in the working tree. Commit what your policy allows (often `results/tables/*` only; large checkpoints via Drive or Git LFS).
- **Drive (or any second path):** Set `DRIVE_BACKUP_ROOT` and run phase `99` (included in `run_all.sh`) to copy artifacts. The last backup path is stored in `results/tables/.last_drive_backup_path.txt`.

**Compiled tables:** `results/tables/table1_main_comparison.{md,csv}` and `metrics_summary.csv` (same as Table 1 CSV). See `docs/metrics_conventions.md`. After Qwen eval, run `DIAG_ONLY=replay` with `phase_12_diagnose.sh` on `qwen25_vl_zero_shot_test.jsonl`.

**Strict alignment (before submission):** `VERIFY_STRICT=1 bash scripts/shell/phase_13_verify_eval.sh` — fails if any stored `n` ≠ current pair count.

See also `docs/colab_pro_t4.md`, `docs/manual_qa_checklist.md`.
