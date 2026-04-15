# Yorùbá OCR — Colab Pro (T4) end-to-end

This guide assumes you use **Google Colab Pro** with a **T4 GPU**, and your project lives on Drive under a folder such as:

`MyDrive/yoruba_ocr_research/`

with data already present as:

- `data/raw/` — export batches  
- `data/processed/` — consolidated PaddleOCR-format dataset (line images + labels)  
- `data/splits/` — optional split files if your workflow uses them  

The pipeline in this repo expects **`data/processed/`** as the working dataset root for training and evaluation. If `processed` is complete and consistent with `01_consolidate_data.py` output, you can skip consolidation or re-run it only if you need a clean merge from `raw`.

---

## 1. New notebook and runtime

1. Create a new Colab notebook.
2. **Runtime → Change runtime type →** GPU (**T4** when available), **High-RAM** if offered.
3. Run once:

```python
!nvidia-smi
```

Note the **CUDA driver** version; you will match **Paddle GPU wheel** to Colab’s CUDA (see step 4).

---

## 2. Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Set a single variable for your project root on Drive (adjust the path if yours differs):

```python
import os
PROJECT_ROOT = "/content/drive/MyDrive/yoruba_ocr_research"
os.chdir(PROJECT_ROOT)
!pwd
```

Confirm:

```python
!ls -la data/processed data/raw 2>/dev/null || echo "Check paths"
```

---

## 3. Python environment

Colab ships one Python per runtime. Install dependencies **into the current interpreter** (no need for a separate venv unless you prefer isolation).

**Install Paddle GPU build** using the command from the official Paddle install page for your Colab CUDA version (do not guess the wheel). Typical pattern:

```bash
# Example only — replace with the exact line from:
# https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html
# python -m pip install paddlepaddle-gpu==... -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

Then install project requirements from the repo root:

```bash
cd /content/drive/MyDrive/yoruba_ocr_research
python -m pip install -U pip
python -m pip install -r requirements.txt
```

If `requirements.txt` pulls a CPU `paddlepaddle` wheel and conflicts with your GPU install, install **`paddlepaddle-gpu`** first per Paddle docs, then:

```bash
python -m pip install -r requirements.txt --no-deps
python -m pip install $(grep -v '^#' requirements.txt | grep -v paddle)
```

Verify GPU is visible to Paddle:

```python
import paddle
paddle.device.cuda.device_count()  # expect >= 1 on T4
```

---

## 4. Clone PaddleOCR (required for training and `05_evaluate.py`)

From `PROJECT_ROOT`:

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git PaddleOCR
python -m pip install -r PaddleOCR/requirements.txt
```

---

## 5. Data layout check

Expected minimum for training/eval:

- `data/processed/images/{train,val,test}/` — PNG crops  
- `data/processed/labels/{train,val,test}.txt` — Paddle tab-separated label files  
- `data/processed/dictionary/yoruba_char_dict.txt` — character dictionary  

If you only uploaded `raw` and need a fresh merge:

```bash
python scripts/01_consolidate_data.py --help
# Run with your repo’s documented arguments after inspecting the script.
```

If `processed` is already final, skip `01`.

---

## 6. End-to-end script order (GPU)

Run from `PROJECT_ROOT` with `python` (or `python3` if that is what Colab uses — be consistent).

| Step | Script | Role |
|------|--------|------|
| 1 | `scripts/02_analyze_dataset.py` | EDA; updates local JSON under `results/tables/` |
| 2 | `scripts/03_generate_config.py` | Downloads pretrained weights (if configured) + writes YAML |
| 3 | `scripts/04_train_paddleocr.py` | Fine-tuning via `PaddleOCR/tools/train.py` |

**Training on T4 (single GPU):**

```bash
python scripts/04_train_paddleocr.py \
  --config configs/paddleocr_yoruba_rec.yml \
  --paddle-dir PaddleOCR \
  --gpus 0 \
  --log-file results/tables/train_run.json
```

Do **not** pass `--cpu` on Colab when you want GPU training. Use `--epochs`, `--batch-size`, `--lr` only if you intentionally override the YAML.

| Step | Script | Role |
|------|--------|------|
| 4 | `scripts/05_evaluate.py` | Paddle checkpoints: CER / WER / DER → `results/tables/metrics.csv` + JSONL |
| 5 | `scripts/06_baseline_pretrained.py` | English pretrained baseline (delegates to `05`) |
| 6 | `scripts/07_baseline_tesseract.py` | Tesseract baselines (needs system `tesseract`) |
| 7 | `scripts/09_baseline_qwen.py` | Qwen 2.5 VL zero-shot (GPU; large download) |
| 8 | `scripts/10_ablation_study.py` | Ablations (trains multiple runs; long) |
| 9 | `scripts/11_compile_results.py` | Paper tables from `metrics.csv` |

**Evaluate fine-tuned model** (point `--model-dir` at the directory containing `best_accuracy.pdparams` or the checkpoint prefix your run produced, often under `experiments/finetuned/`):

```bash
python scripts/05_evaluate.py \
  --model-dir experiments/finetuned \
  --data-dir data/processed \
  --split test \
  --model-name finetuned_paddleocr_v1 \
  --use-gpu \
  --paddle-dir PaddleOCR
```

**Tesseract on Colab:**

```bash
apt-get update -qq && apt-get install -y -qq tesseract-ocr tesseract-ocr-yor
python scripts/07_baseline_tesseract.py --data-dir data/processed --split test --langs eng yor "eng+yor"
```

**Qwen on T4:** use `--quantize` if you hit VRAM limits; set `HF_TOKEN` if the model is gated. Prefer a capped run first:

```bash
export HF_TOKEN="..."   # if required
python scripts/09_baseline_qwen.py --split test --max-samples 50 --quantize --batch-size 10
```

**Compile tables:**

```bash
python scripts/11_compile_results.py
```

---

## 7. Where results land (repo)

All metrics that belong in the paper should trace to:

- `results/tables/metrics.csv` — master table (append-only across runs)  
- `results/tables/*_test.jsonl` / `*_val.jsonl` — per-sample logs  
- `results/tables/train_run.json` — last training invocation metadata (`04_train_paddleocr.py`)  
- `results/tables/table1_main_comparison.md` (and related) — after `11_compile_results.py`  
- `experiments/` — checkpoints and ablation configs  

---

## 8. Copy results to Google Drive (persistent backup)

Colab ephemeral storage is wiped when the runtime ends. After each major stage, mirror outputs to a **Drive** folder (same project or a dedicated `results_archive/`):

```python
import shutil
from pathlib import Path

project = Path("/content/drive/MyDrive/yoruba_ocr_research")
backup = Path("/content/drive/MyDrive/yoruba_ocr_results_backup")  # create once in Drive UI
backup.mkdir(parents=True, exist_ok=True)

for name in ["results", "experiments"]:
    src = project / name
    if src.exists():
        dest = backup / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
print("Backed up results/ and experiments/ to Drive.")
```

Optional: zip for a single file:

```bash
cd /content/drive/MyDrive
zip -r yoruba_ocr_results_$(date +%Y%m%d).zip yoruba_ocr_research/results yoruba_ocr_research/experiments
```

---

## 9. Save back to the Git repository (optional)

Drive holds files; **git history** lives where you push (GitHub/GitLab).

1. On your **local machine** (or a machine with git + your remote):

   - Pull the repo, copy `results/` and selected `experiments/` checkpoints from Drive, **or**  
   - Use `git` inside Colab if you authenticate (PAT or SSH) — acceptable for small `results/` only; large binaries should stay in Drive or Git LFS.

2. Commit only what your project policy allows:

   - `results/tables/*.csv`, `*.json`, `*.md`, JSONL (if size is reasonable)  
   - Avoid committing multi‑GB checkpoints unless you use **Git LFS** or release them separately (Drive / Hugging Face Hub).

3. Push to `main` or a branch named e.g. `results/colab-t4-YYYYMMDD`.

---

## 10. Practical Colab constraints

- **Session timeout:** long training may need Colab Pro+ or manual keep-alive; save to Drive often.  
- **Disk:** clone PaddleOCR + HF models fills space; monitor `!df -h`.  
- **Reproducibility:** record `pip freeze > results/tables/pip_freeze_colab.txt` once per successful run.  
- **Secrets:** never commit `HF_TOKEN`; use Colab secrets or env vars.

---

## 11. Phased shell scripts (optional)

The repo includes `scripts/shell/` with one script per phase plus `run_all.sh`. Set `DRIVE_BACKUP_ROOT` and include phase `99` to copy `results/` (and optionally `experiments/`) to Drive. See `scripts/shell/README.md`.

## 12. Minimal “smoke test” before a full train

```bash
python scripts/05_evaluate.py \
  --model-dir experiments/baseline/pretrained/en_PP-OCRv4_rec_train \
  --data-dir data/processed \
  --split test \
  --model-name baseline_english_pretrained \
  --use-gpu \
  --paddle-dir PaddleOCR
```

If this completes and appends a row to `metrics.csv`, your Colab env, data paths, and PaddleOCR clone are aligned for full training.
