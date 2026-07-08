# Kaggle GPU Notebook Guide

Use [notebooks/kaggle_ocr.ipynb](../notebooks/kaggle_ocr.ipynb) when you want
to run the OCR experiment on Kaggle GPU.

## 1. Create the Kaggle notebook

1. Create a new Kaggle notebook.
2. In notebook settings, enable:
   - **Accelerator:** GPU
   - **Internet:** On, unless all wheels and model caches are already attached
3. Attach datasets:
   - Repository dataset, suggested slug: `yoruba-ocr-research`
   - Processed OCR data dataset, suggested slug: `yoruba-ocr-data`

The notebook expects Kaggle's standard paths:

```text
/kaggle/input/yoruba-ocr-research/
/kaggle/input/yoruba-ocr-data/data/processed/
/kaggle/working/
```

If your dataset slugs differ, set these before running the first notebook cell:

```python
import os
os.environ["KAGGLE_REPO_INPUT"] = "your-repo-dataset-slug"
os.environ["KAGGLE_DATA_INPUT"] = "your-data-dataset-slug"
```

If you do not attach the repo as a dataset, set `GITHUB_REPO` and keep Internet
enabled:

```python
import os
os.environ["GITHUB_REPO"] = "https://github.com/<user>/<repo>.git"
```

## 2. Data layout

The data dataset should contain one of these layouts:

```text
data/processed/
processed/
```

The resolved `processed` folder must contain:

```text
images/train/
images/val/
images/test/
labels/train.txt
labels/val.txt
labels/test.txt
dictionary/yoruba_char_dict.txt
```

The Kaggle notebook symlinks that folder to:

```text
/kaggle/working/yoruba_ocr_research/data/processed
```

## 3. Dependency policy

The notebook installs into Kaggle's current Python interpreter. The install
order matters:

1. Install Paddle GPU.
2. Install `requirements.txt`.
3. Clone/install PaddleOCR requirements.
4. Reinstall the Hugging Face/VLM stack last.

The default Paddle install spec is:

```python
paddlepaddle-gpu>=2.6,<2.7
```

If Kaggle changes its CUDA image and Paddle import fails, set:

```python
import os
os.environ["PADDLE_PIP_SPEC"] = "paddlepaddle-gpu==<version-for-kaggle-cuda>"
```

Then restart the session and rerun setup. If `transformers.__version__` is less
than 5 after installation, restart the Kaggle session and rerun setup cells.

## 4. Run toggles

In the notebook's run-plan cell:

```python
RUN_RESET = True
RUN_PADDLE_BASELINE = True
RUN_VLM_ZERO_SHOT = True
RUN_PADDLEOCRVL16_SFT = False
RUN_ANALYSIS = True
```

Recommended first Kaggle run:

- Keep `RUN_PADDLEOCRVL16_SFT = False`.
- Run PaddleOCR EN baseline, PaddleOCR-VL-1.6 zero-shot, GLM-OCR zero-shot.
- Turn SFT on only after the zero-shot model downloads and inference work.

For low VRAM, try:

```python
import os
os.environ["PADDLEOCRVL16_QUANTIZE_4BIT"] = "1"
os.environ["GLM_QUANTIZE_4BIT"] = "1"
```

Do not mix 4-bit and non-4-bit rows in the same final comparison unless the
precision choice is recorded in the report.

## 5. Outputs

The notebook copies citable outputs to:

```text
/kaggle/working/yoruba_ocr_outputs/
/kaggle/working/yoruba_ocr_outputs.zip
```

Expected useful files after a full successful run:

```text
results/tables/metrics.csv
results/tables/*_test.jsonl
results/tables/meta/*.json
results/tables/table1_main_comparison.csv
results/tables/table1_main_comparison.md
results/tables/figures/*.png
research_approach.md
```

Kaggle will expose `/kaggle/working/yoruba_ocr_outputs.zip` as an output
artifact after the notebook finishes.

## 6. Failure checklist

- **Repo not found:** attach the repo dataset or set `GITHUB_REPO`.
- **Data not found:** check the data dataset slug and make sure it contains
  `data/processed` or `processed`.
- **Paddle import fails:** set `PADDLE_PIP_SPEC` for Kaggle's CUDA version,
  restart, and rerun setup.
- **PaddleOCR-VL or GLM load fails:** confirm Internet is on, `transformers>=5`,
  and enough disk remains under `/kaggle/working`.
- **Out of memory:** enable 4-bit quantization for zero-shot runs or reduce
  sample count with `PADDLEOCRVL16_MAX_SAMPLES` / `GLM_MAX_SAMPLES`.

