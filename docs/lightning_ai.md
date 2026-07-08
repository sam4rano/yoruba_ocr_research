# Lightning AI Studio Guide

Use [notebooks/lightning_ocr.ipynb](../notebooks/lightning_ocr.ipynb) when you
want to run the OCR experiment in a Lightning AI Studio or notebook.

Lightning is usually a better fit than short notebook sessions for
PaddleOCR-VL-1.6 SFT because the Studio filesystem is persistent and GPU jobs
can be resumed more comfortably.

## 1. Create the Studio

1. Create or open a Lightning AI Studio.
2. Select a GPU machine.
3. Open the notebook interface or terminal.
4. Put the repository at one of:

```text
/teamspace/studios/this_studio/yoruba_ocr_research
<current working directory>/yoruba_ocr_research
```

If the repo is not present, set `GITHUB_REPO` before the first notebook cell:

```python
import os
os.environ["GITHUB_REPO"] = "https://github.com/<user>/<repo>.git"
```

You can also set `PROJECT_ROOT` explicitly:

```python
import os
os.environ["PROJECT_ROOT"] = "/teamspace/studios/this_studio/yoruba_ocr_research"
```

## 2. Upload or link data

The notebook first checks:

```text
PROJECT_ROOT/data/processed
```

If that folder is missing, upload the processed dataset somewhere in the Studio
and set `DATA_ROOT`. `DATA_ROOT` may point either to a parent folder containing
`processed/` or to the `processed/` folder itself:

```python
import os
os.environ["DATA_ROOT"] = "/teamspace/studios/this_studio/data"
```

Required processed layout:

```text
images/train/
images/val/
images/test/
labels/train.txt
labels/val.txt
labels/test.txt
dictionary/yoruba_char_dict.txt
```

## 3. Install dependencies

The notebook installs into the active Studio Python environment:

1. Upgrade pip.
2. Install Paddle GPU.
3. Install project requirements.
4. Clone/install PaddleOCR.
5. Reassert the Hugging Face/VLM stack last.

Default Paddle spec:

```python
paddlepaddle-gpu>=2.6,<2.7
```

If the Studio CUDA image requires a specific Paddle wheel, set:

```python
import os
os.environ["PADDLE_PIP_SPEC"] = "paddlepaddle-gpu==<version-for-studio-cuda>"
```

Restart the kernel after major package upgrades, especially after changing
Paddle, Torch, `transformers`, or `accelerate`.

## 4. Run toggles

The notebook exposes:

```python
RUN_RESET = True
RUN_PADDLE_BASELINE = True
RUN_VLM_ZERO_SHOT = True
RUN_PADDLEOCRVL16_SFT = False
RUN_ANALYSIS = True
```

Suggested order:

1. First run with `RUN_PADDLEOCRVL16_SFT = False`.
2. Confirm `metrics.csv` has rows for:
   - `paddleocr_en_pretrained`
   - `paddleocrvl16_zero_shot`
   - `glm_ocr_zero_shot`
3. Turn on `RUN_PADDLEOCRVL16_SFT = True`.
4. Rerun SFT phases and analysis.

For resume after an interrupted SFT run:

```python
import os
os.environ["PADDLEOCRVL16_SFT_RESUME"] = "1"
```

For a quick smoke run:

```python
import os
os.environ["PADDLEOCRVL16_MAX_SAMPLES"] = "5"
os.environ["GLM_MAX_SAMPLES"] = "5"
```

Remove those caps before final paper metrics.

## 5. Outputs

Normal outputs remain in the repo:

```text
results/tables/
experiments/
research_approach.md
```

The notebook also packages:

```text
platform_outputs/lightning_ai/
platform_outputs/lightning_ai.zip
```

After a citable run, keep:

```text
results/tables/metrics.csv
results/tables/*_test.jsonl
results/tables/meta/*.json
results/tables/table1_main_comparison.csv
results/tables/table1_main_comparison.md
results/tables/figures/*.png
research_approach.md
```

If SFT was run, preserve:

```text
experiments/paddleocrvl16_sft/
```

## 6. Failure checklist

- **No GPU:** verify the Studio machine type and run `nvidia-smi`.
- **Data path wrong:** set `DATA_ROOT` and rerun the data cell.
- **Paddle import fails:** adjust `PADDLE_PIP_SPEC`, restart, reinstall.
- **Hugging Face model fails:** confirm `transformers>=5`, Internet access, and
  enough disk in the Studio.
- **SFT interrupted:** set `PADDLEOCRVL16_SFT_RESUME=1` and rerun the SFT cell.
- **Final table missing rows:** check `results/tables/metrics.csv` and per-model
  JSONL logs before running analysis and compile.

