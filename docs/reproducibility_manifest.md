# Reproducibility Manifest

This manifest records the minimum information needed to reproduce a citable
paper run. Update it after the final metrics run and before exporting a
submission or artifact package.

## Repository state

| Field | Value |
| --- | --- |
| Manifest date | 2026-07-12 |
| Source repository commit before pipeline cleanup | `5476b181f8e2e5c96bef7ed30bf6ccd09bcb84b6` |
| Working tree state | Pipeline cleanup in progress; replace this row with the clean final commit before the experiment run. |
| Canonical local root | `/Users/mac/Desktop/yoruba_ocr_research` |
| Primary Colab notebook | `notebooks/colab_ocr.ipynb` |
| Primary Kaggle notebook | `notebooks/kaggle_ocr.ipynb` |
| Primary Lightning notebook | `notebooks/lightning_ocr.ipynb` |

## Dataset snapshot

The active dataset snapshot is the frozen `data/processed` split.

| Split | Label file | Examples |
| --- | --- | ---: |
| Train | `data/processed/labels/train.txt` | 2,367 |
| Validation | `data/processed/labels/val.txt` | 252 |
| Test | `data/processed/labels/test.txt` | 326 |
| Total |  | 2,945 |

Additional dataset facts:

| Field | Value |
| --- | --- |
| Image files under `data/processed/images` | 2,945 |
| Character dictionary | `data/processed/dictionary/yoruba_char_dict.txt` |
| Character dictionary size | 99 |
| HF export manifest | `data/hf_export/manifest.json` |

## Dataset checksums

These checksums should be regenerated whenever labels, images, or dictionary
files change.

| Artifact | SHA-256 |
| --- | --- |
| `data/processed/labels/train.txt` | `7b55d485739fd223b0419d2a8c7af2d560de4ba9ee3f2a8fe32a6fc2aa6c1313` |
| `data/processed/labels/val.txt` | `21117af3a75bbcc2d18d3e040e8aabb0b0be76dda2977f36ab27c33bb5d63461` |
| `data/processed/labels/test.txt` | `0f3574ec5d5e123c60e813c89b22bc691037ca4f0877c9f8e9ccb6331c017a10` |
| `data/processed/dictionary/yoruba_char_dict.txt` | `0e052868728f7e83530c7c409d24a350dc135c7dd8ef773a8272184595c7dec7` |
| Sorted image-file path list under `data/processed/images` | `23b76ed6bcf6807928b5e62f522a0663883ec66d577b395f9789898bdb7d247f` |
| `data/hf_export/manifest.json` | `80203f5a2c32e8d729fde693970b9f93eb1504409d226edf314702f237b543e8` |

Recommended regeneration commands:

```bash
shasum -a 256 data/processed/labels/*.txt
shasum -a 256 data/processed/dictionary/yoruba_char_dict.txt
find data/processed/images -type f | sort | shasum -a 256
```

For a stronger archive checksum, create a deterministic tarball outside the
repository and checksum that file.

## Active model rows

| Model key | Source | Role | Revision policy |
| --- | --- | --- | --- |
| `paddleocr_en_pretrained` | English PP-OCRv3 recognition checkpoint loaded by `scripts/evaluate_paddleocr_en_pretrained.py` | Classical OCR control | Record downloaded checkpoint URL/hash in `results/tables/meta/*.json` for the final run. |
| `paddleocrvl16_zero_shot` | `PaddlePaddle/PaddleOCR-VL-1.6` | Zero-shot VLM OCR | Record Hugging Face resolved commit/revision in the per-run meta file. |
| `glm_ocr_zero_shot` | `zai-org/GLM-OCR` | Zero-shot VLM OCR | Record Hugging Face resolved commit/revision in the per-run meta file. |
| `paddleocrvl16_sft` | Fine-tuned from `PaddlePaddle/PaddleOCR-VL-1.6` | Supervised VLM adaptation | Record base model revision, checkpoint directory, epoch count, and training arguments. |

Model-row definitions are maintained in `docs/model_matrix.md`.

## Runtime targets

| Environment | Purpose | Notes |
| --- | --- | --- |
| Colab Pro T4 | Main low-cost GPU path | Use `docs/colab_pro_t4.md` and `notebooks/colab_ocr.ipynb`. |
| Kaggle GPU | Alternative GPU path | Use `docs/kaggle_gpu.md` and `notebooks/kaggle_ocr.ipynb`. |
| Lightning AI GPU | Persistent GPU path | Use `docs/lightning_ai.md` and `notebooks/lightning_ocr.ipynb`. |
| Local macOS workspace | Code editing, syntax checks, documentation | Local Python versions observed: system `Python 3.14.4`, project venv `Python 3.11.15`. |

Final paper artifacts should also record:

- GPU type and VRAM.
- CUDA, PyTorch, PaddlePaddle, `transformers`, and `accelerate` versions.
- Whether 4-bit quantization was enabled.
- Generation caps such as `PADDLEOCRVL16_MAX_NEW_TOKENS` and `GLM_MAX_NEW_TOKENS`.
- SFT arguments such as epochs, learning rate, gradient accumulation, max pixels,
  and checkpoint resume state.

## Citable output checklist

Before citing numbers in the paper, verify that these files exist and are from
the same clean repository commit:

- `results/tables/metrics.csv` contains full rows for each cited model.
- `results/tables/*_test.jsonl` contains per-sample predictions.
- `results/tables/meta/*.json` contains provenance for each model run.
- `results/tables/table1_main_comparison.csv` and `.md` are regenerated.
- `results/tables/figures/` contains plots regenerated without
  `--allow-placeholder`.
- `paper/sections/*.md` and the regenerated IJCAI manuscript cite the same
  final table values. The stale pre-regeneration manuscript is intentionally
  absent from the clean workspace.
