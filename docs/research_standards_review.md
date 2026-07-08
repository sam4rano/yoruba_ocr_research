# Research pipeline vs common practice (self-critical review)

This document contrasts the repository’s OCR research pipeline with typical **2024–2026** practice for **evaluation**, **multimodal VLM zero-shot inference**, and **classical OCR** training. It is meant for internal rigour: where we align with standards, where we deliberately simplify, and where a reviewer or reviewer-shaped benchmark might object.

---

## 1. Active OCR Stack

The benchmark contains the following active model families:
1. **Base PaddleOCR** (English pretrained recognition, optional supervised recognition fine-tune)
2. **PaddleOCR-VL-1.6** (zero-shot multimodal vision-language model)
3. **GLM-OCR** (zero-shot multimodal vision-language model)
4. **PaddleOCR-VL-1.6 SFT** (optional supervised fine-tuning on the training split)

Older PaddleOCR-VL-1.5 LoRA, Qwen, TrOCR, and ablation claims are no longer part of the active pipeline unless their scripts and metrics are regenerated. Current paper claims must be based on fresh rows in `results/tables/metrics.csv`.

---

## 2. Multimodal VLM Evaluation (PaddleOCR-VL-1.6 & GLM-OCR)

### What matches common practice

- **Data hygiene:** Zero-shot evaluation does not mutate `data/processed/`. The same line crops and NFC text are evaluated across all models, matching the expectation that all table rows refer to one benchmark definition.
- **Prompt consistency:** VL-1.6 and GLM-OCR use deterministic generation and fixed transcript prompts for line crops.
- **Deduplication & Splits:** All model rows are evaluated on the exact same frozen test split.

### Gaps vs "production" VLM benchmarks

- **Throughput:** Zero-shot VLM evaluation scripts run single-image loops. While simple and reliable, they do not yet leverage large batch inference, resume-by-sample JSONL checkpoints, or vLLM-style serving.
- **Quantization:** Standard model parameters are loaded in float16/bfloat16. High-resource environments may run in 8-bit/4-bit quantization, which is supported but optional here.

---

## 3. Classical PaddleOCR Recognition (`03`–`06`, `05`)

### Alignment

- **Eval YAML:** Eval `05_evaluate.py` uses the same YAML family as the checkpoint. The config fallback is `configs/paddleocr_yoruba_rec.yml` to prevent mixing incompatible weights and architectures.
- **Dictionary:** Forcing the Yorùbá character dict at decode time for both baseline and fine-tuned models is a deliberate **fair decoding** choice; it is not the same as “English-only out-of-the-box PaddleOCR,” and must be stated clearly in the paper.

### Standard CRNN / CTC expectations

- Training via upstream `tools/train.py` with a frozen architecture block and CTC loss is normal.
- **Reporting:** CER/WER/DER on a held-out **test** split is appropriate; train/val/test splits are strictly disjoint, and filenames are deduplicated at consolidation time.

---

## 4. Metrics (DER / CER / WER)

- **NFC** before character/word edit distance matches Unicode-normalisation guidance for African languages in many NLP pipelines.
- **DER** (combining marks in NFD) is a project-specific metric; it is defined once and used consistently. It is **not** a standard like CER on a shared benchmark (IAM, etc.), so external comparability is limited.

---

## 5. Documentation and reproducibility checklist

- [ ] Record `pip freeze`, Paddle / CUDA / `transformers` versions per paper run.
- [ ] Store `experiments/finetuned/config.yml` or exact CLI for `05` next to published PaddleOCR recognition numbers.
- [ ] Record SFT epoch-level validation CER/DER and best-checkpoint selection before citing the VL-1.6 fine-tuned row.

---

## 6. Epoch budget (PaddleOCR recognition)

- **PaddleOCR recognition** (`03` / `04`): Default **`epoch_num` is 40** in generated and checked-in YAMLs. Recognition CTC runs often **plateau well before 100**; validation accuracy from Paddle’s training logs or a held-out eval should be monitored. Override with `CONFIG_EPOCHS=30` when running `phase_03_config.sh`, or `--epochs` on `03_generate_config.py`.

---

## 7. Bottom line

The repo is **methodologically coherent** when all rows are regenerated from one frozen `data/processed` split: shared crops, NFC text, aligned decoding choices, deterministic VLM generation, and traceable JSONL/meta files. The main **honest limitations** are engine heterogeneity across baselines, project-specific DER, single-image VLM throughput, and the need to freeze one split definition before citation.
