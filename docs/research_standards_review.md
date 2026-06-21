# Research pipeline vs common practice (self-critical review)

This document contrasts the repository’s OCR research pipeline with typical **2024–2026** practice for **evaluation**, **multimodal VLM zero-shot inference**, and **classical OCR** training. It is meant for internal rigour: where we align with standards, where we deliberately simplify, and where a reviewer or reviewer-shaped benchmark might object.

---

## 1. Active 3-Model OCR Stack

The benchmark contains exactly 3 models:
1. **Base PaddleOCR** (PP-OCRv4 English pretrained + CRNN fine-tune ablations)
2. **PaddleOCR-VL-1.6** (zero-shot multimodal vision-language model)
3. **GLM-OCR** (zero-shot multimodal vision-language model)

All fine-tuning on VLMs (LoRA, PEFT) has been completely removed to avoid instability and focus on robust, out-of-the-box multimodal zero-shot capabilities alongside classical CRNN supervised fine-tuning.

---

## 2. Multimodal VLM Evaluation (PaddleOCR-VL-1.6 & GLM-OCR)

### What matches common practice

- **Data hygiene:** Zero-shot evaluation does not mutate `data/processed/`. The same line crops and NFC text are evaluated across all models, matching the expectation that all table rows refer to one benchmark definition.
- **Prompt consistency:** VL-1.6 and GLM-OCR use consistent zero-shot transcript prompt instructions to transcribe the line crops.
- **Deduplication & Splits:** We evaluate zero-shot models on the exact same test split as the classical fine-tuned PP-OCRv4 model.

### Gaps vs "production" VLM benchmarks

- **Throughput:** Zero-shot VLM evaluation scripts run single-image loops. While simple and reliable, it does not leverage large batch inference or vLLM engines.
- **Quantization:** Standard model parameters are loaded in float16/bfloat16. High-resource environments may run in 8-bit/4-bit quantization, which is supported but optional here.

---

## 3. Classical PaddleOCR PP-OCRv4 (`03`–`06`, `05`)

### Alignment

- **Eval YAML:** Eval `05_evaluate.py` uses the same YAML family as the checkpoint. The config fallback is `configs/paddleocr_yoruba_rec.yml` to prevent mixing PP-OCRv3/v4 weights with conflicting architectures.
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
- [ ] Store `experiments/finetuned/config.yml` or exact CLI for `05` next to published numbers.

---

## 6. Epoch budget (PP-OCR CRNN)

- **PP-OCRv4 recognition** (`03` / `04`): Default **`epoch_num` is 40** in generated and checked-in YAMLs. Recognition CTC runs often **plateau well before 100**; validation accuracy from Paddle’s training logs or a held-out eval should be monitored. Override with `CONFIG_EPOCHS=30` when running `phase_03_config.sh`, or `--epochs` on `03_generate_config.py`.

---

## 7. Bottom line

The repo is **methodologically coherent** for a controlled comparison on one dataset: shared crops, NFC text, aligned decoding choices for Paddle CRNN. The main **honest limitations** are: engine heterogeneity across baselines, and project-specific DER. Addressing those in text is preferable to implying parity with large shared OCR leaderboards.
