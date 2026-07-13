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

### Runtime and efficiency policy

- **Throughput:** Zero-shot scripts batch compatible processor calls and automatically retry a failed/OOM batch one image at a time. Batch size is recorded in provenance.
- **Interruption safety:** Every sample is appended to a fingerprinted `.jsonl.partial` checkpoint. A rerun resumes only when model, prompt, precision, split, generation cap, and ordered sample identities match.
- **Failure integrity:** Failed samples stop metrics publication by default. `--allow-failures` is an explicit lower-bound mode that records each error and counts its prediction as empty.
- **Quantization:** Standard model parameters are loaded in float16/bfloat16. High-resource environments may run in 8-bit/4-bit quantization, which is supported but optional here.
- **Remaining scaling limit:** The scripts use direct Hugging Face generation rather than a dedicated serving engine such as vLLM. This is appropriate for the 326-line test split but not a high-throughput production service.

---

## 3. Classical PaddleOCR Recognition (`03`–`06`, `05`)

### Alignment

- **Eval YAML:** Eval `evaluate_paddleocr_recognition.py` uses the same YAML family as the checkpoint. The config fallback is `configs/paddleocr_yoruba_rec.yml` to prevent mixing incompatible weights and architectures.
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
- [ ] Preserve the generated PaddleOCR recognition config and per-run metadata next to published recognition numbers.
- [ ] Confirm the generated SFT `training_log.jsonl` and `best_validation.json` are included before citing the VL-1.6 fine-tuned row.

---

## 6. Epoch budget (PaddleOCR recognition)

- **PaddleOCR recognition** (`03` / `04`): Default **`epoch_num` is 40** in generated and checked-in YAMLs. Recognition CTC runs often **plateau well before 100**; validation accuracy from Paddle’s training logs or a held-out eval should be monitored. Override with `CONFIG_EPOCHS=30` when running `phase_03_config.sh`, or `--epochs` on `generate_paddleocr_config.py`.

---

## 7. Bottom line

The repo is **methodologically coherent** when all rows are regenerated from one frozen `data/processed` split: shared crops, NFC text, aligned decoding choices, deterministic resumable VLM generation, and traceable JSONL/meta files. The main **honest limitations** are engine heterogeneity across baselines, project-specific DER, direct-Hugging-Face rather than serving-engine throughput, and the need to freeze one split definition before citation.
