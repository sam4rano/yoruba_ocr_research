# Research pipeline vs common practice (self-critical review)

This document contrasts the repository’s OCR research pipeline with typical **2024–2026** practice for **evaluation**, **supervised fine-tuning (SFT)** of multimodal LMs, and **classical OCR** training. It is meant for internal rigour: where we align with standards, where we deliberately simplify, and where a reviewer or reviewer-shaped benchmark might object.

---

## 1. PaddleOCR-VL-1.5 LoRA (`14` → `16` → `15`)

### What matches common practice

- **Data hygiene:** Export **14** does not mutate `data/processed/`; JSONL holds the same line crops and NFC text as the CRNN track. That matches the expectation that all table rows refer to one benchmark definition.
- **LoRA on linear layers** (`q_proj`, `v_proj`, …): Standard PEFT usage for instruction-tuned VLMs.
- **Assistant-only loss:** Script **16** masks non-assistant positions with `-100`, which is the usual PyTorch / Transformers convention for causal LM SFT (see TRL `SFTTrainer` patterns: train only on completion tokens).
- **Prefix sanity check:** Tokenizing the user turn with `add_generation_prompt=True` and requiring equality with the prefix of the full sequence catches many template inconsistencies. This is the same *idea* as HF’s `return_assistant_tokens_mask`, without relying on processor support (multimodal `assistant_masks` have had known bugs; see upstream issues on multimodal `apply_chat_template`).

### Gaps vs “production” SFT

| Topic | This repo | Typical stronger setup |
|--------|-----------|---------------------------|
| Trainer | Manual loop, one image per forward | `SFTTrainer`, DeepSpeed/ZeRO, or LLaMA-Factory |
| Validation | None during **16** | Val loss / CER on `val` split, early stopping |
| Learning rate | Fixed AdamW | Schedulers, warmup, layer-wise LR |
| Batching | Micro-batch = 1 image | Packed batches or larger micro-batches if memory allows |
| Reproducibility | Seed set | Often `deterministic` flags, logged hyperparams JSON |
| Objective | Causal LM on transcript | Some OCR work uses token-level CE only on answer; we are close, but not using TRL’s collators |

### Residual risk

- If the chat template ever encodes the **full** conversation differently from **user + `add_generation_prompt=True`**, prefix equality fails and we **skip** the step — better than wrong loss, but you should monitor skip rate in logs.
- **Gradient accumulation** scales effective batch size for the optimizer, not VRAM per step; wall time per image is unchanged.

---

## 2. Classical PaddleOCR PP-OCRv3/v4 (`03`–`06`, `05`)

### Alignment

- Eval **05** should use the **same YAML family** as the checkpoint (v3 vs v4). The default fallback was corrected to `configs/paddleocr_yoruba_rec.yml` when `experiments/finetuned/config.yml` is absent, so baseline and fine-tuned runs do not mix PP-OCRv3 weights with a v4-oriented config file.
- **Dictionary:** Forcing the Yorùbá character dict at decode time for both baseline and fine-tuned models is a deliberate **fair decoding** choice (documented in `paper/sections/experiments.md`); it is not the same as “English-only out-of-the-box PaddleOCR,” and must be stated clearly in the paper.

### Standard CRNN / CTC expectations

- Training via upstream `tools/train.py` with a frozen architecture block and CTC loss is normal.
- **Reporting:** CER/WER/DER on a held-out **test** split is appropriate; ensure no duplicate lines between train and test at consolidation time (**01** deduplicates by filename).

---

## 3. Baselines: Tesseract, Qwen, zero-shot VL-1.5

### Fairness of inputs

- **07**, **09**, **15** all call `evaluate_utils.load_test_pairs` for the same `--data-dir` and `--split` → same crops and ground truth. **Tesseract** uses PSM 7 (single line), which matches the crop type.
- **Differences that are intentional:** engine capabilities (no VLM reasoning parity with Tesseract), prompt wording (Qwen vs VL-1.5 vs fixed OCR prompt in `paddle_vl_shared.py`), and runtime (zero-shot VLMs are slow).

### Multimodal standards note

- Qwen and PaddleOCR-VL eval are **single-image loops**; `batch_size` in scripts is mostly **logging cadence**, not GPU batching. That is honest but not throughput-optimal.

---

## 4. Metrics (DER / CER / WER)

- **NFC** before character/word edit distance matches Unicode-normalisation guidance for African languages in many NLP pipelines.
- **DER** (combining marks in NFD) is a project-specific metric; it is defensible if defined once and used consistently. It is **not** a standard like CER on a shared benchmark (IAM, etc.), so external comparability is limited.

---

## 5. Documentation and reproducibility checklist

- [ ] Record `pip freeze`, Paddle / CUDA / `transformers` versions per paper run.
- [ ] Store `experiments/finetuned/config.yml` or exact CLI for **05** next to published numbers.
- [ ] For VL-1.5: archive **14**’s `manifest.json` and the **16** command line (epochs, `--max-samples`, `--gradient-accumulation-steps`).
- [ ] Log fraction of **skipped** SFT steps (prefix mismatch) if training long runs.

---

## 6. Epoch budget (PP-OCR CRNN vs VL LoRA)

**Primary fine-tuning in this project is PaddleOCR-VL-1.5 LoRA** (`14` → `16` → `15` with adapter), not PP-OCR CRNN. The bullets below about `epoch_num` apply only to the **classical comparison** pipeline (`03` / `04`); VL LoRA defaults remain **`--epochs 1`** in `16` unless you raise them deliberately.

- **PP-OCRv3/v4 recognition** (`03` / `04`): Default **`epoch_num` is 40** in generated and checked-in YAMLs. Recognition CTC runs often **plateau well before 100**; use validation accuracy from Paddle’s training logs or a held-out eval to stop early. Override with `CONFIG_EPOCHS=30` when running `phase_03_config.sh`, or `--epochs` on `03_generate_config.py`.
- **PaddleOCR-VL-1.5 LoRA** (`16`): Default is **`--epochs 1`**; each epoch is already a full pass over `train.jsonl` with a heavy VLM. Raising VL epochs increases wall time linearly; prefer **more epochs only after** you see val/test CER move.

**Efficiency elsewhere:** `VL15_GRAD_ACCUM`, `--max-samples` (debug), `--quantize-4bit` on **15**, and batch size in **03** (VRAM permitting) address speed without changing epoch count alone.

## 7. Bottom line

The repo is **methodologically coherent** for a controlled comparison on one dataset: shared crops, NFC text, aligned decoding choices for Paddle CRNN, and SFT masking aligned with common HF practice for **16**. The main **honest limitations** are: minimal VL training loop (no val loop, no TRL integration), engine heterogeneity across baselines, and project-specific DER. Addressing those in text is preferable to implying parity with large shared OCR leaderboards.
