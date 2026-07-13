# Methodology

## Dataset

The benchmark uses line-level crops from the *Yorùbá di Wúrà* graded reader series. Raw annotation exports are consolidated by `scripts/consolidate_data.py`, which normalizes labels to NFC, applies hygiene filters, writes PaddleOCR-format labels, and records a consolidation report. The current frozen processed split contains 2,945 unique line crops: 2,367 train, 252 validation, and 326 test. Any future resplit must trigger full reevaluation before the paper is regenerated.

The character dictionary contains 99 non-space characters observed after normalization. PaddleOCR uses `use_space_char=True`, so space is handled separately from the dictionary file.

## Data Quality

The read-only audit `scripts/audit_data_quality.py` checks label lengths, image dimensions, Unicode blocks, non-Yorùbá codepoints, duplicate image hashes, and cross-split duplicate hashes. Tall crops are reported for manual QA because they may represent multi-line noise; they are not silently removed from an already frozen split.

## Systems

The active benchmark contains:

- **Base PaddleOCR English-pretrained recognition** evaluated with the project Yorùbá dictionary.
- **PaddleOCR-VL-1.6 zero-shot** with deterministic generation and a fixed Yorùbá line-transcription prompt.
- **GLM-OCR zero-shot** with deterministic generation and a fixed OCR prompt.
- **PaddleOCR-VL-1.6 SFT**, optional supervised fine-tuning on the training split using assistant-only causal language-model loss.
- **Optional PaddleOCR recognition fine-tune**, launched through `scripts/train_paddleocr_recognition.py`, when a classical supervised comparison is needed.

Removed pilot systems such as PaddleOCR-VL-1.5 LoRA, Qwen, and TrOCR are not part of the active comparison unless rerun and logged through the current result schema.

## Evaluation

All systems are evaluated on the same frozen split. Scripts write aggregate rows to `results/tables/metrics.csv`, per-sample predictions to JSONL, and provenance to `results/tables/meta/*.json`.

CER and WER are computed on NFC-normalized text. DER is computed over NFD combining marks and reported as a corpus-level micro-average over samples with at least one ground-truth diacritic. Lines with no ground-truth diacritics are excluded from DER and summarized separately with `der_insertion_rate`.

Before publication, the pipeline must pass:

- `scripts/verify_eval_alignment.py` so reported `n` matches current labels.
- `scripts/diagnose_experiment.py checkpoints` so stale or phantom checkpoint rows are excluded.
- `scripts/generate_plots.py` without placeholder data.
