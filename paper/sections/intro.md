# Introduction

The digitization of African-language texts remains a bottleneck for language technology. Yorùbá is especially challenging because tone marks and subdots are not decorative: they distinguish lexical meaning. A recognizer that drops the grave in *ògùn* or the subdot in *ọmọ* may preserve the base Latin letters while corrupting the word.

Standard OCR metrics such as Character Error Rate (CER) and Word Error Rate (WER) are necessary but incomplete for this setting. They do not isolate the error pattern most damaging to Yorùbá archival transcription: systematic failure on combining tone marks and subdots. This work therefore evaluates both general transcription quality and diacritic fidelity.

We make three contributions:

1. **A curated Yorùbá OCR dataset** of 2,945 annotated line crops from the *Yorùbá di Wúrà* graded reader series, stored as PaddleOCR-format train/validation/test labels under `data/processed`.
2. **A reproducible active benchmark** covering Base PaddleOCR English-pretrained recognition, PaddleOCR-VL-1.6 zero-shot, GLM-OCR zero-shot, and optional PaddleOCR-VL-1.6 supervised fine-tuning.
3. **Diacritic Error Rate (DER)**, a corpus-level metric over Unicode combining marks that directly measures tone-mark and subdot recovery.

The current paper build treats `results/tables/metrics.csv` and sibling JSONL/meta files as the source of truth. Stale pilot results from removed or renamed systems are not cited.
