# Abstract

Optical Character Recognition (OCR) for African languages remains underdeveloped, particularly for tone languages such as Yorùbá, where diacritics are semantically contrastive and transcription errors can change lexical meaning. We present a reproducible Yorùbá line-crop OCR benchmark built from the *Yorùbá di Wúrà* graded reader series, with UTF-8 NFC-normalized labels and a closed 99-character dictionary.

The active benchmark evaluates Base PaddleOCR English-pretrained recognition, PaddleOCR-VL-1.6, GLM-OCR, and optional PaddleOCR-VL-1.6 supervised fine-tuning on one frozen `data/processed` split. We report Character Error Rate (CER), Word Error Rate (WER), and Diacritic Error Rate (DER), a corpus-level metric that isolates tone-mark and subdot fidelity.

All headline numbers in the paper are regenerated from `results/tables/metrics.csv` and per-sample JSONL logs. Older PaddleOCR-VL-1.5 LoRA, Qwen, TrOCR, and ablation pilot results are excluded unless rerun through the current pipeline.
