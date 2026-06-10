# Table 1 — Main Model Comparison (test split)

| Model | CER ↓ | Median CER ↓ | Micro CER ↓ | WER ↓ | DER ↓ |
|-------|------:|-------------:|------------:|------:|------:|
| PaddleOCR-VL-1.5 (zero-shot) | 543.3 | 100.0 | 270.6 | 840.9 | 200.9 |
| Qwen 2.5 VL-3B (zero-shot) | 253.5 | 114.3 | 158.1 | 329.5 | 119.6 |
| PaddleOCR-VL-1.5 (LoRA fine-tuned — main supervised) | 96.5 | **88.8** | **84.9** | 122.6 | **66.4** |
| PaddleOCR PP-OCRv4 (CRNN fine-tuned — comparison) | **91.5** | 96.3 | 93.9 | **103.4** | 91.8 |
