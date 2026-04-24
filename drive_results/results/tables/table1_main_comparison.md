# Table 1 — Main Model Comparison (test split)

| Model | CER ↓ | WER ↓ | DER ↓ |
|-------|------:|------:|------:|
| PaddleOCR PP-OCRv4 (EN pretrained) | 174.5 | **100.0** | 84.0 |
| Tesseract (eng) | 120.3 | 153.5 | 98.5 |
| Tesseract (yor) | 124.4 | 163.7 | 87.7 |
| Tesseract (eng+yor) | 122.6 | 160.0 | 93.9 |
| PaddleOCR-VL-1.5 (zero-shot) | 543.3 | 840.9 | 200.9 |
| Qwen 2.5 VL (zero-shot) | 253.5 | 329.5 | 119.6 |
| PaddleOCR-VL-1.5 (LoRA fine-tuned — main supervised) | **96.5** | 122.6 | **66.4** |
