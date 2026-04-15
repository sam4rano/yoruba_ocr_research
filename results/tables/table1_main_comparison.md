# Table 1 — Main Model Comparison (test split)

| Model | CER ↓ | WER ↓ | DER ↓ |
|-------|------:|------:|------:|
| PaddleOCR PP-OCRv4 (EN pretrained) | 100.0 | **100.0** | 89.8 |
| Tesseract (eng) | 100.0 | 100.0 | **77.0** |
| Tesseract (yor) | 100.0 | 100.0 | 78.3 |
| Tesseract (eng+yor) | 100.0 | 100.0 | 77.6 |
| PaddleOCR PP-OCRv4 (fine-tuned) | **96.1** | 100.0 | 81.3 |
