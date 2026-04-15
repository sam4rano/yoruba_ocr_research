# Table 1 — Main Model Comparison (test split)

| Model | CER ↓ | WER ↓ | DER ↓ |
|-------|------:|------:|------:|
| PaddleOCR PP-OCRv4 (EN pretrained) | 102.0 | **100.8** | 89.2 |
| Tesseract (eng) | 150.7 | 167.5 | **77.0** |
| Tesseract (yor) | 162.2 | 180.0 | 78.3 |
| Tesseract (eng+yor) | 152.5 | 175.3 | 77.6 |
| PaddleOCR PP-OCRv4 (fine-tuned) | **96.1** | 101.2 | 81.3 |
