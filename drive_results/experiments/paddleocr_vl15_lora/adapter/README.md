---
language: 
- yo
- en
license: apache-2.0
base_model: PaddlePaddle/PaddleOCR-VL-1.5
tags:
- ocr
- yoruba
- multimodal
- paddleocr
---

# Yorùbá OCR LoRA Adapter (PaddleOCR-VL-1.5)

This is a LoRA adapter for the [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) model, fine-tuned specifically for Yorùbá line crops as part of ongoing OCR research.

## Training Details
- **Dataset**: ~2,300 unique Yorùbá text line images.
- **Epochs**: 5
- **Architecture**: LoRA (r=16) on Qwen2-VL visual/text projection layers.
- **Normalization**: Unicode NFC normalization with Yorùbá combining diacritics.

## Research Results (Table 1 Comparison)
| display_name                       |   cer_pct |   wer_pct |   der_pct |
|:-----------------------------------|----------:|----------:|----------:|
| PaddleOCR PP-OCRv4 (EN pretrained) |     174.5 |     100   |      84   |
| Tesseract (eng)                    |     120.3 |     153.5 |      98.5 |
| Tesseract (yor)                    |     124.4 |     163.7 |      87.7 |
| Tesseract (eng+yor)                |     122.6 |     160   |      93.9 |
| PaddleOCR-VL-1.5 (zero-shot)       |     543.3 |     840.9 |     200.9 |
| Qwen 2.5 VL (zero-shot)            |     253.5 |     329.5 |     119.6 |

## Usage
To use this adapter, load the base model using `transformers` and apply the adapter using `peft`.
