# Diacritic Error Rate

Standard CER weights all character errors equally: a substitution of *ò* → *o* (tone loss) incurs the same penalty as *ò* → *z* (random corruption). For Yorùbá OCR, the former is the dominant and most consequential failure mode—it silently changes word meaning—while the latter is easily detectable by downstream systems.

We define DER to isolate diacritic-specific errors. Given reference y and hypothesis ŷ, we first decompose both strings into their NFD (canonical decomposition) representations, then extract the subsequence of combining diacritical marks:

```
diacritics(s) = [c for c in NFD(s) if unicodedata.combining(c)]
```

This captures combining acute (U+0301), combining grave (U+0300), and combining dot below (U+0323)—the three marks that encode tonal and sub-dot contrasts in Standard Yorùbá.

**Definition (Diacritic Error Rate):** Let d_gt = diacritics(y) and d_pred = diacritics(ŷ). Then:

```
DER = EditDistance(d_pred, d_gt) / max(1, |d_gt|)
```

DER is complementary to CER: two systems with similar CER can diverge sharply in DER if one systematically drops tone marks while the other makes uniformly distributed character errors. A system that perfectly preserves base characters but strips all diacritics would achieve moderate CER (since diacritics are a fraction of total characters) but catastrophic DER. Conversely, a system with high CER due to spacing or punctuation errors may achieve low DER if diacritic marks are preserved.

DER values exceeding 100% indicate that the system inserts spurious diacritics in addition to misrecognizing those present in the reference, which we observe in several zero-shot configurations.

# Experiments

## Evaluation Protocol

All systems received identical line-crop images from the held-out test split (n=326). No additional binarization, denoising, or resolution normalization was applied beyond model-specific input requirements. Per-sample CER, WER, and DER were computed and logged to JSONL files alongside predictions and ground truth for auditability. Aggregate metrics are the arithmetic mean across all test samples.

**Note on pre-trained baseline decoding.** When evaluating the English pre-trained PP-OCRv4 baseline, we configured the inference pipeline to decode using our Yorùbá character dictionary. While this differs from an out-of-the-box English setup, it constitutes a deliberate like-for-like comparison: by restricting the baseline to output characters within the Yorùbá character scope, we measure its ability to recognize diacritic topologies rather than unfairly penalizing it for vocabulary mismatches during decoding.

## Systems Compared

Seven configurations were evaluated on the test split:

1. **PaddleOCR PP-OCRv4 (EN pretrained)** — English-pretrained SVTR_LCNet with Yorùbá dictionary decoding
2. **Tesseract (eng)** — Tesseract 5, English language pack
3. **Tesseract (yor)** — Tesseract 5, Yorùbá language pack
4. **Tesseract (eng+yor)** — Tesseract 5, combined English + Yorùbá
5. **PaddleOCR-VL-1.5 (zero-shot)** — Vision-language model, fixed prompt, no adaptation
6. **Qwen 2.5 VL (zero-shot)** — Multimodal LLM, fixed prompt, temperature 0
7. **PaddleOCR-VL-1.5 (LoRA fine-tuned)** — LoRA-adapted on training split, primary supervised result

## PP-OCRv4 Data Size Ablation

To understand the relationship between training data volume and recognition quality for the classical CRNN pipeline, we trained PP-OCRv4 at four data fractions (25%, 50%, 75%, 100% of training data) and evaluated on both validation and test splits.
