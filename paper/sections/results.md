# Results and Analysis

## Main Comparison

The LoRA fine-tuned PaddleOCR-VL-1.5 achieves the lowest error rates across all three metrics (Table 1), with a mean CER of 96.5%, WER of 122.6%, and DER of 77.6% on the 326-sample test split. It is the only system to produce any perfect transcriptions (23 out of 326 lines with CER=0), and the only system where more than 10% of samples fall below 10% CER (33/326).

**Table 1: Main results on the book-level test split (n=326). Bold indicates best per column.**

| System | CER (%) ↓ | WER (%) ↓ | DER (%) ↓ |
|--------|----------:|----------:|----------:|
| PaddleOCR PP-OCRv4 (EN pretrained) | 174.5 | **100.0** | 110.9 |
| Tesseract (eng) | 120.4 | 153.5 | 95.9 |
| Tesseract (yor) | 124.4 | 163.7 | 87.1 |
| Tesseract (eng+yor) | 122.6 | 160.0 | 92.1 |
| PaddleOCR-VL-1.5 (zero-shot) | 543.3 | 840.9 | 227.3 |
| Qwen 2.5 VL (zero-shot) | 253.5 | 329.5 | 152.2 |
| **PaddleOCR-VL-1.5 (LoRA fine-tuned)** | **96.5** | 122.6 | **77.6** |

Error rates exceeding 100% are characteristic of systems that hallucinate text beyond the reference length—particularly the zero-shot vision-language models, which often generate verbose outputs including formatting artifacts, repeated text, or entirely fabricated content. PaddleOCR-VL-1.5 zero-shot is the most extreme case (CER 543.3%), suggesting the model's default behaviour without adaptation is to generate extended document-level predictions from single-line inputs.

## Baseline Analysis

Among the non-fine-tuned systems, Tesseract with the Yorùbá language pack achieves the lowest DER (87.1%), suggesting its language model provides some diacritic awareness. However, its CER (124.4%) and WER (163.7%) are high, indicating substantial base-character errors alongside partial diacritic preservation. The English-only Tesseract configuration shows the opposite pattern: marginally better CER (120.4%) but worse DER (95.9%), consistent with a model that reads character shapes more accurately but lacks the linguistic prior needed to select correct diacritics.

The PaddleOCR PP-OCRv4 English pretrained baseline achieves the lowest WER among non-fine-tuned systems (100.0%—meaning word-level errors equal the word count on average), but its DER of 110.9% indicates systematic diacritic hallucination and deletion. The phantom flag in this model's checkpoint audit confirms that head weights were not fully restored, which may contribute to its inconsistent diacritic behaviour.

## Vision-Language Models: Context vs. Fidelity

Qwen 2.5 VL occasionally recovers the semantically correct word in context despite diacritic errors. It produces 4 perfect transcriptions (CER=0) and 10 lines with perfect DER, compared to zero for any Tesseract configuration. This suggests the model's language component can infer intended words from visual and linguistic context. However, its mean CER (253.5%) and DER (152.2%) are far worse than Tesseract overall, driven by frequent hallucination—the model generates extended outputs, translations, or formatting that inflate edit distance.

This creates a paradox for digitization: a model that sometimes recovers meaning through context may be useful for semantic retrieval tasks, but it is unsuitable for faithful archival transcription where character-level fidelity is the primary desideratum. The gap between DER and CER for Qwen 2.5 VL (152.2% vs. 253.5%) suggests that when the model does recognize text, its diacritic errors are proportionally less severe than its overall character errors—but this advantage is overwhelmed by its propensity for hallucinated content.

## Effect of Fine-Tuning

The LoRA fine-tuned PaddleOCR-VL-1.5 reduces mean CER from 543.3% (zero-shot) to 96.5%—an 82.2% relative reduction. DER drops from 227.3% to 77.6% (65.9% relative reduction). The fine-tuned model's median CER (88.8%) and median DER (66.7%) suggest that for a majority of lines, the model produces transcriptions with meaningful overlap to the ground truth, though substantial errors remain.

Relative to the best non-fine-tuned system on each metric:
- CER: 96.5% vs. 120.4% (Tesseract eng) → 19.9% relative reduction
- DER: 77.6% vs. 87.1% (Tesseract yor) → 10.9% relative reduction
- WER: 122.6% vs. 100.0% (PP-OCRv4 EN) → the fine-tuned model does not achieve the lowest WER, suggesting word-boundary errors remain a challenge

## PP-OCRv4 Data Size Ablation

**Table 2: PP-OCRv4 fine-tuning at varying training data fractions (test split, n=326).**

| Training Data (%) | CER (%) ↓ | WER (%) ↓ | DER (%) ↓ |
|-------------------:|----------:|----------:|----------:|
| 25 | 89.7 | 101.5 | 88.7 |
| 50 | 91.6 | 102.2 | 88.7 |
| 75 | 91.5 | 103.3 | 88.8 |
| 100 | 91.5 | 103.4 | 91.8 |

The ablation reveals a counterintuitive pattern: performance does not monotonically improve with more training data. The 25% fraction achieves the lowest CER (89.7%) and tied-lowest DER (88.7%), while the 100% fraction shows slightly worse DER (91.8%). This suggests that for the classical CRNN architecture, the dataset may contain sufficient redundancy that additional samples do not contribute novel visual patterns, and potential label noise in the full dataset marginally degrades diacritic recognition. Alternatively, the book-level split introduces distribution shift that a larger training set from the same books cannot overcome.
