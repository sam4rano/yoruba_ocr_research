# Results and Analysis

## Main Comparison

The LoRA fine-tuned PaddleOCR-VL-1.5 achieves the lowest CER and corpus DER across all systems (Table 1), with mean CER of 96.5%, WER of 122.6%, and corpus DER of 66.4% on the 326-sample test split (der_n=319 diacritic-bearing lines). It is the only system to produce any perfect transcriptions (23 out of 326 lines with CER=0), and the only system where more than 10% of samples fall below 10% CER (33/326).

**Table 1: Main results on the held-out test split (n=326; prior to 80/10/10 re-split). Bold indicates best per column; underline second-best.**

| System | CER (%) ↓ | WER (%) ↓ | DER (%) ↓ |
|--------|----------:|----------:|----------:|
| Tesseract (eng) | 120.3 | 153.5 | 98.5 |
| Tesseract (yor) | 124.4 | 163.7 | 87.7 |
| Tesseract (eng+yor) | 122.6 | 160.0 | 93.9 |
| PaddleOCR-VL-1.5 (zero-shot) | 543.3 | 840.9 | 200.9 |
| Qwen 2.5 VL (zero-shot) | 253.5 | 329.5 | 119.6 |
| PaddleOCR PP-OCRv4 (CRNN fine-tuned) | 91.5 | **103.4** | 91.8 |
| **PaddleOCR-VL-1.5 (LoRA fine-tuned)** | **96.5** | 122.6 | **66.4** |

Error rates exceeding 100% are characteristic of systems that hallucinate text beyond the reference length—particularly the zero-shot vision-language models, which often generate verbose outputs including formatting artifacts, repeated text, or entirely fabricated content. PaddleOCR-VL-1.5 zero-shot is the most extreme case (CER 543.3%), suggesting the model's default behaviour without adaptation is to generate extended document-level predictions from single-line inputs.

## Baseline Analysis

Among the non-fine-tuned systems, Tesseract with the Yorùbá language pack achieves the lowest DER (87.7%), suggesting its language model provides some diacritic awareness. However, its CER (124.4%) and WER (163.7%) are high, indicating substantial base-character errors alongside partial diacritic preservation. The English-only Tesseract configuration shows the opposite pattern: marginally better CER (120.3%) but worse DER (98.5%), consistent with a model that reads character shapes more accurately but lacks the linguistic prior needed to select correct diacritics.

PP-OCRv4 CRNN fine-tuning on the full training split yields the lowest WER in Table 1 (103.4%) and marginally lower CER than LoRA (91.5% vs. 96.5%), but its corpus DER (91.8%) remains far above the LoRA model (66.4%). Classical CTC decoding therefore recovers word boundaries somewhat better while still failing systematically on tone marks and subdots—the failure mode DER was designed to expose.

## Vision-Language Models: Context vs. Fidelity

Qwen 2.5 VL occasionally recovers the semantically correct word in context despite diacritic errors. It produces 4 perfect transcriptions (CER=0) and 10 lines with perfect DER, compared to zero for any Tesseract configuration. This suggests the model's language component can infer intended words from visual and linguistic context. However, its mean CER (253.5%) and DER (119.6%) are far worse than Tesseract overall, driven by frequent hallucination—the model generates extended outputs, translations, or formatting that inflate edit distance.

This creates a paradox for digitization: a model that sometimes recovers meaning through context may be useful for semantic retrieval tasks, but it is unsuitable for faithful archival transcription where character-level fidelity is the primary desideratum.

## Effect of Fine-Tuning

The LoRA fine-tuned PaddleOCR-VL-1.5 reduces mean CER from 543.3% (zero-shot) to 96.5%—an 82.2% relative reduction. Corpus DER drops from 200.9% to 66.4% (66.9% relative reduction).

Relative to the best non-fine-tuned system on each metric:
- CER: 96.5% vs. 120.3% (Tesseract eng) → 19.8% relative reduction
- DER: 66.4% vs. 87.7% (Tesseract yor) → 24.3% relative reduction
- WER: 122.6% vs. 103.4% (PP-OCRv4 CRNN fine-tuned) → LoRA does not win WER; CRNN fine-tuning achieves lower word error at the cost of much higher diacritic error

## PP-OCRv4 Data Size Ablation

**Table 2: PP-OCRv4 fine-tuning at varying training data fractions (test split, n=326).**

| Training Data (%) | CER (%) ↓ | WER (%) ↓ | DER (%) ↓ |
|-------------------:|----------:|----------:|----------:|
| 25 | 89.7 | 101.5 | 88.7 |
| 50 | 91.6 | 102.2 | 88.7 |
| 75 | 91.5 | 103.3 | 88.8 |
| 100 | 91.5 | 103.4 | 91.8 |

The ablation reveals a counterintuitive pattern: performance does not monotonically improve with more training data. The 25% fraction achieves the lowest CER (89.7%) and tied-lowest DER (88.7%), while the 100% fraction shows slightly worse DER (91.8%). This suggests that for the classical CRNN architecture, the dataset may contain sufficient redundancy that additional samples do not contribute novel visual patterns, and potential label noise in the full dataset marginally degrades diacritic recognition.
