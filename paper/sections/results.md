# Results and Analysis

## Main Comparison

The LoRA fine-tuned PaddleOCR-VL-1.5 achieves the lowest corpus DER across all systems evaluated on the current test split (Table 1), with mean CER of 96.5%, WER of 122.6%, and corpus DER of 66.4% on n=326 lines (der_n=319 diacritic-bearing lines). It is the only system to produce any perfect transcriptions (23 out of 326 lines with CER=0), and the only system where more than 10% of samples fall below 10% CER (33/326).

**Table 1: Main results on the held-out test split (n=326). Bold indicates best per column; underline second-best.**

| System | CER (%) ↓ | Median CER (%) ↓ | Micro CER (%) ↓ | WER (%) ↓ | DER (%) ↓ |
|--------|----------:|-----------------:|----------------:|----------:|----------:|
| PaddleOCR-VL-1.5 (zero-shot) | 543.3 | 100.0 | 270.6 | 840.9 | 200.9 |
| Qwen 2.5 VL-7B (zero-shot, archived pilot) | 253.5 | 114.3 | 158.1 | 329.5 | 119.6 |
| **PaddleOCR-VL-1.5 (LoRA fine-tuned)** | 96.5 | **88.8** | **84.9** | 122.6 | **66.4** |
| PaddleOCR PP-OCRv4 (CRNN fine-tuned — comparison) | **91.5** | 96.3 | 93.9 | **103.4** | 91.8 |

TrOCR-large-printed (zero-shot), Surya v2 (zero-shot), and English-pretrained PP-OCRv4 rows are populated after Section A baseline runs; see `results/tables/metrics.csv`.

Error rates exceeding 100% are characteristic of systems that hallucinate text beyond the reference length—particularly the zero-shot vision-language models, which often generate verbose outputs including formatting artifacts, repeated text, or entirely fabricated content. PaddleOCR-VL-1.5 zero-shot is the most extreme case (CER 543.3%), suggesting the model's default behaviour without adaptation is to generate extended document-level predictions from single-line inputs.

## Baseline Analysis

Among the supervised classical systems, PP-OCRv4 CRNN fine-tuning on the full training split yields the lowest WER in Table 1 (103.4%) and marginally lower CER than LoRA (91.5% vs. 96.5%), but its corpus DER (91.8%) remains far above the LoRA model (66.4%). Classical CTC decoding therefore recovers word boundaries somewhat better while still failing systematically on tone marks and subdots—the failure mode DER was designed to expose.

## Vision-Language Models: Context vs. Fidelity

Qwen 2.5 VL occasionally recovers the semantically correct word in context despite diacritic errors. It produces 4 perfect transcriptions (CER=0) and 10 lines with perfect DER. This suggests the model's language component can infer intended words from visual and linguistic context. However, its mean CER (253.5%) and DER (119.6%) are far worse than the CRNN fine-tune overall, driven by frequent hallucination—the model generates extended outputs, translations, or formatting that inflate edit distance.

This creates a paradox for digitization: a model that sometimes recovers meaning through context may be useful for semantic retrieval tasks, but it is unsuitable for faithful archival transcription where character-level fidelity is the primary desideratum.

## Effect of Fine-Tuning

The LoRA fine-tuned PaddleOCR-VL-1.5 reduces mean CER from 543.3% (zero-shot) to 96.5%—an 82.2% relative reduction. Median CER decreases from 100.0% to 88.8%, and Micro CER decreases from 270.6% to 84.9%. Corpus DER drops from 200.9% to 66.4% (66.9% relative reduction).

Relative to PP-OCRv4 CRNN fine-tuning on the same split:
- CER: 96.5% vs. 91.5% → LoRA does not win mean CER; CRNN retains a 5.0 pp advantage. However, LoRA achieves a lower Median CER (88.8% vs. 96.3%) and Micro CER (84.9% vs. 93.9%), demonstrating superior performance on typical/typical-length sentences despite macro-mean inflation.
- DER: 66.4% vs. 91.8% → 27.7% relative reduction in diacritic error
- WER: 122.6% vs. 103.4% → LoRA WER is 19.3 pp higher (bootstrap 95% CI [7.4, 32.9])

## PP-OCRv4 Data Size Ablation

**Table 2: PP-OCRv4 fine-tuning at varying training data fractions (test split, n=326).**

| Training Data (%) | CER (%) ↓ | WER (%) ↓ | DER (%) ↓ |
|-------------------:|----------:|----------:|----------:|
| 25 | 89.7 | 101.5 | 88.7 |
| 50 | 91.6 | 102.2 | 88.7 |
| 75 | 91.5 | 103.3 | 88.8 |
| 100 | 91.5 | 103.4 | 91.8 |

The ablation reveals a counterintuitive pattern: performance does not monotonically improve with more training data. The 25% fraction achieves the lowest CER (89.7%) and tied-lowest DER (88.7%), while the 100% fraction shows slightly worse DER (91.8%). This suggests that for the classical CRNN architecture, the dataset may contain sufficient redundancy that additional samples do not contribute novel visual patterns, and potential label noise in the full dataset marginally degrades diacritic recognition.

## Minimal-Pair and Stratified Error Analysis

We mined **106 diacritic minimal-pair skeleton groups** from the test vocabulary (NFD base strings with ≥2 distinct surface tonographs; e.g. *eko* → Ẹ̀KỌ́, Ẹ̀kọ́, ẹ̀kọ́) and evaluated systems on **279 lines** (85.6% of the split) containing at least one such type. Source: `results/tables/minimal_pair_vocabulary.json`, `results/tables/minimal_pair_subset.csv`.

**Table 3: Minimal-pair evaluation subset (n=279).**

| System | CER (%) ↓ | WER (%) ↓ | DER (%) ↓ |
|--------|----------:|----------:|----------:|
| PaddleOCR PP-OCRv4 (CRNN fine-tuned) | 92.1 | 99.6 | 91.9 |
| **PaddleOCR-VL-1.5 (LoRA fine-tuned)** | **88.8** | 105.6 | **64.9** |

On this subset, LoRA retains the lowest DER while improving CER relative to the full split (88.8% vs. 96.5%), indicating that failures on tonographically rich lines are not the sole driver of headline error rates.

**Diacritic density quartiles** (combining-mark count / NFC character length; edges in `stratified_error_analysis.json`): LoRA corpus DER decreases from **79.4%** (Q1, n=88) to **56.7%** (Q4, n=79), whereas PP-OCRv4 CRNN fine-tuning stays near **90–93%** across quartiles (`stratified_der_by_density.csv`). LoRA appears relatively stronger on high-density lines—possibly because heavily marked text is more visually distinctive in this book series.

**Per-book DER (LoRA)** is currently consolidated under a single `"unknown"` book source in `stratified_der_by_book.csv` (since explicit book metadata is absent in the processed splits).

**Error taxonomy** (n_der=319 diacritic-bearing lines; `error_taxonomy.csv`):

| Category | LoRA | PP-OCRv4 CRNN |
|----------|-----:|--------------:|
| Exact diacritics | 32 (10.0%) | 0 |
| Deletion-heavy | 161 (50.5%) | 188 (58.9%) |
| Insertion-heavy | 95 (29.8%) | 2 (0.6%) |
| Substitution | 22 (6.9%) | 1 (0.3%) |
| Total tone drop | 9 (2.8%) | 128 (40.1%) |

LoRA more often partially edits the tonograph sequence (deletions and insertions), with 32 lines of exact diacritic recovery. PP-OCRv4 fine-tuning remains deletion-heavy with frequent total tone drop—aligning with CTC's weak diacritic inductive bias despite lower WER on the full split.

## DER Universe Ablation

We recomputed corpus DER under four diacritic-universe definitions from test JSONL logs (`scripts/18_der_universe_ablation.py`; `results/tables/der_universe_ablation.csv`).

**Table 4: Corpus DER by 𝒰 definition (n_der=319 for mark-level rows).**

| 𝒰 definition | LoRA (%) | PP-OCRv4 CRNN (%) |
|--------------|---------:|------------------:|
| Combining marks (default: U+0300, U+0301, U+0323) | 66.3 | 91.8 |
| Tone only (excludes subdot) | 65.8 | 95.9 |
| All combining (NFD) | 66.4 | 91.8 |
| Marked grapheme (NFC tonographs) | 87.8 | 95.4 |

LoRA remains lowest under mark-level definitions (i–iii). The marked-grapheme tier is stricter—whole-character tonograph errors dominate. Headline Table 1 DER aligns with the “all combining” row within ≤0.2pp because non-standard combining marks are rare in model output on this split.

**Zero-diacritic ground-truth lines** (`der_zero_diac_insertion.csv`): 7 test lines (126 GT characters) carry no combining marks. LoRA predicts 14 spurious marks (insertion rate 0.111); PP-OCRv4 predicts 2 (rate 0.016). These lines are excluded from corpus DER but matter for mid-tone vowel hallucination. *Note that because this subset is extremely small (n=7 lines), these rates should be interpreted as purely diagnostic rather than statistically robust.*

## Bootstrap Confidence Intervals

Line-level bootstrap on n=326 test lines (B=10,000, seed=42; `scripts/19_bootstrap_metric_cis.py`).

**Table 5: Corpus DER with 95% CIs** (`bootstrap_metric_cis.csv`).

| System | DER (%) | 95% CI |
|--------|--------:|--------|
| PaddleOCR-VL-1.5 (LoRA) | 66.4 | [62.3, 70.6] |
| PaddleOCR PP-OCRv4 (CRNN fine-tuned) | 91.8 | [90.4, 93.1] |

**Pairwise DER gaps** (`bootstrap_pairwise_comparison.csv`): LoRA − PP-OCRv4 = −25.3 pp [−29.3, −21.2] (P=1.0 that LoRA DER is lower). LoRA WER remains higher than PP-OCRv4 (+19.3 pp [+7.4, +32.9]).

**Inter-annotator agreement:** Raw exports store only final corrected labels; Cohen's κ on diacritic edits is not available without a dedicated dual-annotation audit (`bootstrap_metric_cis.json` documents this limitation).
