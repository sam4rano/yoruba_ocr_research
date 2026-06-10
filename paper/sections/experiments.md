# Diacritic Error Rate

Standard CER weights all character errors equally: a substitution of *ò* → *o* (tone loss) incurs the same penalty as *ò* → *z* (random corruption). For Yorùbá OCR, the former is the dominant and most consequential failure mode.

Let Σ denote the Standard Yorùbá character set with |Σ| = 99 after Unicode Normalization Form C (NFC). Let y ∈ Σ* denote ground truth and ŷ ∈ Σ* the hypothesis. Both strings undergo NFC normalization and detached-mark repair (collapsing spurious whitespace or apostrophes between a base letter and its combining mark).

Decompose both sequences into Normalization Form D (NFD). Let 𝒰 denote combining acute (U+0301), combining grave (U+0300), and combining dot below (U+0323). The diacritic subsequence extractor is:

```
d(s) = (cᵢ | cᵢ ∈ NFD(s) ∧ cᵢ ∈ 𝒰)
```

Per-sample DER:

```
DER(y, ŷ) = D_Lev(d(ŷ), d(y)) / max(1, |d(y)|)
```

where D_Lev is Levenshtein distance over diacritic mark sequences.

**Corpus DER (reported in tables):** For test pairs {(yᵢ, ŷᵢ)}, let S = {i : |d(yᵢ)| > 0}. Then:

```
DER_corpus = Σ_{i∈S} D_Lev(d(ŷᵢ), d(yᵢ)) / Σ_{i∈S} |d(yᵢ)|
```

Samples with no GT diacritics are excluded from corpus DER; spurious predicted marks on those lines are tracked via `der_insertion_rate` (see `results/tables/der_zero_diac_insertion.csv`).

### 𝒰 sensitivity ablation

`scripts/18_der_universe_ablation.py` recomputes corpus DER under four extractors without re-running inference. Results: `results/tables/der_universe_ablation.csv`. Mark-level rankings (combining / tone-only / all-combining) match Table 1 ordering; marked-grapheme NFC tokens yield higher DER (stricter whole-character criterion).

DER is complementary to CER: two systems with similar CER can diverge in DER if one systematically drops tone marks while the other distributes errors uniformly. Values exceeding 100% indicate spurious diacritic insertions alongside misrecognitions.

# Experiments

## Evaluation Protocol

All systems received identical line-crop images from the held-out test split (n=326). No additional binarization, denoising, or resolution normalization was applied beyond model-specific input requirements. Per-sample CER, WER, and DER were computed and logged to JSONL files alongside predictions and ground truth for auditability. Aggregate metrics are the arithmetic mean across all test samples.

## Systems Compared

The benchmark spans out-of-the-box recognisers (Section A), fine-tuned vision-language and Foundation models (Section B), and classical PP-OCRv4 CRNN ablations (Section C). On the held-out test split we report at minimum:

1. **PaddleOCR PP-OCRv4 (English pretrained)** — English recognition weights, Yorùbá character dictionary at decode time
2. **TrOCR-large-printed (zero-shot)** — Microsoft TrOCR on printed Latin text, no Yorùbá adaptation
3. **Surya v2 (zero-shot, recognition-only)** — VikParuchuri/surya recognition stack without detection
4. **PaddleOCR-VL-1.5 (zero-shot)** — Vision-language model, fixed prompt, no adaptation
5. **Qwen 2.5 VL (zero-shot)** — Multimodal LLM, fixed prompt, temperature 0
6. **PaddleOCR-VL-1.5 (LoRA fine-tuned)** — LoRA-adapted on training split, primary supervised result
7. **PaddleOCR PP-OCRv4 (CRNN fine-tuned)** — full training split, 40 epochs (100% data-size ablation checkpoint)
8. **Surya Foundation (fine-tuned)** — optional; scripts 26–28 on merged train+val export

## PP-OCRv4 Data Size Ablation

To understand the relationship between training data volume and recognition quality for the classical CRNN pipeline, we trained PP-OCRv4 at four data fractions (25%, 50%, 75%, 100% of training data) and evaluated on both validation and test splits.
