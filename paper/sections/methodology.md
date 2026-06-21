# Methodology

## Dataset Construction

### Source Material

Line crops originate from the *Yorùbá di Wúrà* graded reader series (Books 1–6), professionally typeset educational material authored by the creators of the graded reader series. The series uses consistent pedagogical progression with controlled vocabulary, providing ground truth with known typographic properties across six distinct book designs.

### Annotation Pipeline

Images were annotated using a custom annotation platform. Annotators segmented page scans at line granularity and transcribed each line into UTF-8 text. All transcriptions underwent NFC normalization to collapse equivalent Unicode representations (e.g., precomposed *é* vs. base *e* + combining acute). The pipeline processed 33 independent annotation export batches, yielding raw annotations that were then consolidated and deduplicated by script `01_consolidate_data.py`.

### Data Hygiene

A hygiene filter removed labels shorter than 3 characters or longer than 100 characters, lines containing non-Yorùbá codepoints, and entries flagged as invalid Yorùbá. The filter removed 640 invalid Yorùbá entries, 386 entries with non-whitelisted codepoints, 348 labels too short, and 236 labels too long. The resulting dataset comprises **2,945 unique line crops**: 2,367 train, 252 validation, and 326 test.

### Character Dictionary

The character dictionary contains **99 unique characters** (excluding the implicit space token), closed under NFC normalization. It encodes all tonal vowel variants (à, á, è, é, ì, í, ò, ó, ù, ú), sub-dotted vowels (ẹ, ọ) with their tonal combinations, the sub-dotted consonant (ṣ), the syllabic nasal (ń), uppercase variants, digits, and common punctuation. Coverage of in-distribution text is 99.0%.

### Train/validation/test split

All six *Yorùbá di Wúrà* volumes are merged into a single corpus after deduplication. Lines are assigned to train, validation, and test at **80/10/10** by uniform random sampling over unique line crops (`scripts/01_consolidate_data.py --resplit --seed 42`), yielding **2,356 train / 294 val / 295 test** from 2,945 unique lines.

This line-level partition maximises training volume but does not hold out entire volumes; typographic generalisation across book designs is therefore not guaranteed by the split alone. Reported model scores in Table 1 (n=326 test) reflect the consolidated export partition at evaluation time and will be refreshed after applying the documented 80/10/10 re-split.

## Model Configurations

### PaddleOCR PP-OCRv4 (Classical Comparison)

We fine-tuned the English-pretrained PP-OCRv4 recognition model (SVTR_LCNet architecture with CTC head) on the full training split with our 99-character Yorùbá dictionary. Configuration: MobileNetV1Enhance backbone (scale 0.5), SequenceEncoder neck (SVTR, dims=64, depth=2), input resolution 3×48×960. Training used Adam (β₁=0.9, β₂=0.999), cosine learning rate schedule (initial lr=0.001, warmup 2 epochs), L2 regularization (factor=3×10⁻⁵), batch size 64, for 40 epochs with RecAug augmentation. The best-accuracy validation checkpoint is reported in Table 1 (100% data-size ablation run).

### Out-of-the-box baselines (TrOCR, English PP-OCRv4)

**TrOCR-large-printed** (`microsoft/trocr-large-printed`) was evaluated zero-shot on line crops with greedy decoding. **English-pretrained PP-OCRv4** uses the stock English recognition checkpoint but decodes with the project Yorùbá character dictionary so that comparisons isolate script adaptation rather than decode-vocabulary mismatch.

### Qwen 2.5 VL (Zero-Shot)

Qwen 2.5 VL was evaluated in zero-shot mode with a fixed instruction template requesting line transcription and temperature set to 0 for deterministic decoding. No fine-tuning or adaptation was applied.

### PaddleOCR-VL-1.5 (Primary Supervised Model)

**Zero-shot.** The PaddleOCR-VL-1.5 model (Hugging Face: `PaddlePaddle/PaddleOCR-VL-1.5`) was evaluated with the same fixed instruction prompt requesting Yorùbá line transcription.

**LoRA fine-tuning.** We applied Low-Rank Adaptation to the language model layers only (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj), excluding the vision encoder (SigLIP) to avoid overfitting visual features on 2,367 training samples. LoRA configuration: rank r=16, α=32, dropout=0.05. Training used AdamW optimizer (lr=2×10⁻⁴, weight decay=0.01) with linear warmup (10% of steps) followed by cosine decay. Gradient clipping was applied at max norm 1.0. Images were resized with thumbnail capping at 800×800 pixels (matching evaluation resolution at max_pixels = 768×28×28 = 602,112). The training objective was assistant-only causal language modelling loss: prompt tokens (vision + user text + generation header) were masked with label=-100, consistent with standard HF/TRL supervised fine-tuning practice. Gradient checkpointing was enabled for memory efficiency. Training ran for 1 epoch on a single NVIDIA T4 GPU (Google Colab Pro).

## Evaluation Protocol

### Metrics

All metrics are computed on NFC-normalized Unicode strings.

**Character Error Rate (CER):** The Levenshtein edit distance between predicted and ground-truth strings, normalized by ground-truth string length:

CER = EditDistance(ŷ, y) / max(1, |y|)

**Word Error Rate (WER):** The word-level edit distance, normalized by ground-truth word count.

**Diacritic Error Rate (DER):** Defined formally in § Diacritic Error Rate.

### Preprocessing

All systems received identical line-crop inputs without additional binarization or resolution normalization beyond model-specific requirements. Evaluation was performed on the held-out test split. Table 1 reports n=326 from the consolidated export prior to enforced 80/10/10 re-partitioning (295 test lines under seed=42).
