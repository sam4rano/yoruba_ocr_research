# Methodology

## Dataset Construction

### Source Material

Line crops originate from the *Yorùbá di Wúrà* graded reader series (Books 1–6), professionally typeset educational material authored by co-authors of this paper. The series uses consistent pedagogical progression with controlled vocabulary, providing ground truth with known typographic properties across six distinct book designs.

### Annotation Pipeline

Images were annotated using a custom web-based annotation platform (yoruba-ocr.vercel.app). Annotators segmented page scans at line granularity and transcribed each line into UTF-8 text. All transcriptions underwent NFC normalization to collapse equivalent Unicode representations (e.g., precomposed *é* vs. base *e* + combining acute). The pipeline processed 33 independent annotation export batches, yielding raw annotations that were then consolidated and deduplicated by script `01_consolidate_data.py`.

### Data Hygiene

A hygiene filter removed labels shorter than 3 characters or longer than 100 characters, lines containing non-Yorùbá codepoints, and entries flagged as invalid Yorùbá. The filter removed 640 invalid Yorùbá entries, 386 entries with non-whitelisted codepoints, 348 labels too short, and 236 labels too long. The resulting dataset comprises **2,945 unique line crops**: 2,367 train, 252 validation, and 326 test.

### Character Dictionary

The character dictionary contains **99 unique characters** (excluding the implicit space token), closed under NFC normalization. It encodes all tonal vowel variants (à, á, è, é, ì, í, ò, ó, ù, ú), sub-dotted vowels (ẹ, ọ) with their tonal combinations, the sub-dotted consonant (ṣ), the syllabic nasal (ń), uppercase variants, digits, and common punctuation. Coverage of in-distribution text is 99.0%.

### Train/Test Split

To probe generalization across typography, we enforce a **book-level split**: training and test sets share no volume from the six-book series. This is stricter than random line splits and better reflects deployment on unseen book designs with different fonts, page layouts, and print quality. The validation set is drawn from books not present in the test set.

## Model Configurations

### PaddleOCR PP-OCRv4 (Classical Baseline)

We evaluate the English-pretrained PP-OCRv4 recognition model (SVTR_LCNet architecture with CTC head) using our Yorùbá character dictionary for decoding. This isolates the model's visual recognition capability on Yorùbá glyphs without penalizing it for vocabulary mismatches. Configuration: MobileNetV1Enhance backbone (scale 0.5), SequenceEncoder neck (SVTR, dims=64, depth=2), input resolution 3×48×960.

For the fine-tuned PP-OCRv4 comparison, we trained with Adam optimizer (β₁=0.9, β₂=0.999), cosine learning rate schedule (initial lr=0.001, warmup 2 epochs), L2 regularization (factor=3×10⁻⁵), batch size 64, for 40 epochs. RecAug augmentation was applied during training. Checkpoints were saved every 10 epochs; the best-accuracy checkpoint on validation was selected.

### Tesseract

Tesseract 5 was evaluated under three language configurations: English only (`eng`), Yorùbá only (`yor`), and combined (`eng+yor`). All used default parameters with no additional preprocessing beyond the line-crop input format.

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

All systems received identical line-crop inputs without additional binarization or resolution normalization beyond model-specific requirements. Evaluation was performed on the held-out test split (n=326 lines) from books not seen during training.
