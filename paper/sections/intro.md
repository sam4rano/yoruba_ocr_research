# Introduction

The digitization of African-language texts remains a significant bottleneck in natural language processing research for the continent. While OCR technology has reached near-human performance for high-resource languages such as English, Chinese, and German [CITE: high-resource OCR survey], low-resource tonal languages—particularly those with complex diacritic systems—remain severely underserved.

Yorùbá, spoken by over 50 million people across West Africa, presents a particularly acute challenge. The language employs a three-tone system (high, mid, and low) encoded orthographically through combinations of acute (´), grave (\`), and unmarked conventions applied to vowels and the syllabic nasal. Critically, these diacritics are not stylistic ornaments; they are semantically contrastive. The base form *ogun*, for instance, maps to four semantically unrelated lexical items depending solely on tonal marking: *ògún* (war), *ògùn* (medicine/juju), *ogún* (inheritance/share), and *ogún* (the number twenty). A system that drops or misreads tone marks does not produce a degraded transcription—it produces a semantically different and often nonsensical one.

Despite this linguistic reality, standard OCR evaluation metrics such as Character Error Rate (CER) and Word Error Rate (WER) treat all character substitutions equally. A system that transcribes *è* as *e* incurs the same penalty as one that substitutes an entirely unrelated character. This conflation obscures the specific failure mode most damaging for Yorùbá: the systematic dropping or misidentification of diacritics.

This paper addresses three interconnected gaps. There is no publicly available, linguistically validated OCR benchmark for Yorùbá. No prior work has systematically evaluated state-of-the-art multimodal large language models on Yorùbá OCR, despite their impressive performance on English text recognition tasks. And existing metrics are insufficiently sensitive to tonal error patterns.

We present the following contributions:

1. **A curated dataset** of 2,945 annotated line crops from the *Yorùbá di Wúrà* graded reader series (Books 1–6), authored by co-authors of this paper, providing first-party ground truth data with known typographic properties. Annotations are human-corrected, UTF-8 normalized (NFC), and constrained to a 99-character Yorùbá diacritic dictionary. The dataset uses a strict book-level train/test split to evaluate cross-font generalization.

2. **A systematic benchmark** comparing PaddleOCR PP-OCRv4, Tesseract (three language configurations), Qwen 2.5 VL, and PaddleOCR-VL-1.5 under zero-shot and LoRA fine-tuned conditions. The fine-tuned vision-language model achieves the lowest error rates across CER, WER, and DER on the held-out test split.

3. **Diacritic Error Rate (DER)**, a novel evaluation metric that isolates tone-mark recognition failures from general character-level errors, enabling more targeted diagnostics of system behaviour on tonal orthographies.

All data, code, and model checkpoints will be released publicly to support reproducible research on African-language OCR.
