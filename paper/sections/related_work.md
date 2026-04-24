# Related Work

## OCR for Low-Resource and Non-Latin Scripts

Industrial OCR pipelines have converged on detection-recognition architectures that achieve strong accuracy on Latin, CJK, and Arabic scripts (Du et al., 2020; Smith, 2007). PaddleOCR provides multiple model sizes including the recent vision-language variant VL-1.5, while Tesseract remains widely deployed through its open-source language packs. Both systems are typically benchmarked on high-resource corpora where diacritics are absent or non-contrastive; African tonal scripts are rarely primary targets in public leaderboards. Work on Amharic, Tigrinya, and Ge'ez script OCR has demonstrated that non-Latin character recognition benefits from script-specific training data and evaluation protocols [CITE: Ethiopian OCR work], yet no comparable effort exists for Yorùbá, whose Latin orthography masks the diacritic complexity beneath a familiar script surface.

## Diacritic Handling in NLP and OCR

Diacritic restoration has been studied as a standalone NLP task for Arabic, Vietnamese, and several European languages [CITE: diacritic restoration survey], where stripped text is re-accented using language models. For Yorùbá specifically, Orife (2018) and De Pauw et al. [CITE: Yorùbá diacritic restoration] have addressed automatic diacritic restoration from unaccented input. These approaches assume clean text as input, side-stepping the compounded difficulty of recognizing diacritics from noisy pixel data. OCR systems trained on English treat combining diacritics as optional noise rather than semantically load-bearing features, producing outputs where tone information is systematically stripped. The interaction between recognition errors and diacritic fidelity remains unquantified in prior work.

## African-Language NLP Benchmarks and Datasets

Cross-lingual corpora for African languages have primarily targeted downstream NLP tasks. MasakhaNER (Adelani et al., 2021) established named entity recognition benchmarks across multiple African languages; AfriSenti and related initiatives have expanded to sentiment analysis and machine translation. OCR complements these efforts: scanned pedagogical and literary texts are a major source of raw material for training language technologies, provided diacritically faithful transcriptions can be recovered. To our knowledge, no prior work releases a Yorùbá line-OCR benchmark with explicit diacritic-aware evaluation, nor does any existing African-language dataset enforce book-level splits to test typographic generalization.

## Multimodal Large Language Models for Document Understanding

Recent multimodal LLMs report strong document parsing and text spotting on English-centric evaluation suites (Qwen Team, 2025; Bai et al., 2023). Qwen 2.5 VL and similar architectures combine vision encoders with autoregressive language models, enabling them to leverage linguistic priors when reading text from images. Their behaviour on low-resource orthographies with contrastive diacritics is under-studied. A key question for Yorùbá OCR is whether these models' contextual reasoning compensates for graphemic errors—recovering the intended word sense from surrounding context even when diacritics are misrecognized—or whether such compensation introduces a different class of unfaithful transcription.
