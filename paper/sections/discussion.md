# Discussion

## Why Fine-Tuning Succeeds Where Zero-Shot Fails

The dramatic improvement from LoRA fine-tuning (CER reduction from 543.3% to 96.5%) reflects a fundamental mismatch between the VL-1.5 model's pretraining distribution and the Yorùbá OCR task. Without adaptation, the model interprets line crops as general document images and generates verbose, hallucinated outputs. Fine-tuning constrains the model's generation behaviour to the expected output format—single-line Yorùbá transcriptions—while the vision encoder (frozen during LoRA training) preserves its ability to extract character-level visual features. This architectural decision is consistent with prior findings that adapter-based methods are sufficient when the visual domain is well-covered by the pretrained encoder and only the output distribution requires realignment.

## The Diacritic Fidelity Gap

Even the best-performing system (LoRA fine-tuned VL-1.5, DER 77.6%) misrecognizes diacritics in a majority of lines. The median DER of 66.7% means that on a typical line, roughly two-thirds of diacritic marks are incorrectly predicted. For a language where every diacritic carries lexical meaning, this error rate renders the output unreliable for archival digitization or downstream NLP tasks that assume correct tonal marking.

The DER gap between Tesseract (yor) at 87.1% and Tesseract (eng) at 95.9% confirms that language-specific priors improve diacritic selection—but by a modest margin insufficient for practical use. The persistence of high DER across all systems, including the fine-tuned model, suggests that recognizing combining diacritics from printed Yorùbá text is inherently difficult at current model capacities and dataset scales.

## Contextual Inference vs. Graphemic Fidelity

Qwen 2.5 VL's occasional perfect transcriptions (4/326) amid generally poor performance illustrate a tension between semantic and graphemic faithfulness. The model sometimes produces the correct diacritized word by inferring meaning from context—selecting *ògún* (war) over *ogún* (twenty) based on surrounding words, for instance. This contextual compensation is useful for tasks like information retrieval or topic classification, where recovering the intended meaning matters more than exact graphemic reproduction.

For archival digitization and corpus construction, however, this behaviour is problematic. A system that infers diacritics from context rather than reading them from the image may introduce systematic biases toward frequent word senses, silently corrupting the very linguistic data that researchers need to study. DER surfaces this distinction: a model with low semantic error but high DER is embedding its own language model priors into the transcription rather than faithfully reproducing the source document.

## Implications for African-Language OCR

The high error rates across all systems underscore the scale of the challenge for low-resource tonal languages. Even with fine-tuning on nearly 2,400 training samples, CER remains above 90%—far from the sub-5% rates achieved on English OCR benchmarks. Several factors contribute: the small training set, typographic diversity across books, high diacritic density in Yorùbá text, and the absence of large-scale pretrained models for Yorùbá script.

These results motivate investment in larger annotated datasets, potentially augmented with synthetic data for rare diacritic combinations, and in OCR architectures that explicitly model diacritical marks as structured predictions rather than independent character classifications.

# Limitations

The dataset covers a single pedagogical book series (*Yorùbá di Wúrà*), limiting font and layout diversity. Books with older typesetting conventions, handwritten elements, or informal orthographic practices are not represented. The book-level split ensures typographic generalization is tested, but performance on radically different document styles (newspapers, government records, social media screenshots) remains unknown.

DER depends on the choice of which Unicode codepoints constitute the diacritic-critical set U. Our definition uses `unicodedata.combining()` to identify combining marks, which captures the three primary marks (acute, grave, dot below) but may not account for all edge cases in non-standard encodings. Sensitivity to the choice of U was not formally ablated.

The multimodal LLM evaluation (Qwen 2.5 VL) is sensitive to prompt wording and decoding parameters. We report results using a fixed instruction template and temperature 0; alternative prompting strategies or few-shot examples might yield different performance profiles. Similarly, the LoRA fine-tuning used a single epoch due to GPU memory constraints on the T4 GPU; additional training epochs or larger adapter ranks might improve results.

The dataset count (2,945 after hygiene filtering) is substantially smaller than the 3,626 initially reported in preliminary versions of this work, reflecting the removal of code-mixed content, overly short labels, and non-whitelisted codepoints during quality control. All reported results use the cleaned dataset.

# Conclusion

A line-image OCR benchmark for Yorùbá with book-level splits, human-corrected annotations constrained to a 99-character Standard Yorùbá dictionary, and systematic evaluation of classical, vision-language, and multimodal LLM systems reveals that diacritic recognition remains the critical bottleneck for tonal-language digitization. The LoRA fine-tuned PaddleOCR-VL-1.5 achieves the best overall performance (CER 96.5%, DER 77.6%), substantially outperforming zero-shot baselines—but the absolute error rates remain high, indicating that faithful Yorùbá OCR is far from solved. DER, proposed as a complementary metric to CER, exposes the specific failure mode that aggregate character metrics obscure: the systematic loss of tonal information that destroys lexical identity. Public release of the dataset, evaluation code, and model checkpoints provides the first reproducible foundation for advancing OCR infrastructure for Yorùbá and, by extension, for the broader family of African tonal languages whose written records await faithful digitization.
