# Discussion

## Why Fine-Tuning Succeeds Where Zero-Shot Fails

The dramatic improvement from LoRA fine-tuning (CER reduction from 543.3% to 96.5%) reflects a fundamental mismatch between the VL-1.5 model's pretraining distribution and the Yorùbá OCR task. Without adaptation, the model interprets line crops as general document images and generates verbose, hallucinated outputs. Fine-tuning constrains the model's generation behaviour to the expected output format—single-line Yorùbá transcriptions—while the vision encoder (frozen during LoRA training) preserves its ability to extract character-level visual features.

## The Diacritic Fidelity Gap

Even the best-performing system (LoRA fine-tuned VL-1.5, corpus DER 66.4%) misrecognizes roughly two-thirds of combining diacritics on diacritic-bearing lines (der_n=319). For a language where every diacritic carries lexical meaning, this renders outputs unreliable for archival digitization or downstream NLP tasks that assume correct tonal marking.

The DER gap between Tesseract (yor) at 87.7% and Tesseract (eng) at 98.5% confirms that language-specific priors improve diacritic selection—but by a modest margin insufficient for practical use.

## Contextual Inference vs. Graphemic Fidelity

Qwen 2.5 VL's occasional perfect transcriptions (4/326) amid generally poor performance illustrate a tension between semantic and graphemic faithfulness. For archival digitization, a system that infers diacritics from context rather than reading them from the image silently corrupts linguistic data. DER surfaces this distinction.

## Implications for African-Language OCR

The high error rates across all systems underscore the scale of the challenge for low-resource tonal languages. Merging all six volumes into one training corpus increases sample count but removes volume-disjoint evaluation; future work should report per-volume or cross-domain stress tests alongside line-level splits.

# Limitations

**Domain and typography.** The dataset covers a single pedagogical book series (*Yorùbá di Wúrà*). Newspapers, handwriting, and informal digital text are not represented.

**Split policy.** The release applies an 80/10/10 line-level split (seed=42) after merging all volumes. This does not guarantee test lines come from unseen book designs. Table 1 metrics (n=326 test) predate enforced re-partitioning (295 test lines) and require re-evaluation.

**Annotation protocol.** Initial hypotheses came from PaddleOCR-VL-1.5; two annotators corrected every line, but exports lack per-annotator fields so κ cannot be recomputed. Bootstrap 95% CIs on n=326 test lines quantify sampling uncertainty instead (`bootstrap_metric_cis.csv`).

**Model coverage.** Surya v2 (recognition-only, zero-shot) and TrOCR-large-printed (fine-tuned on all 2,945 lines) are now scripted (`20`–`22`); Nougat-style document OCR remains excluded. LoRA used one epoch on a T4 GPU without validation early stopping.

**Metrics.** Corpus DER uses combining marks U+0300/U+0301/U+0323 by default; four 𝒰 definitions are ablated in `der_universe_ablation.csv`. Zero-diacritic GT lines (n=7) summarised in `der_zero_diac_insertion.csv`. MLLM scores are prompt-sensitive.

**Scale.** 2,945 lines is small relative to high-resource OCR benchmarks; results are diagnostic, not production estimates.

# Conclusion

A line-image OCR benchmark for Yorùbá with a merged six-volume corpus and reproducible 80/10/10 line-level split reveals that diacritic recognition remains the critical bottleneck. The LoRA fine-tuned PaddleOCR-VL-1.5 achieves the best overall performance (CER 96.5%, corpus DER 66.4%), substantially outperforming zero-shot baselines—but absolute error rates remain high. DER exposes the systematic loss of tonal information that CER obscures.
