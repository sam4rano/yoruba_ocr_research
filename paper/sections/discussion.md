# Discussion

## Diacritic Fidelity

The central question is whether OCR systems preserve Yorùbá tone marks and subdots, not merely whether they recover base Latin letters. DER is designed to expose this failure mode. Once regenerated, the main comparison should be interpreted alongside DER insertion rates on zero-diacritic lines so hallucinated marks are visible.

## Model Behavior

Vision-language models may use linguistic context to produce plausible text, but archival OCR requires graphemic fidelity to the image. The evaluation therefore logs raw per-sample predictions and uses deterministic generation, fixed prompts, and post-hoc cleaning only for common formatting artifacts.

## Limitations

The dataset comes from one pedagogical book series and may not represent newspapers, handwriting, degraded scans, or modern digital typography. The current split is line-level rather than volume-held-out, so cross-book generalization remains an open stress test.

The dataset is small relative to high-resource OCR benchmarks. Results should be framed as diagnostic evidence for Yorùbá diacritic OCR rather than production estimates.

## Reproducibility

Before citing any metric, rerun the active pipeline on the frozen split, regenerate `research_approach.md`, and verify alignment/checkpoint reports. Generated tables and figures must come from real CSV/JSONL inputs; placeholder plots are disabled by default.
