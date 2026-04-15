# Experiments

## Evaluation Protocol / Baselines

**Note on Pre-trained Baselines Decoding**:
When evaluating the English pre-trained baseline (PP-OCRv4), we intentionally configure the inference pipeline to decode using our custom Yorùbá character dictionary. While this differs from an "out-of-the-box" English setup, it constitutes a deliberate, like-for-like comparison against our fine-tuned models. By restricting the English baseline to output characters within the Yorùbá scope, we measure its absolute capability to recognize localized topologies rather than unfairly penalizing it for vocabulary mismatches or out-of-domain Unicode tokenization during decoding.
