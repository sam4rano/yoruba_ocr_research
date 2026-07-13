# Experiments

Let \(y\) denote a ground-truth line and \(\hat{y}\) a model prediction. Both strings are normalized with the project NFC repair pipeline before CER and WER. For DER, strings are decomposed to NFD and filtered to combining marks relevant to Yorùbá tone and subdot fidelity.

## Benchmark Rows

The active test-split comparison is generated from four possible rows:

1. Base PaddleOCR English-pretrained recognition with the Yorùbá dictionary.
2. PaddleOCR-VL-1.6 zero-shot.
3. GLM-OCR zero-shot.
4. PaddleOCR-VL-1.6 SFT, when a supervised VLM run is completed.

Optional PaddleOCR recognition fine-tuning can be reported as a classical supervised comparison only if phase 04 is run and its checkpoint passes provenance checks.

## Analysis Scripts

After inference, the pipeline runs:

- `scripts/compile_results.py` for Table 1.
- `scripts/analyze_stratified_errors.py` for minimal-pair and density diagnostics.
- `scripts/ablate_der_universe.py` for alternative DER universes.
- `scripts/bootstrap_metric_cis.py` for confidence intervals.

All analysis outputs are invalidated by any change to `data/processed`, `metrics.csv`, or the per-sample JSONL logs.
