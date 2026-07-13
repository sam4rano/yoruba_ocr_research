# Results and Analysis

This section is regenerated after the active benchmark is rerun. The authoritative inputs are:

- `results/tables/metrics.csv`
- `results/tables/*_test.jsonl`
- `results/tables/meta/*.json`
- `results/tables/table1_main_comparison.md`
- `results/tables/bootstrap_metric_cis.csv`
- `results/tables/stratified_der_by_density.csv`
- `results/tables/error_taxonomy.csv`

## Main Comparison

Table 1 is produced by `scripts/compile_results.py` from fresh test-split rows. Rows with `phantom=true`, missing provenance, stale checkpoints without JSONL evidence, or mismatched sample counts are not citable.

Expected active rows:

| Model key | Status source |
| --- | --- |
| `paddleocr_en_pretrained` | `scripts/evaluate_paddleocr_recognition.py` |
| `paddleocrvl16_zero_shot` | `scripts/eval_paddleocrvl16.py` |
| `glm_ocr_zero_shot` | `scripts/eval_glm_ocr.py` |
| `paddleocrvl16_sft` | optional SFT via phases 14, 16, 17 |

## Stratified Analysis

After all expected JSONL logs exist, `scripts/analyze_stratified_errors.py` computes minimal-pair, diacritic-density, book/source, and error-taxonomy diagnostics. These diagnostics should be interpreted as secondary analyses, not substitutes for the main frozen-split comparison.

## Uncertainty

`scripts/bootstrap_metric_cis.py` computes line-level bootstrap confidence intervals and pairwise comparisons. Bootstrap outputs are invalidated whenever the split, labels, model rows, or JSONL logs change.
