# Research Outputs for Paper Submission

This project should not cite notebook screenshots or illustrative plots. Paper
tables and figures must be regenerated from real `results/tables` artifacts
after the final evaluation run.

## Required comparison outputs

| Output | Source command | Purpose |
| --- | --- | --- |
| `results/tables/table1_main_comparison.csv` | `python scripts/compile_results.py` | Machine-readable main comparison table. |
| `results/tables/table1_main_comparison.md` | `python scripts/compile_results.py` | Copy-ready paper table. |
| `results/tables/metrics_summary.csv` | `python scripts/compile_results.py` | Plotting alias for the main comparison table. |
| `results/tables/figures/model_metrics_comparison.{png,pdf,svg}` | `python scripts/generate_plots.py` | CER/WER/DER grouped comparison. |
| `results/tables/figures/relative_error_reduction.{png,pdf,svg}` | `python scripts/generate_plots.py` | Relative improvement over English PP-OCR baseline. |
| `results/tables/figures/bootstrap_confidence_intervals.{png,pdf,svg}` | `python scripts/bootstrap_metric_cis.py` then `python scripts/generate_plots.py` | Metric uncertainty. |
| `results/tables/figures/stratified_der_by_density.{png,pdf,svg}` | `python scripts/analyze_stratified_errors.py` then `python scripts/generate_plots.py` | DER by diacritic-density quartile. |

The plotting script writes 300-DPI PNG files and editable vector files (`.pdf`
and `.svg`) for conference paper workflows.

## Final paper-output sequence

Run this only after the model rows have been generated in
`results/tables/metrics.csv`:

```bash
python scripts/compile_results.py
python scripts/analyze_stratified_errors.py
python scripts/bootstrap_metric_cis.py
python scripts/generate_plots.py
```

Do not use placeholder figures for paper claims:

```bash
python scripts/generate_plots.py --allow-placeholder
```

That flag is only for layout mockups while writing slides or drafts.

## Research-grade checks

Before submission, verify:

- Every cited model has a `test` row in `results/tables/metrics.csv`.
- Every cited model has a matching `results/tables/*_test.jsonl` prediction log.
- Every cited model has a provenance file under `results/tables/meta/`.
- Figures are generated without `--allow-placeholder`.
- Paper text, LaTeX tables, Markdown tables, and CSV values match exactly.
- The final run is tied to a clean git commit in `docs/reproducibility_manifest.md`.

