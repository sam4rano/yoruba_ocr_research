# Metric conventions for Yorùbá OCR

## Stored values (`results/tables/metrics.csv`)

- **CER, WER, DER** are **rates** (not capped at 1):  
  normalised edit distance divided by ground-truth length (characters, words, or diacritic sequence length).  
  Values **greater than 1** are valid when insertions or severe misalignment inflate edit distance beyond reference length.

## Compiled tables (`table1_main_comparison.csv`, `metrics_summary.csv`)

- Columns `cer_pct`, `wer_pct`, `der_pct` are **rate × 100** with one decimal, for readability.  
  They are **not** guaranteed to lie in 0–100; treat them as “scaled rates” or report **raw decimals** from `metrics.csv` in the paper if the venue expects classic 0–1 CER.

## Paper text

- Define the formula once (NFC-normalised strings; DER on NFD combining marks).  
- Prefer **relative** comparisons (fine-tuned vs baseline) and **DER** for diacritic fidelity.  
- Frame high absolute CER/WER on off-the-shelf baselines as **expected** on diacritic-heavy low-resource line OCR, not as a benchmark defect.

## Alignment check

After changing `data/processed` or before submission, run:

```bash
bash scripts/shell/phase_13_verify_eval.sh
```

Inspect `results/tables/eval_alignment_report.json`. If `mismatches` is non-empty, **re-run** evaluations or freeze the dataset version referenced in the paper.
