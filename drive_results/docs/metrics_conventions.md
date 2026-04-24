# Metric conventions for Yorùbá OCR

## Stored values (`results/tables/metrics.csv`)

### Rates (CER, WER, DER)

- **CER, WER** are **rates** (not capped at 1): normalised edit distance divided
  by ground-truth length (characters, words). They are macro-averaged per sample
  (each line weighted equally). Values **greater than 1** are valid when
  insertions or severe misalignment inflate edit distance beyond reference
  length — this is normal for diacritic-heavy low-resource OCR on an
  out-of-domain model (e.g. Tesseract `eng`, English-only PP-OCRv3).

- **DER** (Diacritic Error Rate) is a **corpus-level, conditional** metric:
  - Strings are NFD-decomposed to isolate combining diacritics (U+0300 grave,
    U+0301 acute, U+0323 combining dot below — the three diacritics that carry
    Yorùbá tone and vowel quality).
  - Let `E_i` be the edit distance between predicted and reference diacritic
    sequences for sample `i`, and `G_i` the number of GT diacritics. DER is
    computed over the **subset** `S = {i : G_i > 0}` as
    `der = Σ_{i∈S} E_i  /  Σ_{i∈S} G_i` (micro-average).
  - Samples with `G_i = 0` (ground truth contains no diacritics) are excluded
    from `der`. A recall-style ratio is undefined for them, and including them
    as `0.0` / `1.0` floors in a macro mean badly distorts the number.

### Auxiliary DER columns

- **`der_n`** — number of samples with at least one GT diacritic that
  contributed to `der`. When `der_n = 0` the split has no diacritic-bearing
  ground truth; `der` is left blank.
- **`der_insertion_rate`** — predicted diacritics per GT character, computed
  **only** over samples whose GT has zero diacritics. This is a
  precision-side signal: how often the model hallucinates tone marks on plain
  text. Blank when every sample has diacritics.

### Phantom flag

- **`phantom`** — checkpoint-integrity flag from the evaluation script:
  - `true`  — head weights were **not** restored (silent random head); the
    row does not measure a trained model.
  - `false` — integrity check passed.
  - `n/a`   — non-Paddle baseline (Tesseract, Qwen 2.5-VL, PaddleOCR-VL 1.5);
    the phantom concept does not apply.
  - `unknown` — provenance did not include an integrity report (legacy rows).

## Compiled tables (`table1_main_comparison.csv`, `metrics_summary.csv`)

- Columns `cer_pct`, `wer_pct`, `der_pct` are **rate × 100** with one decimal,
  for readability. They are **not** guaranteed to lie in 0–100; treat them as
  "scaled rates" or report **raw decimals** from `metrics.csv` in the paper if
  the venue expects classic 0–1 CER.
- When reporting DER in the paper always cite `der_n` and, when meaningful,
  `der_insertion_rate` alongside it. A DER without a denominator count is
  uninterpretable.

## Paper text

- Define the formula once (NFC-normalised strings for CER/WER; NFD combining
  marks for DER; conditional corpus-level aggregation).
- Prefer **relative** comparisons (fine-tuned vs baseline) and **DER** for
  diacritic fidelity.
- Frame high absolute CER/WER on off-the-shelf baselines as **expected** on
  diacritic-heavy low-resource line OCR, not as a benchmark defect.

## Alignment check

After changing `data/processed` or before submission, run:

```bash
bash scripts/shell/phase_13_verify_eval.sh
```

Inspect `results/tables/eval_alignment_report.json`. If `mismatches` is
non-empty, **re-run** evaluations or freeze the dataset version referenced in
the paper. Then run the checkpoint audit:

```bash
python scripts/12_diagnose_hypotheses.py checkpoints --fail-on-phantom
```

which scans `metrics.csv` + sibling `meta.json` files and exits non-zero if
any row is flagged `phantom="true"` or is missing provenance.
