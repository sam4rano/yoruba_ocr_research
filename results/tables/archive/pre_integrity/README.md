# Pre-integrity metrics archive

This folder holds the `metrics.csv` that existed before the checkpoint-integrity
patch (P1) was merged.

**Do not cite these numbers.** Every one of the 20 rows failed the
`scripts/12_diagnose_hypotheses.py checkpoints` audit with status
`NO_META`, and the forensic analysis already established that the trained
PaddleOCR checkpoints they were supposedly computed against did not exist
on disk at evaluation time (`experiments/finetuned/` and
`experiments/baseline/pretrained/` were empty). The CER/WER/DER values in
`metrics.csv` therefore measured a randomly-initialised CTC head on top of
an English encoder, which explains why "fine-tuned" and "baseline" rows
differed by less than a percentage point.

- `metrics.csv` — the original aggregate table, archived verbatim.
- `checkpoint_audit.json` — the audit that established phantom status.

New, trustworthy rows should be written to `results/tables/metrics.csv` by
the hardened eval pipeline, which emits a sibling `results/tables/meta/*.json`
for every row and asserts CTC-head restoration before running inference.
