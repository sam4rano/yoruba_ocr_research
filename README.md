# Yorùbá OCR Research

Repository for experiments, analysis, and paper writing for Yorùbá OCR.

## Layout

- `data/`: raw inputs, processed artifacts, and split files
- `experiments/`: checkpoints — PP-OCRv4 fine-tune / ablations (all **gitignored** except `.gitkeep`; regenerate via training scripts)
- `paper/`: paper sections, figures, and bibliography
- `scripts/`: preprocessing, training, and evaluation for the zero-shot baselines and PP-OCRv4 CRNN ablations
- `results/tables/`: traceable metrics (`metrics.csv`, compiled tables, per-run `*.jsonl`, `meta/`)
- `results/tables/archive/`: historical metric snapshots (do not cite in the paper)
- `scripts/shell/`: phased bash runners (`run_all.sh`, Drive backup) — see `scripts/shell/README.md`
- `.cursorrules` and `.cursor/rules/`: Cursor project and context rules
