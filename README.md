# Yorùbá OCR Research

Repository for experiments, analysis, and paper writing for Yorùbá OCR.

The project studies line-level OCR for tone-marked Yorùbá text. The central
question is not only "which model reads the line correctly?" but also "which
model preserves the diacritics that carry tone and vowel quality?" For that
reason the benchmark reports standard OCR metrics, CER and WER, plus DER
(Diacritic Error Rate).

## Current Experiment Plan

The active experiment uses one frozen `data/processed` split and separates
three kinds of evidence: baselines, supervised fine-tuning, and ablations.

### 1. Baselines

Baselines measure what works before training on this dataset.

| Model key | How it runs | Purpose |
| --- | --- | --- |
| `paddleocr_en_pretrained` | `scripts/evaluate_paddleocr_en_pretrained.py` via `scripts/shell/phase_05_eval_paddleocr_recognition.sh` | Classical OCR control. Evaluates English-pretrained PP-OCR on Yorùbá while loading its matching English CTC head, so the row is not a random-head phantom. |
| `paddleocrvl16_zero_shot` | `scripts/eval_paddleocrvl16.py` via `scripts/shell/phase_15_eval_paddleocrvl16_zero_shot.sh` | Zero-shot OCR-oriented VLM baseline using PaddleOCR-VL-1.6. |
| `glm_ocr_zero_shot` | `scripts/eval_glm_ocr.py` via `scripts/shell/phase_18_eval_glm_ocr_zero_shot.sh` | Second zero-shot VLM baseline from a different model family. |

The zero-shot rows are prompt-fixed and deterministic. They should not be
described as fine-tuned models.

### 2. Fine-Tuned Models

The main supervised adaptation is:

| Model key | How it runs | Purpose |
| --- | --- | --- |
| `paddleocrvl16_sft` | Export with `scripts/export_paddleocrvl16_sft.py`, train with `scripts/train_paddleocrvl16_sft.py`, evaluate with `scripts/eval_paddleocrvl16.py` | Uses assistant-only supervised OCR loss. The default `lm_head` scope adapts only the output head for T4/L4 memory safety; `non_vision` adapts the language-side parameters on larger GPUs, while `all` also updates the vision tower and is intentionally discouraged for this small dataset. Each scope is a distinct experiment and must not share a resumed checkpoint. |

Optional classical comparison:

| Model key | How it runs | Purpose |
| --- | --- | --- |
| `paddleocr_recognition_finetuned` | `scripts/train_paddleocr_recognition.py` plus `scripts/evaluate_paddleocr_recognition.py` | Fine-tunes the traditional PaddleOCR recognition model. This is useful when comparing VLM SFT against a conventional recognizer trained on the same labels, but it is not part of the default paper run unless explicitly enabled. |

### 3. Ablations and Robustness Checks

Ablations do not introduce extra headline model rows unless they are rerun and
logged through the current metrics schema. The active ablation layer is about
metric and error robustness:

- `scripts/ablate_der_universe.py` recomputes DER under alternative
  diacritic universes, such as tone-only marks, all combining marks, and marked
  grapheme tokens. This checks whether the DER conclusion depends on one narrow
  definition of "diacritic."
- `scripts/bootstrap_metric_cis.py` computes line-level bootstrap confidence
  intervals and aligned pairwise comparisons.
- `scripts/analyze_stratified_errors.py` breaks errors down by linguistic
  features such as diacritic density and book/source groups.
- `scripts/diagnose_experiment.py` checks setup, checkpoint, and evaluation
  failure modes, including phantom PaddleOCR heads.

Older pilot rows such as PaddleOCR-VL-1.5 LoRA, Qwen, TrOCR, or stale ablation
claims are not part of the active comparison unless they are regenerated with
the current scripts and written into `results/tables/metrics.csv`.

## Metrics and Outputs

All citable results must trace back to:

- `results/tables/metrics.csv` — append-only model metrics; reset before a fresh
  paper run with `python scripts/metrics_lifecycle.py reset`
- `results/tables/*_test.jsonl` — per-line predictions for analysis scripts
- `results/tables/meta/*.json` — provenance and checkpoint integrity metadata
- `results/tables/table1_main_comparison.csv` / `.md` — compiled paper table
- `results/tables/figures/` — regenerated plots from real CSV/JSONL inputs

`scripts/compile_results.py` excludes rows marked `phantom=true` or stale.
Placeholder plots are disabled by default; use `--allow-placeholder` only for
layout mockups, never for paper claims.

See also:

- `docs/model_matrix.md` for exact model-row definitions
- `docs/metrics_conventions.md` for CER, WER, DER, and ablation definitions
- `docs/colab_pro_t4.md` for the Colab Pro/T4 execution guide
- `docs/kaggle_gpu.md` for Kaggle GPU notebooks with `/kaggle/input` data
- `docs/lightning_ai.md` for Lightning AI Studio/compute notebooks
- `scripts/shell/README.md` for phase-by-phase orchestration

## Quick Run Order

From a clean workspace:

```bash
python scripts/metrics_lifecycle.py reset
bash scripts/shell/run_all.sh
```

`run_all.sh` intentionally runs the core CPU/Paddle-safe path only. The VLM and
SFT phases require GPU, Hugging Face downloads, and modern `transformers`.

Run zero-shot VLM baselines:

```bash
bash scripts/shell/phase_15_eval_paddleocrvl16_zero_shot.sh
bash scripts/shell/phase_18_eval_glm_ocr_zero_shot.sh
bash scripts/shell/phase_09_compile.sh
```

Run PaddleOCR-VL-1.6 SFT:

```bash
bash scripts/shell/phase_14_export_paddleocrvl16_sft.sh
bash scripts/shell/phase_16_train_paddleocrvl16_sft.sh
bash scripts/shell/phase_17_eval_paddleocrvl16_sft.sh
bash scripts/shell/phase_09_compile.sh
```

Run analysis and ablations after model JSONL logs exist:

```bash
python scripts/analyze_stratified_errors.py
python scripts/ablate_der_universe.py
python scripts/bootstrap_metric_cis.py
python scripts/generate_plots.py
```

## Layout

- `data/`: raw inputs, processed artifacts, and split files
- `experiments/`: checkpoints — optional PaddleOCR recognition fine-tune and PaddleOCR-VL-1.6 SFT outputs (all **gitignored** except `.gitkeep`; regenerate via training scripts)
- `paper/`: paper sections, figures, and bibliography
- `FormattingGuidelines-IJCAI-ECAI-26/`: IJCAI style and bibliography assets; the manuscript is regenerated only after final metrics exist
- `scripts/`: preprocessing, optional training, and evaluation for PaddleOCR, PaddleOCR-VL-1.6, and GLM-OCR
- `notebooks/`: platform-specific notebook runners for Kaggle and Lightning AI
- `results/tables/`: traceable metrics (`metrics.csv`, compiled tables, per-run `*.jsonl`, `meta/`)
- `results/tables/archive/`: historical metric snapshots (do not cite in the paper)
- `scripts/shell/`: phased bash runners (`run_all.sh`, Drive backup) — see `scripts/shell/README.md`
- `.cursorrules` and `.cursor/rules/`: Cursor project and context rules

## Naming Convention

- `scripts/shell/phase_<id>_<action>.sh` owns execution order and numeric phase IDs.
- `scripts/<action>.py` uses descriptive names without phase numbers so one utility can be reused safely by notebooks and multiple phases.
- `notebooks/colab_ocr.ipynb`, `notebooks/kaggle_ocr.ipynb`, and `notebooks/lightning_ocr.ipynb` are the three canonical platform runners.
- Evaluation JSONL files use the stable model keys from `docs/model_matrix.md`.
