# Experiment Model Matrix

This project uses one frozen `data/processed` split and four active model rows.

| Model key | Script | Role | Why it is included |
| --- | --- | --- | --- |
| `paddleocr_en_pretrained` | `scripts/evaluate_paddleocr_en_pretrained.py` via `phase_05_eval_paddleocr_recognition.sh` | Base OCR control | Measures how a standard English-pretrained PaddleOCR recognizer behaves on Yorùbá while loading its matching English CTC head. This is the lowest-friction baseline and exposes out-of-domain diacritic failure. |
| `paddleocrvl16_zero_shot` | `scripts/eval_paddleocrvl16.py` via `phase_15_eval_paddleocrvl16_zero_shot.sh` | Zero-shot VLM OCR | Tests whether PaddleOCR-VL-1.6 can transcribe Yorùbá line crops without task-specific training. Generation is deterministic and prompt-fixed. |
| `glm_ocr_zero_shot` | `scripts/eval_glm_ocr.py` via `phase_18_eval_glm_ocr_zero_shot.sh` | Zero-shot VLM OCR | Provides a second OCR-oriented VLM baseline with a different model family, also deterministic and prompt-fixed. |
| `paddleocrvl16_sft` | `scripts/train_paddleocrvl16_sft.py` and `scripts/eval_paddleocrvl16.py` via phases 14, 16, 17 | Supervised adapted VLM | Uses assistant-only OCR loss. The default T4/L4 configuration trains `lm_head`; `non_vision` is a broader language-side adaptation and `all` includes the vision tower. The selected scope is recorded and resume rejects scope changes. This row is never zero-shot. |

Optional classical comparison:

| Model key | Script | Role |
| --- | --- | --- |
| `paddleocr_recognition_finetuned` | `scripts/train_paddleocr_recognition.py` + `scripts/evaluate_paddleocr_recognition.py` | Optional supervised PaddleOCR recognition fine-tune. Use this when comparing VLM SFT against a classical recognizer trained on the same labels. |

Important distinctions:

- **Zero-shot** means the model is not trained on this dataset. In this repo, that applies to `paddleocrvl16_zero_shot` and `glm_ocr_zero_shot`.
- **SFT** means supervised fine-tuning. `paddleocrvl16_sft` learns from `data/paddleocrvl16_sft/train.jsonl` and must be evaluated separately.
- **Base PaddleOCR EN** is not a Yorùbá-trained model. It stays in the table because it is the simplest reproducible OCR control and shows the cost of using an English recognizer on tone-marked Yorùbá. It must be run through `scripts/evaluate_paddleocr_en_pretrained.py` so the English checkpoint is not accidentally paired with a Yorùbá-sized CTC head.
