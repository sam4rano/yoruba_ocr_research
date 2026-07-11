# -*- coding: utf-8 -*-
"""Colab-ready Kenya ASR fine-tuning/evaluation notebook script.

This is a cleaned replacement for the old `luhya_new_asr.py` Colab export.

Primary path:
    - Fine-tune Whisper-family seq2seq ASR models on DDD-Kenya datasets.

Comparable baseline path:
    - Run Qwen3-ASR as a zero-shot ASR evaluator. Qwen3-ASR is a dedicated
      ASR inference model exposed through the `qwen-asr` package, not a
      drop-in Seq2SeqTrainer replacement for Whisper fine-tuning.

Suggested Colab use:
    1. Upload/open this file as a notebook or paste cells into Colab.
    2. Set DATASET_NAME to Gusii or Kamba.
    3. Start with TRAIN_HOURS=2.0 to verify the pipeline.
    4. Increase TRAIN_HOURS only after the smoke run is clean.
"""

# ============================================================================
# Cell 1 — Install packages
# ============================================================================
# In Colab, run this cell first. Restart runtime if transformers/datasets were
# already imported before the install.
try:
    import google.colab  # type: ignore  # noqa: F401
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if IN_COLAB:
    # Qwen3-ASR and newer Trainer APIs benefit from a recent Transformers stack.
    # Restart the runtime after this cell if Transformers/Datasets were imported earlier.
    !pip install -q -U "transformers>=4.51.0" "datasets>=2.20.0" "evaluate>=0.4.2" jiwer accelerate librosa soundfile huggingface_hub
    # Optional, used for Qwen3-ASR comparable baseline and 4-bit/8-bit experiments.
    !pip install -q -U qwen-asr
    !pip install -q -U bitsandbytes peft


# ============================================================================
# Cell 2 — Imports and configuration
# ============================================================================
import csv
import gc
import inspect
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import evaluate
import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, IterableDataset, load_dataset
from huggingface_hub import HfApi, model_info
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


# ------------------------------
# Dataset choices
# ------------------------------
DATASET_NAME = "DDD-Kenya/Gusii-ASR-Data-Subset-470H"
# DATASET_NAME = "DDD-Kenya/Kamba-ASR-Data-Subset-484H"

DATASET_CONFIG = None
DATASET_REVISION = None
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
TEST_SPLIT = "test"

# Column auto-detection handles common names. Override only if needed.
AUDIO_COLUMN = None
TEXT_COLUMN = None

# Use streaming for 470h/484h datasets to avoid loading the whole corpus.
STREAMING = True
RANDOM_SEED = 42
TARGET_SR = 16_000

# Start small in Colab. Increase progressively after the smoke run.
TRAIN_HOURS = 2.0
VAL_HOURS = 0.25
TEST_HOURS = 0.25
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None
MAX_TEST_SAMPLES = None
SHUFFLE_BUFFER = 2_000

# Optional metadata filters. Leave empty unless the dataset exposes these fields.
FILTERS: dict[str, Any] = {
    # "dialect": "Wanga",
    # "speaker_id": "speaker_001",
}

# ------------------------------
# Model choices
# ------------------------------
ASR_BACKEND = "whisper_finetune"
# ASR_BACKEND = "qwen3_asr_eval"

MODEL_ID = "openai/whisper-small"
# Comparable Whisper-family choices:
# MODEL_ID = "openai/whisper-base"
# MODEL_ID = "openai/whisper-medium"
# MODEL_ID = "openai/whisper-large-v3-turbo"
# MODEL_ID = "distil-whisper/distil-large-v3"

# Qwen3-ASR comparable baseline choices:
QWEN_ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
# QWEN_ASR_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"  # faster/lighter baseline.
QWEN_ASR_LANGUAGE = None  # e.g. "English"; None lets Qwen3-ASR detect language.
QWEN_ASR_DTYPE = "auto"  # auto -> bf16 on supported GPUs, fp16 on T4/V100, fp32 on CPU.
QWEN_ASR_MAX_NEW_TOKENS = 256
QWEN_ASR_MAX_BATCH_SIZE = 4

# Whisper does not have Gusii/Kamba language tokens. `swahili` is a pragmatic proxy.
# Set WHISPER_LANGUAGE = None if you do not want a forced language prompt.
WHISPER_LANGUAGE = "swahili"
WHISPER_TASK = "transcribe"

# ------------------------------
# Training choices
# ------------------------------
OUTPUT_ROOT = Path("/content/drive/MyDrive/kenya_asr_runs") if IN_COLAB else Path("kenya_asr_runs")
RUN_NAME = f"{DATASET_NAME.split('/')[-1]}__{MODEL_ID.split('/')[-1]}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 1e-5
WARMUP_STEPS = 50
MAX_STEPS = 200
NUM_TRAIN_EPOCHS = None
EVAL_STEPS = 50
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 2
GENERATION_MAX_LENGTH = 225
FP16 = torch.cuda.is_available()
GRADIENT_CHECKPOINTING = True
FREEZE_ENCODER = False
RESUME_FROM_CHECKPOINT = True

PUSH_TO_HUB = False
HF_USERNAME = "Sam4rano"
HF_REPO_ID = None  # If None, derived from HF_USERNAME + dataset/model/backend.


# ============================================================================
# Cell 3 — Runtime setup
# ============================================================================
def setup_runtime() -> None:
    """Mount Drive in Colab and configure deterministic-ish execution."""
    if IN_COLAB:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
        if not os.environ.get("HF_TOKEN"):
            try:
                from google.colab import userdata  # type: ignore
                token = userdata.get("HF_TOKEN")
                if token:
                    os.environ["HF_TOKEN"] = token
                    print("Loaded HF_TOKEN from Colab secrets.")
            except Exception:
                print("HF_TOKEN not found in Colab secrets; Hub push requires login/token.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    print("IN_COLAB:", IN_COLAB)
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("OUTPUT_DIR:", OUTPUT_DIR)


setup_runtime()


def slugify(value: str) -> str:
    """Create a conservative Hugging Face repo/file slug."""
    value = value.strip().replace("/", "-")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "kenya-asr-run"


def resolved_hf_repo_id() -> str:
    """Return the destination HF model repo id."""
    if HF_REPO_ID:
        return HF_REPO_ID
    dataset_slug = slugify(DATASET_NAME.split("/")[-1])
    model_slug = slugify((MODEL_ID if ASR_BACKEND == "whisper_finetune" else QWEN_ASR_MODEL_ID).split("/")[-1])
    suffix = "finetuned" if ASR_BACKEND == "whisper_finetune" else "eval"
    return f"{HF_USERNAME}/{dataset_slug}-{model_slug}-{suffix}"


# ============================================================================
# Cell 4 — Dataset helpers
# ============================================================================
TEXT_CANDIDATES = (
    "transcript",
    "text",
    "sentence",
    "translation",
    "normalized_text",
    "raw_transcription",
)
AUDIO_CANDIDATES = ("audio", "file", "path")


def normalize_text(text: Any) -> str:
    """Normalize transcript text for ASR training/evaluation."""
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_column(example: dict[str, Any], override: str | None, candidates: tuple[str, ...], kind: str) -> str:
    """Detect a likely audio/text column from one dataset example."""
    if override:
        if override not in example:
            raise KeyError(f"Configured {kind} column '{override}' not found. Available: {sorted(example)}")
        return override
    for name in candidates:
        if name in example:
            return name
    raise KeyError(f"Could not detect {kind} column. Available columns: {sorted(example)}")


def passes_filters(example: dict[str, Any]) -> bool:
    """Apply exact-match metadata filters."""
    for key, expected in FILTERS.items():
        if expected is None:
            continue
        if example.get(key) != expected:
            return False
    return True


def get_duration_seconds(example: dict[str, Any], audio_column: str) -> float | None:
    """Estimate duration using metadata first, then audio shape if needed."""
    for key in ("duration", "duration_seconds", "length", "seconds"):
        if key in example and example[key] is not None:
            try:
                value = float(example[key])
                if value > 0:
                    return value
            except Exception:
                pass

    audio = example.get(audio_column)
    if isinstance(audio, dict):
        for key in ("duration", "duration_seconds"):
            if key in audio and audio[key] is not None:
                try:
                    value = float(audio[key])
                    if value > 0:
                        return value
                except Exception:
                    pass
        if "array" in audio and "sampling_rate" in audio and audio["array"] is not None:
            return float(len(audio["array"]) / audio["sampling_rate"])
        if "num_frames" in audio and "sampling_rate" in audio:
            return float(audio["num_frames"] / audio["sampling_rate"])
        if audio.get("bytes") is not None:
            try:
                return float(sf.info(BytesIO(audio["bytes"])).duration)
            except Exception:
                pass
        if audio.get("path"):
            try:
                return float(librosa.get_duration(path=audio["path"]))
            except Exception:
                pass
    return None


def audio_to_array(audio: Any, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load an audio example into a mono float32 array."""
    if isinstance(audio, dict):
        if audio.get("array") is not None:
            array = np.asarray(audio["array"], dtype=np.float32)
            if array.ndim > 1:
                array = np.mean(array, axis=1)
            sr = int(audio.get("sampling_rate") or target_sr)
            if sr != target_sr:
                array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
            return array.astype(np.float32)
        if audio.get("bytes") is not None:
            array, sr = sf.read(BytesIO(audio["bytes"]), dtype="float32")
            if array.ndim > 1:
                array = np.mean(array, axis=1)
            if sr != target_sr:
                array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
            return np.asarray(array, dtype=np.float32)
        if audio.get("path"):
            path = audio["path"]
            array, _ = librosa.load(path, sr=target_sr, mono=True)
            return np.asarray(array, dtype=np.float32)
    if isinstance(audio, str):
        array, _ = librosa.load(audio, sr=target_sr, mono=True)
        return np.asarray(array, dtype=np.float32)
    raise ValueError(f"Unsupported audio payload: {type(audio)}")


def load_split(split: str):
    """Load one HF dataset split."""
    kwargs = {
        "path": DATASET_NAME,
        "split": split,
        "streaming": STREAMING,
    }
    if DATASET_CONFIG:
        kwargs["name"] = DATASET_CONFIG
    if DATASET_REVISION:
        kwargs["revision"] = DATASET_REVISION
    return load_dataset(**kwargs)


def first_example(dataset) -> dict[str, Any]:
    """Read one example without assuming map-style indexing."""
    if isinstance(dataset, IterableDataset):
        return next(iter(dataset))
    return dataset[0]


def collect_subset(
    split: str,
    *,
    target_hours: float | None,
    max_samples: int | None,
    seed: int,
) -> tuple[Dataset, dict[str, Any]]:
    """Collect a bounded subset from a split, using streaming when configured."""
    ds = load_split(split)
    if STREAMING and hasattr(ds, "shuffle"):
        ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER, seed=seed)

    probe = first_example(ds)
    audio_column = detect_column(probe, AUDIO_COLUMN, AUDIO_CANDIDATES, "audio")
    text_column = detect_column(probe, TEXT_COLUMN, TEXT_CANDIDATES, "text")

    # Re-open after probing so streaming datasets are not missing the first row.
    ds = load_split(split)
    if STREAMING and hasattr(ds, "shuffle"):
        ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER, seed=seed)

    target_seconds = None if target_hours is None else int(target_hours * 3600)
    rows: list[dict[str, Any]] = []
    total_seconds = 0.0
    seen = 0
    skipped = 0

    iterator = iter(ds) if STREAMING else iter(ds)
    for example in iterator:
        seen += 1
        if not passes_filters(example):
            continue
        text = normalize_text(example.get(text_column))
        if not text:
            skipped += 1
            continue
        duration = get_duration_seconds(example, audio_column)
        clean = {
            "audio": example[audio_column],
            "transcript": text,
            "duration_s": duration if duration is not None else None,
        }
        rows.append(clean)
        if duration:
            total_seconds += float(duration)
        if max_samples and len(rows) >= max_samples:
            break
        if target_seconds and total_seconds >= target_seconds:
            break
        if seen % 500 == 0:
            print(f"{split}: seen={seen:,} selected={len(rows):,} hours≈{total_seconds/3600:.2f}")

    if not rows:
        raise ValueError(f"No usable rows selected from split={split}. Check filters/columns.")

    info = {
        "split": split,
        "seen": seen,
        "selected": len(rows),
        "skipped_empty_text": skipped,
        "hours_estimate": total_seconds / 3600 if total_seconds else None,
        "audio_column": audio_column,
        "text_column": text_column,
    }
    return Dataset.from_list(rows), info


print("Collecting dataset subsets...")
train_raw, train_info = collect_subset(TRAIN_SPLIT, target_hours=TRAIN_HOURS, max_samples=MAX_TRAIN_SAMPLES, seed=RANDOM_SEED)
val_raw, val_info = collect_subset(VAL_SPLIT, target_hours=VAL_HOURS, max_samples=MAX_VAL_SAMPLES, seed=RANDOM_SEED + 1)
test_raw, test_info = collect_subset(TEST_SPLIT, target_hours=TEST_HOURS, max_samples=MAX_TEST_SAMPLES, seed=RANDOM_SEED + 2)

print(json.dumps({"train": train_info, "validation": val_info, "test": test_info}, indent=2))


# ============================================================================
# Cell 5 — Whisper fine-tuning
# ============================================================================
def get_model_revision(model_id: str) -> str | None:
    """Resolve a HF model revision when online/authenticated."""
    try:
        return model_info(model_id).sha
    except Exception as exc:
        print(f"Model revision lookup skipped for {model_id}: {exc}")
        return None


def save_run_manifest(extra: dict[str, Any] | None = None) -> None:
    """Write a reproducibility manifest for this run."""
    active_model_id = MODEL_ID if ASR_BACKEND == "whisper_finetune" else QWEN_ASR_MODEL_ID
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "backend": ASR_BACKEND,
        "model_id": active_model_id,
        "model_revision": get_model_revision(active_model_id),
        "hf_repo_id": resolved_hf_repo_id(),
        "random_seed": RANDOM_SEED,
        "target_sr": TARGET_SR,
        "filters": FILTERS,
        "split_info": {"train": train_info, "validation": val_info, "test": test_info},
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "output_dir": str(OUTPUT_DIR),
    }
    if extra:
        manifest.update(extra)
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def format_hours(value: Any) -> str:
    """Format optional hour estimates."""
    if value is None:
        return "unknown"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def write_model_card(metrics: dict[str, Any] | None = None, *, baseline_only: bool = False) -> Path:
    """Write a Hugging Face model card into OUTPUT_DIR/README.md."""
    metrics = metrics or {}
    active_model_id = MODEL_ID if ASR_BACKEND == "whisper_finetune" else QWEN_ASR_MODEL_ID
    repo_id = resolved_hf_repo_id()
    dataset_slug = DATASET_NAME.split("/")[-1]
    model_name = repo_id.split("/")[-1]
    pipeline_tag = "automatic-speech-recognition"
    license_name = "apache-2.0"
    tags = [
        "automatic-speech-recognition",
        "asr",
        "speech",
        "kenya",
        "low-resource-asr",
        "whisper" if ASR_BACKEND == "whisper_finetune" else "qwen3-asr",
    ]
    wer = metrics.get("test_wer") or metrics.get("eval_wer") or metrics.get("test_loss")
    wer_line = f"- Test WER: `{wer}`\n" if wer is not None else "- Test WER: not recorded yet\n"
    training_statement = (
        "This repository contains evaluation artifacts for a zero-shot Qwen3-ASR baseline; "
        "the base Qwen3-ASR weights are not fine-tuned by this notebook."
        if baseline_only
        else (
            "This repository contains a fine-tuned Whisper-family ASR checkpoint trained "
            "on a bounded subset of the selected DDD-Kenya dataset."
        )
    )
    model_card = f"""---
language:
- sw
- en
license: {license_name}
tags:
{chr(10).join(f"- {tag}" for tag in tags)}
datasets:
- {DATASET_NAME}
base_model:
- {active_model_id}
pipeline_tag: {pipeline_tag}
---

# {model_name}

{training_statement}

## Model Details

- **Owner:** [{HF_USERNAME}](https://huggingface.co/{HF_USERNAME})
- **Base model:** `{active_model_id}`
- **Backend:** `{ASR_BACKEND}`
- **Dataset:** `{DATASET_NAME}`
- **Output repository:** `{repo_id}`
- **Sampling rate:** `{TARGET_SR}` Hz
- **Created:** {datetime.now(timezone.utc).date().isoformat()}

## Intended Use

This model/artifact is intended for automatic speech recognition experiments on
Kenyan language speech datasets from DDD-Kenya, especially Gusii and Kamba
subset experiments. It is suitable for research evaluation, error analysis, and
continued ASR adaptation. It should not be used as the sole basis for high-stakes
decisions without additional validation by native speakers and domain experts.

## Training and Evaluation Data

The run used bounded, reproducible subsets selected from Hugging Face Datasets.
The exact split metadata is saved in `run_manifest.json`.

| Split | Selected examples | Estimated hours |
| --- | ---: | ---: |
| Train | {train_info.get("selected")} | {format_hours(train_info.get("hours_estimate"))} |
| Validation | {val_info.get("selected")} | {format_hours(val_info.get("hours_estimate"))} |
| Test | {test_info.get("selected")} | {format_hours(test_info.get("hours_estimate"))} |

Filters:

```json
{json.dumps(FILTERS, indent=2)}
```

## Training Procedure

{"No fine-tuning was performed for this Qwen3-ASR baseline. The model was evaluated using the `qwen-asr` package." if baseline_only else f"""The model was fine-tuned with Hugging Face `Seq2SeqTrainer`.

- Learning rate: `{LEARNING_RATE}`
- Per-device train batch size: `{PER_DEVICE_TRAIN_BATCH_SIZE}`
- Gradient accumulation steps: `{GRADIENT_ACCUMULATION_STEPS}`
- Max steps: `{MAX_STEPS}`
- Warmup steps: `{WARMUP_STEPS}`
- FP16: `{FP16}`
- Gradient checkpointing: `{GRADIENT_CHECKPOINTING}`
- Whisper language prompt: `{WHISPER_LANGUAGE}`
- Whisper task: `{WHISPER_TASK}`"""}

## Results

{wer_line}
Additional metrics and raw run metadata are saved alongside the checkpoint:

- `test_metrics.json` or backend-specific metrics JSON
- `run_manifest.json`
- prediction CSV files when generated

## Limitations

- Gusii and Kamba are not native Whisper language-token targets; this run uses
  the configured language prompt as an approximation unless set to `None`.
- Subset-based Colab runs are useful for debugging and early experiments, but
  final claims should use a larger fixed subset or the full benchmark split.
- Orthography, speaker distribution, audio quality, and dialect coverage should
  be manually audited before publication.
- Qwen3-ASR baseline rows are comparable evaluation results, not fine-tuned
  derivative model weights.

## Reproducibility

See `run_manifest.json` for:

- dataset name/config/revision
- base model revision, when resolvable
- split sizes and estimated hours
- Python, PyTorch, CUDA, and GPU metadata
- output directory and repository target

## Citation

If you use this artifact, cite the original base model authors and the DDD-Kenya
dataset source. For Qwen3-ASR baseline experiments, cite the Qwen3-ASR model card
and technical report referenced by the upstream repository.
"""
    card_path = OUTPUT_DIR / "README.md"
    card_path.write_text(model_card, encoding="utf-8")
    return card_path


def maybe_push_to_hub() -> None:
    """Optionally push OUTPUT_DIR contents to the configured HF model repo."""
    if not PUSH_TO_HUB:
        print("PUSH_TO_HUB=False — skipping Hugging Face upload.")
        print("To upload later:")
        print(f"  huggingface-cli upload {resolved_hf_repo_id()} {OUTPUT_DIR}")
        return
    api = HfApi()
    repo_id = resolved_hf_repo_id()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(OUTPUT_DIR),
        commit_message=f"Upload {ASR_BACKEND} ASR run",
    )
    print(f"Pushed to https://huggingface.co/{repo_id}")


wer_metric = evaluate.load("wer")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Whisper collator copied from the HF fine-tuning pattern, with type cleanup."""
    processor: WhisperProcessor
    decoder_start_token_id: int | None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if self.decoder_start_token_id is not None and labels.shape[1] > 0:
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def build_training_args() -> Seq2SeqTrainingArguments:
    """Create TrainingArguments across transformers versions."""
    kwargs = {
        "output_dir": str(OUTPUT_DIR),
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "fp16": FP16,
        "save_strategy": "steps",
        "save_steps": SAVE_STEPS,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "logging_steps": 10,
        "predict_with_generate": True,
        "generation_max_length": GENERATION_MAX_LENGTH,
        "report_to": "none",
        "remove_unused_columns": False,
        # Keep Trainer's built-in Hub push disabled so it does not generate a
        # generic model card over our custom README.md. maybe_push_to_hub()
        # uploads the final folder after metrics/card generation.
        "push_to_hub": False,
        "load_best_model_at_end": False,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
    }
    if MAX_STEPS:
        kwargs["max_steps"] = MAX_STEPS
    elif NUM_TRAIN_EPOCHS:
        kwargs["num_train_epochs"] = NUM_TRAIN_EPOCHS
    else:
        kwargs["num_train_epochs"] = 1

    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[eval_key] = "steps"
    kwargs["eval_steps"] = EVAL_STEPS

    # Newer versions accept this; older ones may not.
    if "save_safetensors" in signature.parameters:
        kwargs["save_safetensors"] = True

    return Seq2SeqTrainingArguments(**kwargs)


def prepare_whisper_dataset(dataset: Dataset, processor: WhisperProcessor) -> Dataset:
    """Decode audio and create Whisper input features/labels."""
    def prepare_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        arrays = [audio_to_array(audio, TARGET_SR) for audio in batch["audio"]]
        inputs = processor.feature_extractor(arrays, sampling_rate=TARGET_SR)
        labels = processor.tokenizer(batch["transcript"], truncation=True, max_length=448)
        return {"input_features": inputs["input_features"], "labels": labels["input_ids"]}

    return dataset.map(
        prepare_batch,
        batched=True,
        batch_size=8,
        remove_columns=dataset.column_names,
        desc="Whisper preprocessing",
    )


def run_whisper_finetune() -> None:
    """Fine-tune a Whisper-family model and evaluate on held-out test subset."""
    processor = WhisperProcessor.from_pretrained(
        MODEL_ID,
        language=WHISPER_LANGUAGE,
        task=WHISPER_TASK,
    )
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    if WHISPER_LANGUAGE:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=WHISPER_LANGUAGE, task=WHISPER_TASK)
        model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.suppress_tokens = []
    model.config.use_cache = False

    if FREEZE_ENCODER:
        model.freeze_encoder()

    train_dataset = prepare_whisper_dataset(train_raw, processor)
    val_dataset = prepare_whisper_dataset(val_raw, processor)
    test_dataset = prepare_whisper_dataset(test_raw, processor)

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(pred) -> dict[str, float]:
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = pred.label_ids
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = [normalize_text(text) for text in pred_str]
        label_str = [normalize_text(text) for text in label_str]
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    args = build_training_args()
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor.feature_extractor
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    save_run_manifest({"whisper_language": WHISPER_LANGUAGE, "whisper_task": WHISPER_TASK})
    write_model_card(baseline_only=False)

    checkpoint = None
    if RESUME_FROM_CHECKPOINT and any(OUTPUT_DIR.glob("checkpoint-*")):
        checkpoint = True
        print("Resuming from latest checkpoint in", OUTPUT_DIR)

    print("Starting Whisper fine-tuning...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    trainer.train(resume_from_checkpoint=checkpoint)

    print("Saving model/processor...")
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    print("Evaluating on held-out test subset...")
    metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    (OUTPUT_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_model_card(metrics, baseline_only=False)
    maybe_push_to_hub()
    print(json.dumps(metrics, indent=2))


# ============================================================================
# Cell 6 — Qwen3-ASR zero-shot comparable baseline
# ============================================================================
def run_qwen3_asr_eval() -> None:
    """Run Qwen3-ASR as a comparable zero-shot ASR baseline."""
    try:
        from qwen_asr import Qwen3ASRModel
    except Exception as exc:
        raise RuntimeError(
            "qwen-asr is unavailable. In Colab, run the install cell, restart "
            "runtime, and verify `pip install -U qwen-asr` completed."
        ) from exc

    dtype_setting = QWEN_ASR_DTYPE.lower()
    if dtype_setting == "auto":
        if not torch.cuda.is_available():
            dtype = torch.float32
        elif torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype_setting, torch.float16 if torch.cuda.is_available() else torch.float32)
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3ASRModel.from_pretrained(
        QWEN_ASR_MODEL_ID,
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=QWEN_ASR_MAX_BATCH_SIZE,
        max_new_tokens=QWEN_ASR_MAX_NEW_TOKENS,
    )

    rows = []
    predictions = []
    references = []

    for idx, example in enumerate(test_raw):
        audio = audio_to_array(example["audio"], target_sr=TARGET_SR)
        result = model.transcribe(audio=(audio, TARGET_SR), language=QWEN_ASR_LANGUAGE)[0]
        pred = normalize_text(getattr(result, "text", result.get("text") if isinstance(result, dict) else str(result)))
        detected_language = getattr(result, "language", result.get("language") if isinstance(result, dict) else None)
        ref = normalize_text(example["transcript"])
        predictions.append(pred)
        references.append(ref)
        rows.append({"idx": idx, "prediction": pred, "reference": ref, "detected_language": detected_language})
        if (idx + 1) % 10 == 0:
            print(f"Qwen3-ASR eval {idx + 1}/{len(test_raw)}")

    wer = wer_metric.compute(predictions=predictions, references=references)
    out_csv = OUTPUT_DIR / "qwen3_asr_predictions.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["idx", "prediction", "reference", "detected_language"])
        writer.writeheader()
        writer.writerows(rows)

    save_run_manifest({
        "qwen_asr_language": QWEN_ASR_LANGUAGE,
        "qwen_asr_dtype": QWEN_ASR_DTYPE,
        "qwen_asr_max_new_tokens": QWEN_ASR_MAX_NEW_TOKENS,
        "qwen_asr_max_batch_size": QWEN_ASR_MAX_BATCH_SIZE,
    })
    result = {"test_wer": wer, "n": len(rows), "predictions_csv": str(out_csv)}
    (OUTPUT_DIR / "qwen3_asr_test_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_model_card(result, baseline_only=True)
    maybe_push_to_hub()
    print(json.dumps(result, indent=2))


# ============================================================================
# Cell 7 — Run selected backend
# ============================================================================
if ASR_BACKEND == "whisper_finetune":
    run_whisper_finetune()
elif ASR_BACKEND == "qwen3_asr_eval":
    run_qwen3_asr_eval()
else:
    raise ValueError(f"Unknown ASR_BACKEND={ASR_BACKEND!r}")
