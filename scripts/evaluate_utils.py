"""
Shared evaluation utilities: metric computation, data loading, result persistence.

Imported by evaluation and baseline scripts (e.g. 05_evaluate.py, 06_baseline_pretrained.py,
09_baseline_qwen.py, 15_baseline_paddleocr_vl15.py, 12_diagnose_hypotheses.py,
13_verify_eval_alignment.py).
Not intended to be run directly.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import editdistance

# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

# Columns written when the metrics CSV is first created. Older CSVs may
# predate some of these; see _write_csv_row for the compatibility path.
_DEFAULT_CSV_FIELDS = [
    "model",
    "split",
    "n",
    "cer",
    "wer",
    "der",
    "der_n",
    "der_insertion_rate",
    "phantom",
    "meta_path",
    "timestamp",
]


def _sha256_file(path: Path) -> str | None:
    """Return the hex SHA-256 of ``path``, or ``None`` if unreadable."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, FileNotFoundError):
        return None


def _git_sha() -> str | None:
    """Return the current git HEAD sha (short), or ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return None


def _detect_existing_fieldnames(csv_path: Path) -> list[str] | None:
    """Return the header row of ``csv_path`` if it already exists and has one."""
    if not csv_path.exists():
        return None
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            return header if header else None
    except (OSError, StopIteration):
        return None


_NON_PADDLE_MODEL_KINDS = frozenset(
    {"tesseract", "qwen_vl", "qwen2_vl", "paddleocr_vl"}
)


def _phantom_flag_from_provenance(provenance: dict | None) -> str:
    """Derive the CSV ``phantom`` column value from a provenance payload.

    Values:

    * ``"true"``  — checkpoint-integrity check found missing / shape-
      mismatched head weights; the row measures a random head.
    * ``"false"`` — integrity check passed.
    * ``"n/a"``   — ``model_kind`` is a non-Paddle baseline (Tesseract,
      Qwen, PaddleOCR-VL) for which the "phantom head" concept does not
      apply.
    * ``"unknown"`` — no integrity data and no recognised ``model_kind``.
    """
    if not provenance:
        return "unknown"
    model_kind = provenance.get("model_kind")
    if model_kind in _NON_PADDLE_MODEL_KINDS:
        return "n/a"
    integrity = provenance.get("checkpoint_integrity")
    if not integrity:
        return "unknown"
    head_bad = integrity.get("missing_by_component", {}).get("head", 0) + integrity.get(
        "shape_mismatch_by_component", {}
    ).get("head", 0)
    return "true" if head_bad > 0 else "false"


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def nfc(text: str) -> str:
    """Return NFC-normalised form for consistent character comparison."""
    return unicodedata.normalize("NFC", text)


def compute_cer(pred: str, gt: str) -> float:
    """
    Character Error Rate.

    edit_distance over NFC-normalised character sequences,
    divided by ground-truth character count.
    """
    pred_n, gt_n = nfc(pred), nfc(gt)
    if not gt_n:
        return 0.0 if not pred_n else 1.0
    return editdistance.eval(list(pred_n), list(gt_n)) / len(gt_n)


def compute_wer(pred: str, gt: str) -> float:
    """
    Word Error Rate.

    edit_distance over NFC-normalised word sequences,
    divided by ground-truth word count.
    """
    pred_words = nfc(pred).split()
    gt_words = nfc(gt).split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    return editdistance.eval(pred_words, gt_words) / len(gt_words)


def compute_der(pred: str, gt: str) -> float:
    """
    Per-sample Diacritic Error Rate — diagnostic value kept in JSONL rows.

    NFD-decomposes both strings to isolate combining diacritics
    (U+0300 grave / low tone, U+0301 acute / high tone,
    U+0323 combining dot below for ẹ/ọ/ṣ variants),
    then computes edit distance over those combining character sequences
    normalised by the ground-truth diacritic count.

    A DER of 0.0 means all tone marks and subdots were reproduced exactly.
    A DER close to 1.0 indicates the model is producing base letters
    without tonal fidelity — the failure mode of English pretrained models
    on Yorùbá text.

    .. warning::

       For the **aggregate** DER reported in ``metrics.csv``, do **not**
       macro-average this per-sample ratio. Short GT strings with 1 or 2
       diacritics produce 0.0 / 1.0 floors that dominate the mean and hide
       real performance differences. Use :func:`aggregate_metrics`, which
       computes a corpus-level (micro-averaged) DER over the subset of
       samples whose GT contains at least one diacritic, and reports the
       insertion rate on the zero-diacritic subset separately.
    """
    pred_diacs = [
        c for c in unicodedata.normalize("NFD", pred) if unicodedata.combining(c)
    ]
    gt_diacs = [c for c in unicodedata.normalize("NFD", gt) if unicodedata.combining(c)]

    if not gt_diacs:
        return 0.0 if not pred_diacs else 1.0
    return editdistance.eval(pred_diacs, gt_diacs) / len(gt_diacs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_test_pairs(data_dir: Path, split: str) -> list[tuple[Path, str]]:
    """
    Load (image_path, ground_truth_text) pairs from a PaddleOCR label file.

    Skips entries where the image file is missing from disk.
    Returns pairs ordered as they appear in the label file.
    """
    label_file = data_dir / "labels" / f"{split}.txt"
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    pairs: list[tuple[Path, str]] = []
    missing = 0
    with label_file.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            rel_path, gt = parts
            img_path = data_dir / rel_path
            if not img_path.exists():
                missing += 1
                continue
            pairs.append((img_path, unicodedata.normalize("NFC", gt)))

    return pairs


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


def aggregate_metrics(pairs: list[tuple[str, str]]) -> dict:
    """
    Compute aggregate CER / WER / DER over all ``(prediction, ground_truth)`` pairs.

    * **CER, WER** — macro-averaged per-sample rates (each sample weighted
      equally). Rates may exceed 1 when insertions inflate edit distance
      beyond reference length; see ``docs/metrics_conventions.md``.
    * **DER** — **corpus-level (micro-averaged)** ratio of total diacritic
      edit distance to total GT diacritic count, computed only over the
      subset of samples whose ground-truth contains at least one combining
      diacritic. Samples whose GT has zero diacritics cannot meaningfully
      contribute to a recall-style ratio and are excluded from ``der``.
      Instead they are summarised in ``der_insertion_rate`` — predicted
      diacritics per GT character, a precision-side signal for the model
      hallucinating tone marks where none exist.

    Additional fields returned alongside the headline numbers:

    * ``der_n`` — number of samples with ≥1 GT diacritic that contributed
      to ``der``. When ``der_n == 0`` the dataset had no diacritic-bearing
      ground truth and ``der`` is ``None``.
    * ``der_insertion_rate`` — ``None`` when every sample has diacritics.
    * ``rows[*].der`` — preserved per-sample legacy float (see
      :func:`compute_der`) for audit and JSONL inspection. Do not macro-
      average this externally; use the aggregate ``der`` instead.
    """
    rows = []
    total_der_edits = 0
    total_gt_diacs = 0
    der_n_samples = 0

    total_insertions = 0
    total_gt_chars_nodiac = 0
    for pred, gt in pairs:
        pred_nfd = unicodedata.normalize("NFD", pred)
        gt_nfd = unicodedata.normalize("NFD", gt)
        pred_diacs = [c for c in pred_nfd if unicodedata.combining(c)]
        gt_diacs = [c for c in gt_nfd if unicodedata.combining(c)]

        if gt_diacs:
            edits = editdistance.eval(pred_diacs, gt_diacs)
            total_der_edits += edits
            total_gt_diacs += len(gt_diacs)
            der_n_samples += 1
            per_sample_der = edits / len(gt_diacs)
        else:
            total_insertions += len(pred_diacs)
            total_gt_chars_nodiac += len(gt)
            per_sample_der = 0.0 if not pred_diacs else 1.0

        rows.append(
            {
                "pred": pred,
                "gt": gt,
                "cer": compute_cer(pred, gt),
                "wer": compute_wer(pred, gt),
                "der": per_sample_der,
            }
        )

    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "cer": None,
            "wer": None,
            "der": None,
            "der_n": 0,
            "der_insertion_rate": None,
            "rows": [],
        }

    der = round(total_der_edits / total_gt_diacs, 4) if total_gt_diacs else None
    der_insertion_rate = (
        round(total_insertions / total_gt_chars_nodiac, 4)
        if total_gt_chars_nodiac
        else None
    )

    return {
        "n": n,
        "cer": round(sum(r["cer"] for r in rows) / n, 4),
        "wer": round(sum(r["wer"] for r in rows) / n, 4),
        "der": der,
        "der_n": der_n_samples,
        "der_insertion_rate": der_insertion_rate,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def save_results(
    metrics: dict,
    model_name: str,
    split: str,
    csv_path: Path,
    jsonl_path: Path,
    *,
    provenance: dict | None = None,
) -> Path | None:
    """
    Persist aggregate scores to the metrics CSV, a per-sample JSONL log,
    and — when ``provenance`` is supplied — a sibling ``meta.json`` that
    records exactly which artifacts produced this row.

    The CSV row is appended so multiple models accumulate in the same table.

    Columns of interest beyond the usual ``cer`` / ``wer``:

    * ``der`` — corpus-level diacritic error rate over the subset of
      samples whose GT contains ≥1 diacritic. Blank when the split has no
      diacritic-bearing ground truth.
    * ``der_n`` — number of samples that contributed to ``der``.
    * ``der_insertion_rate`` — diacritics predicted per GT character on
      the zero-diacritic subset. Blank when every sample has diacritics.
    * ``phantom`` — checkpoint-integrity flag:
      ``"true"``  = head weights not restored; the row measures a random head;
      ``"false"`` = clean;
      ``"n/a"``   = non-Paddle baseline (Tesseract, Qwen, PaddleOCR-VL);
      ``"unknown"`` = provenance did not include an integrity report.

    Returns the path to the sibling meta file (or ``None`` when no
    provenance was given, for backward compatibility with older callers).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    phantom_flag = _phantom_flag_from_provenance(provenance)

    meta_path: Path | None = None
    if provenance is not None:
        meta_dir = csv_path.parent / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / f"{model_name}_{split}.json"

    row: dict[str, object] = {
        "model": model_name,
        "split": split,
        "n": metrics["n"],
        "cer": metrics["cer"],
        "wer": metrics["wer"],
        "der": metrics["der"] if metrics["der"] is not None else "",
        "der_n": metrics.get("der_n", ""),
        "der_insertion_rate": (
            metrics["der_insertion_rate"]
            if metrics.get("der_insertion_rate") is not None
            else ""
        ),
        "phantom": phantom_flag,
        "meta_path": (
            meta_path.relative_to(csv_path.parent.parent).as_posix()
            if meta_path is not None
            else ""
        ),
        "timestamp": timestamp,
    }

    existing_fields = _detect_existing_fieldnames(csv_path)
    if existing_fields is None:
        # Fresh file: write the full schema.
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_DEFAULT_CSV_FIELDS)
            writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in _DEFAULT_CSV_FIELDS})
    else:
        # Respect whatever header is already on disk so we don't corrupt
        # older runs' ``metrics.csv``. Missing columns (e.g. ``phantom``)
        # are simply dropped; to adopt the new schema delete or archive
        # the existing file first.
        with csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=existing_fields)
            writer.writerow({k: row.get(k, "") for k in existing_fields})

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for sample_row in metrics["rows"]:
            fh.write(json.dumps(sample_row, ensure_ascii=False) + "\n")

    if provenance is not None and meta_path is not None:
        meta_payload = _build_meta_payload(
            model_name=model_name,
            split=split,
            metrics=metrics,
            timestamp=timestamp,
            jsonl_path=jsonl_path,
            csv_path=csv_path,
            phantom_flag=phantom_flag,
            provenance=provenance,
        )
        meta_path.write_text(
            json.dumps(meta_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return meta_path


def _build_meta_payload(
    *,
    model_name: str,
    split: str,
    metrics: dict,
    timestamp: str,
    jsonl_path: Path,
    csv_path: Path,
    phantom_flag: str,
    provenance: dict,
) -> dict:
    """Assemble the sibling ``meta.json`` payload.

    We resolve content hashes for the checkpoint, dictionary and rec config
    (whichever are present in ``provenance``) so that re-running an eval
    against a drifted dictionary or a silently re-trained checkpoint
    produces a different hash and is easy to spot during audits.
    """
    ckpt_integrity = provenance.get("checkpoint_integrity") or {}
    ckpt_prefix = ckpt_integrity.get("ckpt_prefix") or provenance.get("ckpt_prefix")
    ckpt_params = Path(str(ckpt_prefix) + ".pdparams") if ckpt_prefix else None

    dict_path_raw = provenance.get("dict_path")
    dict_path = Path(dict_path_raw) if dict_path_raw else None

    rec_config_raw = provenance.get("rec_config")
    rec_config_path = Path(rec_config_raw) if rec_config_raw else None

    dict_size: int | None = None
    if dict_path is not None and dict_path.exists():
        try:
            dict_size = sum(1 for _ in dict_path.open("r", encoding="utf-8"))
        except OSError:
            dict_size = None

    return {
        "model": model_name,
        "split": split,
        "timestamp": timestamp,
        "phantom": phantom_flag,
        "metrics": {
            "n": metrics["n"],
            "cer": metrics["cer"],
            "wer": metrics["wer"],
            "der": metrics["der"],
            "der_n": metrics.get("der_n"),
            "der_insertion_rate": metrics.get("der_insertion_rate"),
        },
        "git_sha": _git_sha(),
        "env": {
            "cwd": os.getcwd(),
        },
        "artifacts": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "checkpoint_pdparams": str(ckpt_params) if ckpt_params else None,
            "checkpoint_sha256": _sha256_file(ckpt_params) if ckpt_params else None,
            "dict_path": str(dict_path) if dict_path else None,
            "dict_sha256": _sha256_file(dict_path) if dict_path else None,
            "dict_size": dict_size,
            "rec_config": str(rec_config_path) if rec_config_path else None,
            "rec_config_sha256": (
                _sha256_file(rec_config_path) if rec_config_path else None
            ),
        },
        "provenance": provenance,
    }
