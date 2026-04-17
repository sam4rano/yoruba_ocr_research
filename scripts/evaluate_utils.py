"""
Shared evaluation utilities: metric computation, data loading, result persistence.

Imported by evaluation and baseline scripts (e.g. 05_evaluate.py, 06_baseline_pretrained.py,
09_baseline_qwen.py, 15_baseline_paddleocr_vl15.py, 12_diagnose_hypotheses.py,
13_verify_eval_alignment.py).
Not intended to be run directly.
"""

from __future__ import annotations

import csv
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import editdistance


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
    Diacritic Error Rate — novel metric for this paper.

    NFD-decomposes both strings to isolate combining diacritics
    (U+0300 grave / low tone, U+0301 acute / high tone,
    U+0323 combining dot below for ẹ/ọ/ṣ variants),
    then computes edit distance over those combining character sequences
    normalised by the ground-truth diacritic count.

    A DER of 0.0 means all tone marks and subdots were reproduced exactly.
    A DER close to 1.0 indicates the model is producing base letters
    without tonal fidelity — the failure mode of English pretrained models
    on Yorùbá text.
    """
    pred_diacs = [c for c in unicodedata.normalize("NFD", pred) if unicodedata.combining(c)]
    gt_diacs = [c for c in unicodedata.normalize("NFD", gt) if unicodedata.combining(c)]

    if not gt_diacs:
        return 0.0 if not pred_diacs else 1.0
    return editdistance.eval(pred_diacs, gt_diacs) / len(gt_diacs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_pairs(
    data_dir: Path, split: str
) -> list[tuple[Path, str]]:
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
    Compute mean CER, WER, DER over all (prediction, ground_truth) pairs.

    Returns a dict with aggregate scores and per-sample rows for full
    traceability.
    """
    rows = []
    for pred, gt in pairs:
        rows.append(
            {
                "pred": pred,
                "gt": gt,
                "cer": compute_cer(pred, gt),
                "wer": compute_wer(pred, gt),
                "der": compute_der(pred, gt),
            }
        )

    n = len(rows)
    if n == 0:
        return {"n": 0, "cer": None, "wer": None, "der": None, "rows": []}

    return {
        "n": n,
        "cer": round(sum(r["cer"] for r in rows) / n, 4),
        "wer": round(sum(r["wer"] for r in rows) / n, 4),
        "der": round(sum(r["der"] for r in rows) / n, 4),
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
) -> None:
    """
    Persist aggregate scores to results/tables/metrics.csv and a per-sample
    JSONL log.

    The CSV row is appended so multiple models accumulate in the same table,
    ready for direct comparison in the paper's results section.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "split", "n", "cer", "wer", "der", "timestamp"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "model": model_name,
                "split": split,
                "n": metrics["n"],
                "cer": metrics["cer"],
                "wer": metrics["wer"],
                "der": metrics["der"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in metrics["rows"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
