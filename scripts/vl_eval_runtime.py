"""Resumable, failure-aware runtime helpers for multimodal OCR evaluation."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable, Sequence

log = logging.getLogger(__name__)

BatchTranscriber = Callable[[list[Path]], list[str]]


def run_fingerprint(payload: dict) -> str:
    """Return a stable fingerprint for model, prompt, data, and runtime settings."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sample_id(image_path: Path, ground_truth: str) -> str:
    """Identify a sample independently of notebook or mount location."""
    stable_path = "/".join(image_path.parts[-3:])
    value = f"{stable_path}\0{ground_truth}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def load_partial(path: Path, fingerprint: str) -> dict[str, dict]:
    """Load completed rows from a compatible partial JSONL checkpoint."""
    if not path.is_file():
        return {}
    rows: dict[str, dict] = {}
    incompatible = 0
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Ignoring malformed partial row %d in %s", line_no, path)
                continue
            if row.get("run_fingerprint") != fingerprint:
                incompatible += 1
                continue
            if row.get("status") != "ok":
                continue
            sid = row.get("sample_id")
            if sid:
                rows[sid] = row
    if incompatible:
        log.warning(
            "Ignored %d partial rows from a different model/prompt/data configuration.",
            incompatible,
        )
    return rows


def _append_partial(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def _infer_batch(
    batch: list[tuple[Path, str]],
    transcribe_batch: BatchTranscriber,
    fingerprint: str,
) -> tuple[list[dict], bool]:
    """Infer one batch, retrying per image when batched processing is unsupported."""
    started = time.perf_counter()
    try:
        predictions = transcribe_batch([path for path, _ in batch])
        if len(predictions) != len(batch):
            raise ValueError(
                f"batch returned {len(predictions)} predictions for {len(batch)} images"
            )
        elapsed = time.perf_counter() - started
        return [
            {
                "sample_id": sample_id(path, gt),
                "image_path": path.as_posix(),
                "gt": gt,
                "pred": pred,
                "status": "ok",
                "error": None,
                "elapsed_seconds": round(elapsed / max(1, len(batch)), 4),
                "run_fingerprint": fingerprint,
            }
            for (path, gt), pred in zip(batch, predictions, strict=True)
        ], False
    except Exception as batch_exc:  # noqa: BLE001
        if len(batch) == 1:
            path, gt = batch[0]
            return [
                {
                    "sample_id": sample_id(path, gt),
                    "image_path": path.as_posix(),
                    "gt": gt,
                    "pred": "",
                    "status": "error",
                    "error": f"{type(batch_exc).__name__}: {batch_exc}",
                    "elapsed_seconds": round(time.perf_counter() - started, 4),
                    "run_fingerprint": fingerprint,
                }
            ], False

        log.warning(
            "Batch of %d failed (%s); retrying each image independently.",
            len(batch),
            batch_exc,
        )
        rows: list[dict] = []
        for item in batch:
            item_rows, _ = _infer_batch([item], transcribe_batch, fingerprint)
            rows.extend(item_rows)
        return rows, True


def evaluate_resumable(
    pairs: list[tuple[Path, str]],
    *,
    transcribe_batch: BatchTranscriber,
    batch_size: int,
    partial_path: Path,
    fingerprint: str,
    resume: bool,
    description: str,
) -> tuple[list[tuple[str, str]], list[dict]]:
    """Evaluate all pairs with incremental JSONL checkpoints and ordered output."""
    from tqdm import tqdm

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    completed = load_partial(partial_path, fingerprint) if resume else {}
    if completed:
        log.info("Resuming %d completed samples from %s", len(completed), partial_path)

    pending = [
        pair for pair in pairs if sample_id(pair[0], pair[1]) not in completed
    ]
    progress = tqdm(total=len(pairs), initial=len(pairs) - len(pending), desc=description, unit="img")
    start = 0
    effective_batch_size = batch_size
    while start < len(pending):
        batch = pending[start : start + effective_batch_size]
        rows, used_fallback = _infer_batch(batch, transcribe_batch, fingerprint)
        _append_partial(partial_path, rows)
        for row in rows:
            completed[row["sample_id"]] = row
        progress.update(len(batch))
        start += len(batch)
        if used_fallback:
            effective_batch_size = 1
            log.warning("Continuing with batch_size=1 after batched generation failed.")
    progress.close()

    ordered_rows = [completed[sample_id(path, gt)] for path, gt in pairs]
    metric_pairs = [(row["pred"], row["gt"]) for row in ordered_rows]
    return metric_pairs, ordered_rows
