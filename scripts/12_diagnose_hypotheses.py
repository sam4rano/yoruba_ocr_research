"""
Hypothesis tests for OCR evaluation: separate data issues, metric bugs, and model setup.

Run from repo root in this order:

1. eval      — Assert known (pred, gt) pairs produce expected CER/WER/DER.
2. identity  — Ground-truth as prediction → all metrics must be 0 (loader + metrics).
3. data      — Label file stats, missing images, optional random sample for human review.
4. replay    — Recompute metrics from a per-sample JSONL log (must match saved rows).

Setup (Qwen, Paddle, etc.) is validated indirectly: if (1)–(3) pass, bad model scores
reflect the recogniser or prompt, not the dataset or eval code.

Usage:
    python scripts/12_diagnose_hypotheses.py eval
    python scripts/12_diagnose_hypotheses.py identity --data-dir data/processed --split test
    python scripts/12_diagnose_hypotheses.py data --split test --sample 15 --seed 0
    python scripts/12_diagnose_hypotheses.py replay --jsonl results/tables/qwen25_vl_zero_shot_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import unicodedata
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_eval_assertions() -> None:
    """Verify metric functions against hand-checked examples."""
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_utils import compute_cer, compute_der, compute_wer  # noqa: E402

    # Perfect copy
    s = unicodedata.normalize("NFC", "àbọ̀ ẹ̀kọ́")
    c, w, d = compute_cer(s, s), compute_wer(s, s), compute_der(s, s)
    assert (c, w, d) == (0.0, 0.0, 0.0), f"identity triple got {(c, w, d)}"

    # Empty gt: defined as 0 error if pred also empty
    assert (compute_cer("", ""), compute_wer("", ""), compute_der("", "")) == (0.0, 0.0, 0.0)

    # Empty pred vs non-empty gt
    assert compute_cer("", "ab") == 1.0
    assert compute_wer("", "two words") == 1.0
    gt_tone = unicodedata.normalize("NFC", "à")
    assert abs(compute_der("", gt_tone) - 1.0) < 1e-9

    # CER can exceed 1 when insertions dominate
    cer_long = compute_cer("x" * 20, "hi")
    assert cer_long > 1.0, "CER should exceed 1 when pred is much longer than gt"

    log.info("eval: all metric assertions passed.")


def run_identity(data_dir: Path, split: str, max_pairs: int | None) -> None:
    """Use ground truth as prediction; mean CER/WER/DER must be 0."""
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_utils import aggregate_metrics, load_test_pairs  # noqa: E402

    pairs = load_test_pairs(data_dir, split)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    pred_gt = [(gt, gt) for _, gt in pairs]
    m = aggregate_metrics(pred_gt)
    log.info(
        "identity: n=%d  CER=%s  WER=%s  DER=%s",
        m["n"],
        m["cer"],
        m["wer"],
        m["der"],
    )
    if m["n"] == 0:
        raise SystemExit("identity: no pairs loaded — check data-dir and split.")
    tol = 1e-9
    if abs(m["cer"] or 0) > tol or abs(m["wer"] or 0) > tol or abs(m["der"] or 0) > tol:
        bad = [r for r in m["rows"] if r["cer"] > tol or r["wer"] > tol or r["der"] > tol]
        log.error("identity failed on %d rows (gt != pred after NFC?)", len(bad))
        for r in bad[:5]:
            log.error("  cer=%s wer=%s der=%s gt=%r", r["cer"], r["wer"], r["der"], r["gt"])
        raise SystemExit(1)
    log.info("identity: PASS — labels + metrics pipeline is self-consistent.")


def run_data_inventory(
    data_dir: Path,
    split: str,
    sample: int,
    seed: int,
    out_json: Path,
    sample_jsonl: Path,
) -> None:
    """Summarise label file and write a random sample for manual image/GT checks."""
    label_file = data_dir / "labels" / f"{split}.txt"
    if not label_file.exists():
        raise SystemExit(f"Missing label file: {label_file}")

    lines_total = 0
    parsed = 0
    missing_img = 0
    char_lens: list[int] = []
    loaded: list[tuple[str, str, Path]] = []

    with label_file.open(encoding="utf-8") as fh:
        for line in fh:
            lines_total += 1
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            rel_path, gt = parts
            parsed += 1
            p = data_dir / rel_path
            if not p.exists():
                missing_img += 1
                continue
            gt_nfc = unicodedata.normalize("NFC", gt)
            char_lens.append(len(gt_nfc))
            loaded.append((rel_path, gt_nfc, p))

    rng = random.Random(seed)
    n_sample = min(sample, len(loaded))
    picked = rng.sample(loaded, n_sample) if loaded else []

    sample_rows = []
    for rel, gt, full in picked:
        sample_rows.append(
            {
                "relative_path": rel,
                "absolute_path": str(full.resolve()),
                "gt": gt,
                "gt_char_len": len(gt),
            }
        )

    summary = {
        "label_file": str(label_file),
        "split": split,
        "lines_in_file": lines_total,
        "tab_parsed_rows": parsed,
        "pairs_with_existing_image": len(loaded),
        "missing_image_skipped_by_loader": missing_img,
        "char_len_min": min(char_lens) if char_lens else None,
        "char_len_max": max(char_lens) if char_lens else None,
        "char_len_mean": round(sum(char_lens) / len(char_lens), 4) if char_lens else None,
        "sample_size": n_sample,
        "sample_seed": seed,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("data: wrote %s", out_json)

    with sample_jsonl.open("w", encoding="utf-8") as fh:
        for row in sample_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("data: wrote %s (%d rows for manual review)", sample_jsonl, len(sample_rows))
    log.info(
        "data: Open images listed in the JSONL and confirm gt matches visible text."
    )


def run_replay(jsonl_path: Path, tolerance: float) -> None:
    """Recompute per-row metrics from JSONL and compare to stored values."""
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_utils import compute_cer, compute_der, compute_wer  # noqa: E402

    if not jsonl_path.exists():
        raise SystemExit(f"JSONL not found: {jsonl_path}")

    mismatches = 0
    n = 0
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pred, gt = row["pred"], row["gt"]
            c, w, d = compute_cer(pred, gt), compute_wer(pred, gt), compute_der(pred, gt)
            n += 1
            if abs(c - row["cer"]) > tolerance:
                log.warning("replay CER mismatch line %d: %s vs %s", n, c, row["cer"])
                mismatches += 1
            if abs(w - row["wer"]) > tolerance:
                log.warning("replay WER mismatch line %d: %s vs %s", n, w, row["wer"])
                mismatches += 1
            if abs(d - row["der"]) > tolerance:
                log.warning("replay DER mismatch line %d: %s vs %s", n, d, row["der"])
                mismatches += 1

    if n == 0:
        raise SystemExit("replay: empty JSONL")
    if mismatches:
        raise SystemExit(f"replay: FAILED with {mismatches} metric mismatches")
    log.info("replay: PASS — %d rows match recomputed CER/WER/DER", n)


def run_setup_hint() -> None:
    """Print how to isolate Qwen vs Paddle without conflating eval issues."""
    log.info(
        "setup: If eval + identity + replay pass, poor Qwen scores indicate "
        "model/prompt/quantisation — not label files or metric code."
    )
    log.info(
        "setup: Quick Qwen check: "
        "SKIP_QWEN=0 bash scripts/shell/phase_07_qwen.sh with "
        "QWEN_MAX_SAMPLES=5 and inspect results/tables/qwen25_vl_zero_shot_test.jsonl."
    )
    log.info(
        "setup: Ablation: run same --max-samples with and without --quantize "
        "in scripts/09_baseline_qwen.py."
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Diagnose dataset vs eval vs setup for Yorùbá OCR experiments."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("eval", help="Unit-test metric functions")

    p_id = sub.add_parser("identity", help="pred=gt must yield zero error rates")
    p_id.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p_id.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_id.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Limit pairs (default: all loaded with existing images).",
    )

    p_d = sub.add_parser("data", help="Inventory labels + sample paths for manual QA")
    p_d.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p_d.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_d.add_argument("--sample", type=int, default=20, help="Random lines to export")
    p_d.add_argument("--seed", type=int, default=42)
    p_d.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/tables/diagnose_data_inventory.json"),
    )
    p_d.add_argument(
        "--sample-jsonl",
        type=Path,
        default=Path("results/tables/diagnose_sample_for_review.jsonl"),
    )

    p_r = sub.add_parser("replay", help="Verify JSONL metrics match recomputation")
    p_r.add_argument(
        "--jsonl",
        type=Path,
        default=Path("results/tables/qwen25_vl_zero_shot_test.jsonl"),
    )
    p_r.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Allowed float drift for stored vs recomputed metrics.",
    )

    sub.add_parser("setup-hint", help="Reminders for isolating Qwen/Paddle issues")

    return parser.parse_args()


def main() -> None:
    """Dispatch subcommands."""
    args = parse_args()
    if args.command == "eval":
        run_eval_assertions()
    elif args.command == "identity":
        run_identity(args.data_dir, args.split, args.max_pairs)
    elif args.command == "data":
        run_data_inventory(
            args.data_dir,
            args.split,
            args.sample,
            args.seed,
            args.out_json,
            args.sample_jsonl,
        )
    elif args.command == "replay":
        run_replay(args.jsonl, args.tolerance)
    elif args.command == "setup-hint":
        run_setup_hint()
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
