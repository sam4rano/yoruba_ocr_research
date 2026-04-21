"""
Hypothesis tests for OCR evaluation: separate data issues, metric bugs, and model setup.

Run from repo root in this order:

1. eval        — Assert known (pred, gt) pairs produce expected CER/WER/DER.
2. identity    — Ground-truth as prediction → all metrics must be 0 (loader + metrics).
3. data        — Label file stats, missing images, optional random sample for human review.
4. replay      — Recompute metrics from a per-sample JSONL log (must match saved rows).
5. checkpoints — Flag rows in metrics.csv that came from phantom / unverifiable checkpoints.

Setup (Qwen, Paddle, etc.) is validated indirectly: if (1)–(3) pass, bad model scores
reflect the recogniser or prompt, not the dataset or eval code.

Usage:
    python scripts/12_diagnose_hypotheses.py eval
    python scripts/12_diagnose_hypotheses.py identity --data-dir data/processed --split test
    python scripts/12_diagnose_hypotheses.py data --split test --sample 15 --seed 0
    python scripts/12_diagnose_hypotheses.py replay --jsonl results/tables/qwen25_vl_zero_shot_test.jsonl
    python scripts/12_diagnose_hypotheses.py checkpoints --csv results/tables/metrics.csv
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


def run_checkpoints_audit(
    csv_path: Path,
    *,
    write_report: Path | None = None,
    fail_on_phantom: bool = False,
) -> None:
    """Audit a metrics CSV against sibling ``meta.json`` files.

    A row is flagged as:

    * ``phantom``   — the integrity report attached to the row's meta file
      says at least one CTC-head weight was not restored. The row is
      measuring a randomly-initialised head and should be treated as a
      zero-skill reference, not a trained model.
    * ``no_meta``   — no ``meta.json`` was ever written for this row (i.e.
      it predates the provenance patch). Treat with caution; the row could
      be phantom and we have no way to prove otherwise.
    * ``stale``     — the meta file exists but the checkpoint it refers to
      has since disappeared, so the evidence backing this row is gone.
    * ``ok``        — integrity report attached and clean.

    This surfaces exactly the class of problem we documented in the
    forensic analysis: baseline/fine-tune rows whose CER/WER numbers came
    from a random CTC projection on top of an English encoder.
    """
    import csv as _csv

    if not csv_path.exists():
        raise SystemExit(f"checkpoints: metrics CSV not found: {csv_path}")

    tables_dir = csv_path.parent
    meta_dir = tables_dir / "meta"

    summary: list[dict] = []
    counts: dict[str, int] = {"ok": 0, "phantom": 0, "no_meta": 0, "stale": 0}

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            model = row.get("model", "")
            split = row.get("split", "")
            csv_phantom_flag = (row.get("phantom") or "").strip().lower()
            meta_path = meta_dir / f"{model}_{split}.json"

            status = "ok"
            notes: list[str] = []
            ckpt_sha: str | None = None
            ckpt_path: str | None = None

            if not meta_path.exists():
                status = "no_meta"
                if csv_phantom_flag == "true":
                    status = "phantom"
                    notes.append("CSV phantom=true but no sibling meta.json")
            else:
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    status = "no_meta"
                    notes.append(f"unreadable meta.json: {exc}")
                    meta = None

                if meta is not None:
                    phantom = str(meta.get("phantom", "")).lower()
                    artifacts = meta.get("artifacts") or {}
                    ckpt_path = artifacts.get("checkpoint_pdparams")
                    ckpt_sha = artifacts.get("checkpoint_sha256")

                    if phantom == "true":
                        status = "phantom"
                        integrity = (
                            meta.get("provenance", {}).get("checkpoint_integrity") or {}
                        )
                        bad_head = (
                            integrity.get("missing_by_component", {}).get("head", 0)
                            + integrity.get("shape_mismatch_by_component", {}).get("head", 0)
                        )
                        notes.append(f"head weights not restored: {bad_head}")

                    if ckpt_path:
                        ckpt_file = Path(ckpt_path)
                        if not ckpt_file.exists():
                            if status == "ok":
                                status = "stale"
                            notes.append(f"checkpoint missing on disk: {ckpt_path}")

            counts[status] = counts.get(status, 0) + 1
            summary.append(
                {
                    "model": model,
                    "split": split,
                    "status": status,
                    "cer": row.get("cer"),
                    "wer": row.get("wer"),
                    "der": row.get("der"),
                    "meta_path": str(meta_path) if meta_path.exists() else None,
                    "checkpoint_pdparams": ckpt_path,
                    "checkpoint_sha256": ckpt_sha,
                    "notes": notes,
                }
            )

    log.info(
        "checkpoints: scanned %d rows — ok=%d phantom=%d no_meta=%d stale=%d",
        len(summary),
        counts.get("ok", 0),
        counts.get("phantom", 0),
        counts.get("no_meta", 0),
        counts.get("stale", 0),
    )
    for row in summary:
        if row["status"] != "ok":
            log.warning(
                "  %-10s %s/%s cer=%s wer=%s der=%s %s",
                row["status"].upper(),
                row["model"],
                row["split"],
                row["cer"],
                row["wer"],
                row["der"],
                "; ".join(row["notes"]) if row["notes"] else "",
            )

    if write_report is not None:
        write_report.parent.mkdir(parents=True, exist_ok=True)
        write_report.write_text(
            json.dumps({"counts": counts, "rows": summary}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        log.info("checkpoints: report written to %s", write_report)

    if fail_on_phantom and counts.get("phantom", 0) > 0:
        raise SystemExit(
            f"checkpoints: FAILED — {counts['phantom']} phantom rows in {csv_path}"
        )


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

    p_c = sub.add_parser(
        "checkpoints",
        help="Flag metrics.csv rows whose checkpoint cannot be verified",
    )
    p_c.add_argument(
        "--csv",
        type=Path,
        default=Path("results/tables/metrics.csv"),
        help="Metrics CSV to audit (default: results/tables/metrics.csv).",
    )
    p_c.add_argument(
        "--report",
        type=Path,
        default=Path("results/tables/checkpoint_audit.json"),
        help="Where to write the per-row JSON audit report.",
    )
    p_c.add_argument(
        "--fail-on-phantom",
        action="store_true",
        default=False,
        help="Exit non-zero when any row is flagged as phantom (for CI).",
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
    elif args.command == "checkpoints":
        run_checkpoints_audit(
            args.csv,
            write_report=args.report,
            fail_on_phantom=args.fail_on_phantom,
        )
    elif args.command == "setup-hint":
        run_setup_hint()
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
