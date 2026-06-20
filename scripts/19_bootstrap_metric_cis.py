"""
Bootstrap 95% confidence intervals for test-split CER, WER, and corpus DER.

Resamples test lines with replacement (line-level bootstrap) from existing
JSONL evaluation logs. Per-line CER/WER and diacritic edit counts are
precomputed once; each bootstrap replicate aggregates in O(n).

Inter-annotator agreement (Cohen's κ) is not computed here: raw exports contain
only final corrected labels (`corrected=yes`) with no per-annotator fields.

Usage:
    python scripts/19_bootstrap_metric_cis.py
    python scripts/19_bootstrap_metric_cis.py --n-bootstrap 10000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import editdistance
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from evaluate_utils import (  # noqa: E402
    aggregate_metrics,
    compute_cer,
    compute_wer,
    normalize_yoruba_text,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MODEL_JSONLS = [
    "baseline_english_pretrained_test.jsonl",
    "paddleocr_vl16_zero_shot_test.jsonl",
    "glm_ocr_zero_shot_test.jsonl",
    "surya_v2_zero_shot_test.jsonl",
    "paddleocr_vl16_finetuned_test.jsonl",
]

PAIRWISE_COMPARISONS = (
    ("paddleocr_vl16_zero_shot", "baseline_english_pretrained"),
    ("glm_ocr_zero_shot", "baseline_english_pretrained"),
    ("surya_v2_zero_shot", "baseline_english_pretrained"),
    ("paddleocr_vl16_finetuned", "paddleocr_vl16_zero_shot"),
    ("paddleocr_vl16_finetuned", "glm_ocr_zero_shot"),
)


@dataclass(frozen=True)
class LineStats:
    """Precomputed per-line quantities for fast bootstrap aggregation."""

    cer: float
    wer: float
    der_edits: int
    gt_diac_count: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for OCR metrics on test JSONL logs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        action="append",
        default=None,
        help="Per-sample JSONL log (repeatable). Defaults to Table 1 supervised rows.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap replicates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible resampling.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence level (default 0.95).",
    )
    return parser.parse_args()


def load_jsonl_pairs(path: Path) -> list[tuple[str, str]]:
    """Load (prediction, ground_truth) pairs from a JSONL eval log."""
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            pairs.append((row["pred"], row["gt"]))
    return pairs


def model_label_from_jsonl(path: Path) -> str:
    """Derive a stable model label from a JSONL filename."""
    name = path.name
    if name.endswith("_test.jsonl"):
        return name[: -len("_test.jsonl")]
    return path.stem


def precompute_line_stats(pairs: list[tuple[str, str]]) -> list[LineStats]:
    """Compute per-line CER, WER, and diacritic edit counts once."""
    stats: list[LineStats] = []
    for pred, gt in pairs:
        pred_norm = normalize_yoruba_text(pred)
        gt_norm = normalize_yoruba_text(gt)
        pred_diacs = [
            c
            for c in unicodedata.normalize("NFD", pred_norm)
            if unicodedata.combining(c)
        ]
        gt_diacs = [
            c
            for c in unicodedata.normalize("NFD", gt_norm)
            if unicodedata.combining(c)
        ]
        edits = editdistance.eval(pred_diacs, gt_diacs) if gt_diacs else 0
        stats.append(
            LineStats(
                cer=compute_cer(pred_norm, gt_norm),
                wer=compute_wer(pred_norm, gt_norm),
                der_edits=edits,
                gt_diac_count=len(gt_diacs),
            )
        )
    return stats


def aggregate_from_stats(sample: list[LineStats]) -> tuple[float, float, float]:
    """Return macro CER, macro WER, and micro corpus DER for a line sample."""
    n = len(sample)
    cer = sum(s.cer for s in sample) / n
    wer = sum(s.wer for s in sample) / n
    total_edits = sum(s.der_edits for s in sample if s.gt_diac_count > 0)
    total_gt_diacs = sum(s.gt_diac_count for s in sample if s.gt_diac_count > 0)
    der = total_edits / total_gt_diacs if total_gt_diacs else float("nan")
    return cer, wer, der


def bootstrap_metrics(
    stats: list[LineStats],
    *,
    n_bootstrap: int,
    seed: int,
    model_name: str = "",
) -> dict[str, np.ndarray]:
    """Line-level bootstrap distributions for CER, WER, and corpus DER."""
    n = len(stats)
    rng = np.random.default_rng(seed)
    cer_samples = np.empty(n_bootstrap, dtype=np.float64)
    wer_samples = np.empty(n_bootstrap, dtype=np.float64)
    der_samples = np.empty(n_bootstrap, dtype=np.float64)

    from tqdm import tqdm
    desc = f"Bootstrapping {model_name}" if model_name else "Bootstrapping"
    for b in tqdm(range(n_bootstrap), desc=desc, leave=False):
        idx = rng.integers(0, n, size=n)
        sample = [stats[i] for i in idx]
        cer, wer, der = aggregate_from_stats(sample)
        cer_samples[b] = cer
        wer_samples[b] = wer
        der_samples[b] = der

    return {"cer": cer_samples, "wer": wer_samples, "der": der_samples}


def percentile_ci(samples: np.ndarray, ci: float) -> tuple[float, float]:
    """Return percentile CI bounds."""
    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(samples, alpha))
    high = float(np.quantile(samples, 1.0 - alpha))
    return low, high


def write_metric_cis_csv(path: Path, rows: list[dict]) -> None:
    """Write per-model bootstrap CI table."""
    fieldnames = [
        "model",
        "metric",
        "point_estimate_pct",
        "ci_lower_pct",
        "ci_upper_pct",
        "n_lines",
        "n_bootstrap",
        "ci_level",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pairwise_csv(path: Path, rows: list[dict]) -> None:
    """Write aligned pairwise bootstrap comparison table."""
    fieldnames = [
        "model_a",
        "model_b",
        "metric",
        "mean_diff_pct",
        "ci_lower_pct",
        "ci_upper_pct",
        "p_a_lower_than_b",
        "n_bootstrap",
        "ci_level",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Entry point."""
    args = parse_args()
    jsonl_paths = args.jsonl
    if not jsonl_paths:
        jsonl_paths = [args.output_dir / name for name in DEFAULT_MODEL_JSONLS]

    jsonl_paths = [p for p in jsonl_paths if p.exists()]
    if not jsonl_paths:
        raise FileNotFoundError("No JSONL logs found — run evaluation first.")

    model_stats: dict[str, list[LineStats]] = {}
    model_pairs: dict[str, list[tuple[str, str]]] = {}
    for jsonl_path in jsonl_paths:
        label = model_label_from_jsonl(jsonl_path)
        pairs = load_jsonl_pairs(jsonl_path)
        model_pairs[label] = pairs
        model_stats[label] = precompute_line_stats(pairs)
        log.info("Loaded %d pairs for %s", len(pairs), label)

    n_lines = len(next(iter(model_pairs.values())))
    for label, pairs in model_pairs.items():
        if len(pairs) != n_lines:
            raise ValueError(
                f"{label}: expected {n_lines} rows, got {len(pairs)} — "
                "pairwise bootstrap requires aligned JSONL logs."
            )

    metric_rows: list[dict] = []
    bootstrap_store: dict[str, dict[str, np.ndarray]] = {}
    summary: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_lines": n_lines,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "ci_level": args.ci,
        "inter_annotator_agreement": {
            "available": False,
            "reason": (
                "Raw exports store only final corrected labels (corrected=yes); "
                "no per-annotator or pre/post-correction text fields."
            ),
        },
        "models": {},
    }

    for model, pairs in model_pairs.items():
        boot = bootstrap_metrics(
            model_stats[model],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            model_name=model,
        )
        bootstrap_store[model] = boot
        point_agg = aggregate_metrics(pairs)
        model_summary: dict = {"point_from_aggregate_metrics": point_agg}

        for metric in ("cer", "wer", "der"):
            samples = boot[metric]
            ci_low, ci_high = percentile_ci(samples, args.ci)
            point = point_agg[metric]
            metric_rows.append(
                {
                    "model": model,
                    "metric": metric.upper(),
                    "point_estimate_pct": f"{point * 100:.1f}",
                    "ci_lower_pct": f"{ci_low * 100:.1f}",
                    "ci_upper_pct": f"{ci_high * 100:.1f}",
                    "n_lines": n_lines,
                    "n_bootstrap": args.n_bootstrap,
                    "ci_level": args.ci,
                }
            )
            model_summary[metric] = {
                "point": point,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
            }
            log.info(
                "%s %s: %.1f%% [%.1f, %.1f]",
                model,
                metric.upper(),
                point * 100,
                ci_low * 100,
                ci_high * 100,
            )
        summary["models"][model] = model_summary

    pairwise_rows: list[dict] = []
    for model_a, model_b in PAIRWISE_COMPARISONS:
        if model_a not in bootstrap_store or model_b not in bootstrap_store:
            continue
        for metric in ("cer", "wer", "der"):
            diff = bootstrap_store[model_a][metric] - bootstrap_store[model_b][metric]
            mean_diff = float(np.mean(diff))
            ci_low, ci_high = percentile_ci(diff, args.ci)
            p_a_lower = float(np.mean(diff < 0))
            pairwise_rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric.upper(),
                    "mean_diff_pct": f"{mean_diff * 100:.1f}",
                    "ci_lower_pct": f"{ci_low * 100:.1f}",
                    "ci_upper_pct": f"{ci_high * 100:.1f}",
                    "p_a_lower_than_b": f"{p_a_lower:.3f}",
                    "n_bootstrap": args.n_bootstrap,
                    "ci_level": args.ci,
                }
            )
            log.info(
                "pair %s vs %s %s: diff=%.1f [%.1f, %.1f] P(a<b)=%.3f",
                model_a,
                model_b,
                metric.upper(),
                mean_diff * 100,
                ci_low * 100,
                ci_high * 100,
                p_a_lower,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metric_cis_csv(args.output_dir / "bootstrap_metric_cis.csv", metric_rows)
    write_pairwise_csv(
        args.output_dir / "bootstrap_pairwise_comparison.csv", pairwise_rows
    )
    summary["pairwise"] = pairwise_rows
    (args.output_dir / "bootstrap_metric_cis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote bootstrap CIs to %s", args.output_dir)


if __name__ == "__main__":
    main()
