"""
Ablation of corpus DER under alternative diacritic-universe definitions.

Recomputes micro-averaged DER and der_insertion_rate from existing test JSONL
logs without re-running model inference. Universes:

  * combining_marks — U+0300, U+0301, U+0323 (reporting default)
  * tone_only — U+0300, U+0301 (excludes subdot)
  * all_combining — any Unicode combining character in NFD
  * marked_grapheme — NFC codepoints whose NFD contains a mark in the standard set

Usage:
    python scripts/18_der_universe_ablation.py
    python scripts/18_der_universe_ablation.py \\
        --jsonl results/tables/paddleocr_vl15_lora_finetuned_test.jsonl
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

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from evaluate_utils import normalize_yoruba_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

U_STANDARD = frozenset({"\u0300", "\u0301", "\u0323"})
U_TONE_ONLY = frozenset({"\u0300", "\u0301"})

DEFAULT_MODEL_JSONLS = [
    "paddleocr_vl15_lora_finetuned_test.jsonl",
    "ablation_data_size_100pct_test_test.jsonl",
    "paddleocr_vl15_zero_shot_test.jsonl",
    "surya_v2_zero_shot_test.jsonl",
    "surya_finetuned_test.jsonl",
    "trocr_large_printed_zero_shot_test.jsonl",
    "trocr_large_printed_finetuned_test.jsonl",
]


@dataclass(frozen=True)
class UniverseSpec:
    """Definition of how diacritic tokens are extracted from a string."""

    universe_id: str
    description: str
    mode: str  # "combining" | "grapheme"
    codepoints: frozenset[str] | None  # None => any combining mark


UNIVERSES: tuple[UniverseSpec, ...] = (
    UniverseSpec(
        "combining_marks",
        "Combining grave, acute, and dot below (reporting default)",
        "combining",
        U_STANDARD,
    ),
    UniverseSpec(
        "tone_only",
        "Combining grave and acute only (subdot excluded)",
        "combining",
        U_TONE_ONLY,
    ),
    UniverseSpec(
        "all_combining",
        "All Unicode combining characters in NFD",
        "combining",
        None,
    ),
    UniverseSpec(
        "marked_grapheme",
        "NFC tonographs with any standard combining mark",
        "grapheme",
        U_STANDARD,
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Recompute corpus DER under alternative diacritic universes."
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
        help="Per-sample JSONL log (repeatable). Defaults to main test runs.",
    )
    return parser.parse_args()


def extract_tokens(text: str, spec: UniverseSpec) -> list[str]:
    """
    Extract a diacritic token sequence from ``text`` under ``spec``.

    For ``combining`` mode, returns a list of combining codepoints in NFD order.
    For ``grapheme`` mode, returns NFC characters that carry at least one mark
    in the configured set.
    """
    normalized = normalize_yoruba_text(text)
    if spec.mode == "combining":
        nfd = unicodedata.normalize("NFD", normalized)
        if spec.codepoints is None:
            return [c for c in nfd if unicodedata.combining(c)]
        return [c for c in nfd if c in spec.codepoints]

    tokens: list[str] = []
    for char in normalized:
        marks = [
            c
            for c in unicodedata.normalize("NFD", char)
            if unicodedata.combining(c)
        ]
        if not marks:
            continue
        if spec.codepoints is None or any(m in spec.codepoints for m in marks):
            tokens.append(char)
    return tokens


def aggregate_der(pairs: list[tuple[str, str]], spec: UniverseSpec) -> dict:
    """
    Compute corpus DER and der_insertion_rate for ``pairs`` under ``spec``.

    Returns dict with keys n, der, der_n, der_insertion_rate, n_zero_diac_gt,
    total_gt_tokens, total_pred_tokens_on_zero_gt.
    """
    total_edits = 0
    total_gt_tokens = 0
    der_n_samples = 0

    total_insertions = 0
    total_gt_chars_nodiac = 0
    n_zero_diac_gt = 0

    for pred, gt in pairs:
        pred_tokens = extract_tokens(pred, spec)
        gt_tokens = extract_tokens(gt, spec)

        if gt_tokens:
            edits = editdistance.eval(pred_tokens, gt_tokens)
            total_edits += edits
            total_gt_tokens += len(gt_tokens)
            der_n_samples += 1
        else:
            n_zero_diac_gt += 1
            total_insertions += len(pred_tokens)
            total_gt_chars_nodiac += len(normalize_yoruba_text(gt))

    n = len(pairs)
    der = round(total_edits / total_gt_tokens, 4) if total_gt_tokens else None
    der_insertion_rate = (
        round(total_insertions / total_gt_chars_nodiac, 4)
        if total_gt_chars_nodiac
        else None
    )

    return {
        "n": n,
        "der": der,
        "der_n": der_n_samples,
        "der_insertion_rate": der_insertion_rate,
        "n_zero_diac_gt": n_zero_diac_gt,
        "total_gt_tokens": total_gt_tokens,
        "total_pred_tokens_on_zero_gt": total_insertions,
    }


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


def write_ablation_csv(path: Path, rows: list[dict]) -> None:
    """Write the universe ablation table."""
    fieldnames = [
        "model",
        "universe_id",
        "description",
        "n",
        "der_pct",
        "der_n",
        "der_insertion_rate",
        "n_zero_diac_gt",
        "total_gt_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            der = row["der"]
            writer.writerow(
                {
                    "model": row["model"],
                    "universe_id": row["universe_id"],
                    "description": row["description"],
                    "n": row["n"],
                    "der_pct": f"{der * 100:.1f}" if der is not None else "",
                    "der_n": row["der_n"],
                    "der_insertion_rate": row["der_insertion_rate"],
                    "n_zero_diac_gt": row["n_zero_diac_gt"],
                    "total_gt_tokens": row["total_gt_tokens"],
                }
            )


def write_insertion_csv(path: Path, rows: list[dict]) -> None:
    """Write zero-diacritic-GT insertion summary (combining_marks universe)."""
    fieldnames = [
        "model",
        "n_zero_diac_gt",
        "total_gt_chars",
        "spurious_marks_predicted",
        "der_insertion_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    """Entry point."""
    args = parse_args()
    jsonl_paths = args.jsonl
    if not jsonl_paths:
        jsonl_paths = [args.output_dir / name for name in DEFAULT_MODEL_JSONLS]

    jsonl_paths = [p for p in jsonl_paths if p.exists()]
    if not jsonl_paths:
        raise FileNotFoundError("No JSONL logs found — run evaluation first.")

    ablation_rows: list[dict] = []
    insertion_rows: list[dict] = []
    summary: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "universes": [
            {
                "universe_id": spec.universe_id,
                "description": spec.description,
                "mode": spec.mode,
            }
            for spec in UNIVERSES
        ],
        "models": {},
    }

    for jsonl_path in jsonl_paths:
        model = model_label_from_jsonl(jsonl_path)
        pairs = load_jsonl_pairs(jsonl_path)
        model_summary: dict = {}

        for spec in UNIVERSES:
            metrics = aggregate_der(pairs, spec)
            record = {
                "model": model,
                "universe_id": spec.universe_id,
                "description": spec.description,
                **metrics,
            }
            ablation_rows.append(record)
            model_summary[spec.universe_id] = metrics
            log.info(
                "%s / %s: DER=%s der_n=%d insertion=%s",
                model,
                spec.universe_id,
                metrics["der"],
                metrics["der_n"],
                metrics["der_insertion_rate"],
            )

        default = model_summary["combining_marks"]
        gt_chars_zero = sum(
            len(normalize_yoruba_text(gt))
            for _, gt in pairs
            if not extract_tokens(gt, UNIVERSES[0])
        )
        insertion_rows.append(
            {
                "model": model,
                "n_zero_diac_gt": default["n_zero_diac_gt"],
                "total_gt_chars": gt_chars_zero,
                "spurious_marks_predicted": default["total_pred_tokens_on_zero_gt"],
                "der_insertion_rate": default["der_insertion_rate"],
            }
        )
        summary["models"][model] = model_summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_ablation_csv(args.output_dir / "der_universe_ablation.csv", ablation_rows)
    write_insertion_csv(
        args.output_dir / "der_zero_diac_insertion.csv", insertion_rows
    )
    (args.output_dir / "der_universe_ablation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote DER universe ablation to %s", args.output_dir)


if __name__ == "__main__":
    main()
