"""
Evaluate a PaddleOCR recognition model on a labelled split and compute
CER, WER, and DER (Diacritic Error Rate — novel contribution of this paper).

CER: character-level edit distance / ground-truth length (NFC-normalised)
WER: word-level edit distance / ground-truth word count (NFC-normalised)
DER: edit distance between NFD-extracted combining diacritics / gt diacritic count

Both baseline (pretrained English) and fine-tuned models can be evaluated
with this script by pointing --model-dir at the appropriate weights directory.

Usage:
    # Fine-tuned model evaluation
    python scripts/05_evaluate.py \
        --model-dir experiments/finetuned/best_accuracy \
        --data-dir data/processed \
        --split test \
        --model-name finetuned_paddleocr

    # Pretrained English baseline evaluation
    python scripts/05_evaluate.py \
        --model-dir experiments/baseline/pretrained/en_PP-OCRv3_rec_train \
        --data-dir data/processed \
        --split test \
        --model-name baseline_english_pretrained
"""

from __future__ import annotations

import argparse
import logging
import unicodedata
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_utils import (  # noqa: E402
    load_test_pairs,
    aggregate_metrics,
    save_results,
)


# ---------------------------------------------------------------------------
# Checkpoint integrity
# ---------------------------------------------------------------------------

def _classify_param(key: str) -> str:
    """Bucket a parameter name into 'head', 'backbone', or 'other'.

    PaddleOCR recognition models namespace parameters as
    ``backbone.*`` / ``neck.*`` / ``head.*``. The CTC head (the part that
    maps encoder features to the output vocabulary) is what silently goes
    random when the dictionary size is swapped at eval time, so we flag any
    key under ``head`` for the strict check.
    """
    k = key.lower()
    if k.startswith("head") or ".head." in k or ".ctc_head." in k or "ctc_encoder" in k:
        return "head"
    if k.startswith("backbone") or k.startswith("neck"):
        return "backbone"
    return "other"


def inspect_checkpoint_restoration(ckpt_prefix: Path, model) -> dict:
    """Compare a Paddle checkpoint against a freshly built model.

    Runs **before** ``ppocr.utils.save_load.load_model``, so we can abort
    on shape mismatches that ``load_model`` otherwise only logs as a
    warning (historically this is how "phantom baselines" ended up in
    ``metrics.csv`` — the English encoder restored but the CTC head stayed
    random after we swapped in the Yorùbá dictionary).

    Returns a report dict with per-key status buckets. ``missing`` and
    ``shape_mismatch`` entries on head weights are the blocking signal.
    """
    import paddle  # type: ignore  # local import: only when running eval

    params_path = ckpt_prefix.with_suffix(".pdparams")
    if not params_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {params_path}")

    ckpt = paddle.load(str(params_path))
    model_state = model.state_dict()

    restored: list[str] = []
    missing: list[str] = []
    shape_mismatch: list[dict] = []
    extra: list[str] = []

    for key, val in model_state.items():
        if key not in ckpt:
            missing.append(key)
            continue
        src = ckpt[key]
        if list(src.shape) != list(val.shape):
            shape_mismatch.append(
                {"key": key, "model_shape": list(val.shape), "ckpt_shape": list(src.shape)}
            )
            continue
        restored.append(key)

    extra = [k for k in ckpt.keys() if k not in model_state]

    def _by_component(keys: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {"head": 0, "backbone": 0, "other": 0}
        for k in keys:
            counts[_classify_param(k)] += 1
        return counts

    report = {
        "ckpt_prefix": str(ckpt_prefix),
        "n_model_params": len(model_state),
        "n_ckpt_params": len(ckpt),
        "n_restored": len(restored),
        "n_missing": len(missing),
        "n_shape_mismatch": len(shape_mismatch),
        "n_extra_in_ckpt": len(extra),
        "missing_by_component": _by_component(missing),
        "shape_mismatch_by_component": _by_component([m["key"] for m in shape_mismatch]),
        "missing_sample": missing[:10],
        "shape_mismatch_sample": shape_mismatch[:10],
    }
    return report


class PhantomCheckpointError(RuntimeError):
    """Raised when a checkpoint cannot faithfully restore the model it was
    asked to evaluate — i.e. the metrics we would record would measure a
    randomly-initialised head rather than the trained model.
    """


def enforce_checkpoint_integrity(
    report: dict,
    *,
    allow_head_reinit: bool = False,
) -> None:
    """Raise ``PhantomCheckpointError`` when a head weight cannot be restored.

    The default policy (strict) is appropriate for evaluation: an unrestored
    head parameter means the CTC projection is random and any metric
    produced is meaningless. Pass ``allow_head_reinit=True`` only when
    deliberately evaluating a fresh head (e.g. a dictionary ablation where
    we *expect* reinit and want to record CER ≈ 1.0 as the zero-skill
    lower bound).
    """
    head_missing = report["missing_by_component"].get("head", 0)
    head_shape_mismatch = report["shape_mismatch_by_component"].get("head", 0)
    head_bad = head_missing + head_shape_mismatch

    if head_bad == 0:
        return

    msg_lines = [
        "PHANTOM CHECKPOINT DETECTED — refusing to record eval metrics.",
        f"  checkpoint : {report['ckpt_prefix']}.pdparams",
        f"  head weights not restored: {head_bad} "
        f"(missing={head_missing}, shape_mismatch={head_shape_mismatch})",
        "  this model's CTC head would be random; CER/WER/DER would be meaningless.",
    ]
    if report["shape_mismatch_sample"]:
        msg_lines.append("  first mismatches:")
        for m in report["shape_mismatch_sample"][:5]:
            msg_lines.append(
                f"    - {m['key']}: model={m['model_shape']} ckpt={m['ckpt_shape']}"
            )
    msg_lines.append(
        "  fix: retrain with the matching dictionary, or pass "
        "--allow-head-reinit if this is a deliberate zero-skill/ablation run."
    )
    if allow_head_reinit:
        log.warning("\n".join(msg_lines))
        log.warning("--allow-head-reinit set; proceeding with random head.")
        return

    raise PhantomCheckpointError("\n".join(msg_lines))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    pairs: list[tuple[Path, str]],
    model_dir: Path,
    dict_path: Path,
    use_gpu: bool,
    paddle_dir: Path,
    rec_config: Path,
    allow_head_reinit: bool = False,
) -> tuple[list[tuple[str, str]], dict]:
    """
    Run PaddleOCR recognition-only inference on a list of image paths.

    Returns ``(pairs, integrity_report)`` where ``integrity_report`` is the
    dict produced by :func:`inspect_checkpoint_restoration` and should be
    persisted next to the metrics row so phantom baselines are visible in
    the audit trail.
    """
    # PaddleOCR>=3.x (pip) no longer supports passing training checkpoints and
    # custom dictionaries via the legacy `PaddleOCR(..., use_gpu=..., rec_model_dir=...)`.
    # For reproducible evaluation, we load the trained checkpoint using the
    # PaddleOCR training code (`ppocr.*`) from the cloned repo.

    paddle_dir = paddle_dir.resolve()
    if not (paddle_dir / "ppocr").exists():
        raise FileNotFoundError(
            f"PaddleOCR repo not found at {paddle_dir}. Expected to find {paddle_dir / 'ppocr'}."
        )

    import sys as _sys

    # Ensure `ppocr` can be imported.
    if str(paddle_dir) not in _sys.path:
        _sys.path.insert(0, str(paddle_dir))

    import yaml  # type: ignore
    import paddle  # type: ignore
    from ppocr.data import create_operators, transform  # type: ignore
    from ppocr.modeling.architectures import build_model  # type: ignore
    from ppocr.postprocess import build_post_process  # type: ignore
    from ppocr.utils.save_load import load_model  # type: ignore

    def resolve_checkpoint_prefix(path: Path) -> Path:
        # Accept:
        # - directory containing best_accuracy.pdparams
        # - prefix path without extension
        # - explicit .pdparams file
        if path.is_dir():
            cand = path / "best_accuracy"
            if (cand.with_suffix(".pdparams")).exists():
                return cand
            # also allow "latest" naming
            cand = path / "latest"
            if (cand.with_suffix(".pdparams")).exists():
                return cand
            raise FileNotFoundError(
                f"Could not find checkpoint prefix in directory {path} (expected best_accuracy.pdparams or latest.pdparams)."
            )

        if path.suffix == ".pdparams":
            return Path(str(path)[: -len(".pdparams")])
        return path

    ckpt_prefix = resolve_checkpoint_prefix(model_dir)
    if not (ckpt_prefix.with_suffix(".pdparams")).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_prefix}.pdparams")

    with rec_config.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # Force dictionary to our consolidated Yorùbá dict for decoding.
    config["Global"]["character_dict_path"] = str(dict_path.resolve())

    # Ensure the checkpoint is used (not pretrained model).
    config["Global"]["checkpoints"] = str(ckpt_prefix.resolve())
    config["Global"]["pretrained_model"] = None
    config["Global"]["infer_mode"] = True

    # Device selection for inference.
    device = "gpu" if use_gpu else "cpu"
    paddle.set_device(device)

    post_process_class = build_post_process(config["PostProcess"], config["Global"])

    # Ensure out_channels matches dict size for CTC head.
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if (
            config.get("Architecture", {})
            .get("Head", {})
            .get("name", "")
            .lower()
            .startswith("ctc")
        ):
            config["Architecture"]["Head"]["out_channels"] = char_num

    model = build_model(config["Architecture"])

    integrity_report = inspect_checkpoint_restoration(ckpt_prefix, model)
    log.info(
        "Checkpoint integrity: restored=%d/%d  missing=%d  shape_mismatch=%d",
        integrity_report["n_restored"],
        integrity_report["n_model_params"],
        integrity_report["n_missing"],
        integrity_report["n_shape_mismatch"],
    )
    enforce_checkpoint_integrity(
        integrity_report, allow_head_reinit=allow_head_reinit
    )

    load_model(config, model)
    model.eval()

    # Build preprocessing ops from Eval transforms; drop label-only/aug ops.
    transforms_cfg = config["Eval"]["dataset"]["transforms"]
    drop_ops = {"RecAug", "CTCLabelEncode", "KeepKeys"}
    ops_cfg = [op for op in transforms_cfg if list(op.keys())[0] not in drop_ops]
    ops = create_operators(ops_cfg, config["Global"])

    results: list[tuple[str, str]] = []
    for img_path, gt in pairs:
        img_bytes = img_path.read_bytes()
        data = {"image": img_bytes}
        batch = transform(data, ops)
        if batch is None:
            pred_text = ""
        else:
            import numpy as np  # type: ignore

            images = np.expand_dims(batch["image"], axis=0)
            images = paddle.to_tensor(images)
            preds = model(images)
            post_result = post_process_class(preds)
            # Typical format: [[text, score]]
            if isinstance(post_result, list) and post_result and isinstance(
                post_result[0], (list, tuple)
            ):
                pred_text = str(post_result[0][0])
            else:
                pred_text = ""

        results.append((pred_text, gt))

    return results, integrity_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate PaddleOCR model on Yorùbá OCR test set."
    )
    parser.add_argument(
        "--model-dir",
        "--model_dir",
        dest="model_dir",
        type=Path,
        required=True,
        help="Checkpoint prefix, .pdparams file, or directory containing best_accuracy.pdparams.",
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        type=Path,
        default=Path("data/processed"),
        help="Consolidated dataset root.",
    )
    parser.add_argument(
        "--dict-path",
        "--dict_path",
        dest="dict_path",
        type=Path,
        default=None,
        help="Character dictionary path. Defaults to data-dir/dictionary/yoruba_char_dict.txt.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--model-name",
        "--model_name",
        dest="model_name",
        type=str,
        default="model",
        help="Label for this model in the results table.",
    )
    parser.add_argument(
        "--use-gpu",
        "--use_gpu",
        dest="use_gpu",
        action="store_true",
        default=False,
        help="Run inference on GPU.",
    )
    parser.add_argument(
        "--paddle-dir",
        "--paddle_dir",
        dest="paddle_dir",
        type=Path,
        default=Path("PaddleOCR"),
        help="Path to the cloned PaddleOCR repository (for `ppocr.*` evaluation).",
    )
    parser.add_argument(
        "--rec-config",
        "--rec_config",
        dest="rec_config",
        type=Path,
        default=None,
        help=(
            "PaddleOCR YAML config used for training (must match checkpoint family). "
            "Defaults to experiments/finetuned/config.yml if present, else "
            "configs/paddleocr_yoruba_rec.yml (same as scripts/03_generate_config.py and "
            "04_train_paddleocr.py). Override only if you trained with a custom YAML; "
            "the baseline and fine-tuned runs MUST use the same --rec-config."
        ),
    )
    parser.add_argument(
        "--results-csv",
        "--results_csv",
        dest="results_csv",
        type=Path,
        default=Path("results/tables/metrics.csv"),
        help="Shared results table (CSV) — appended on each run.",
    )
    parser.add_argument(
        "--per-sample-log",
        "--per_sample_log",
        dest="per_sample_log",
        type=Path,
        default=None,
        help="JSONL file for per-sample predictions. Defaults to results/tables/{model_name}_{split}.jsonl.",
    )
    parser.add_argument(
        "--allow-head-reinit",
        "--allow_head_reinit",
        dest="allow_head_reinit",
        action="store_true",
        default=False,
        help=(
            "Proceed even when the CTC head weights cannot be restored from the "
            "checkpoint (i.e. a random head). Use ONLY for deliberate zero-skill "
            "lower-bound runs such as the dictionary ablation; never for baselines "
            "that will be presented as trained models."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation pipeline end-to-end."""
    args = parse_args()

    dict_path = args.dict_path or (
        args.data_dir / "dictionary" / "yoruba_char_dict.txt"
    )
    rec_config = args.rec_config
    if rec_config is None:
        cand = Path("experiments/finetuned/config.yml")
        # Align with 03_generate_config / 04_train default (PP-OCRv3 English pretrained).
        # Mixing an arbitrary YAML with a mismatched checkpoint silently re-initialises
        # weights and makes baseline vs fine-tuned metrics meaningless.
        rec_config = cand if cand.exists() else Path("configs/paddleocr_yoruba_rec.yml")
    if not rec_config.exists():
        raise FileNotFoundError(f"Config not found: {rec_config}")
    per_sample_log = args.per_sample_log or (
        Path("results/tables") / f"{args.model_name}_{args.split}.jsonl"
    )

    log.info("Loading %s split from %s ...", args.split, args.data_dir)
    pairs = load_test_pairs(args.data_dir, args.split)
    log.info("Loaded %d labelled examples.", len(pairs))

    pred_pairs, integrity_report = run_inference(
        pairs,
        args.model_dir,
        dict_path,
        args.use_gpu,
        paddle_dir=args.paddle_dir,
        rec_config=rec_config,
        allow_head_reinit=args.allow_head_reinit,
    )

    metrics = aggregate_metrics(pred_pairs)
    log.info(
        "Results — CER: %.4f  WER: %.4f  DER: %.4f  (n=%d)",
        metrics["cer"],
        metrics["wer"],
        metrics["der"],
        metrics["n"],
    )

    save_results(
        metrics,
        model_name=args.model_name,
        split=args.split,
        csv_path=args.results_csv,
        jsonl_path=per_sample_log,
        provenance={
            "script": "scripts/05_evaluate.py",
            "model_dir": str(args.model_dir),
            "dict_path": str(dict_path),
            "rec_config": str(rec_config),
            "split": args.split,
            "allow_head_reinit": args.allow_head_reinit,
            "checkpoint_integrity": integrity_report,
        },
    )
    log.info("Results appended to %s", args.results_csv)
    log.info("Per-sample log written to %s", per_sample_log)


if __name__ == "__main__":
    main()
