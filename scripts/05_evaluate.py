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
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    pairs: list[tuple[Path, str]],
    model_dir: Path,
    dict_path: Path,
    use_gpu: bool,
    paddle_dir: Path,
    rec_config: Path,
) -> list[tuple[str, str]]:
    """
    Run PaddleOCR recognition-only inference on a list of image paths.

    Returns list of (prediction, ground_truth) strings.
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

    return results


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
            "04_train_paddleocr.py). If you trained with configs/paddleocr_yoruba_rec_final.yml "
            "(PP-OCRv4 pretrained), pass that path explicitly for both baseline and fine-tuned eval."
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
        # Using paddleocr_yoruba_rec_final.yml without the matching v4 checkpoint + train
        # run mis-loads weights and makes baseline vs fine-tuned metrics meaningless.
        rec_config = cand if cand.exists() else Path("configs/paddleocr_yoruba_rec.yml")
    if not rec_config.exists():
        raise FileNotFoundError(f"Config not found: {rec_config}")
    per_sample_log = args.per_sample_log or (
        Path("results/tables") / f"{args.model_name}_{args.split}.jsonl"
    )

    log.info("Loading %s split from %s ...", args.split, args.data_dir)
    pairs = load_test_pairs(args.data_dir, args.split)
    log.info("Loaded %d labelled examples.", len(pairs))

    pred_pairs = run_inference(
        pairs,
        args.model_dir,
        dict_path,
        args.use_gpu,
        paddle_dir=args.paddle_dir,
        rec_config=rec_config,
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
    )
    log.info("Results appended to %s", args.results_csv)
    log.info("Per-sample log written to %s", per_sample_log)


if __name__ == "__main__":
    main()
