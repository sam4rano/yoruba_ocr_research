"""
Generate the PaddleOCR fine-tuning YAML config with absolute paths resolved
from the current machine's project root.

Downloads the PP-OCRv3 English pretrained recognition weights if not present.

Usage:
    python scripts/03_generate_config.py
    python scripts/03_generate_config.py \
        --data-dir data/processed \
        --output-config configs/paddleocr_yoruba_rec.yml \
        --pretrained-dir experiments/baseline/pretrained \
        --epochs 100 \
        --batch-size 64 \
        --lr 0.0005
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml  # PyYAML

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# PP-OCRv3 English recognition pretrained weights
PRETRAINED_URL = (
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/"
    "en_PP-OCRv3_rec_train.tar"
)
PRETRAINED_DIR_NAME = "en_PP-OCRv3_rec_train"


def download_pretrained(pretrained_dir: Path) -> Path:
    """
    Download and extract PP-OCRv3 English recognition weights.

    Returns the path to the directory containing best_accuracy.pdparams.
    Skips download if the file already exists.
    """
    tar_path = pretrained_dir / "en_PP-OCRv3_rec_train.tar"
    model_dir = pretrained_dir / PRETRAINED_DIR_NAME

    if (model_dir / "best_accuracy.pdparams").exists():
        log.info("Pretrained weights already present at %s", model_dir)
        return model_dir

    pretrained_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading pretrained weights from %s ...", PRETRAINED_URL)
    urllib.request.urlretrieve(PRETRAINED_URL, tar_path)
    log.info("Extracting %s ...", tar_path)
    subprocess.run(
        ["tar", "-xf", str(tar_path), "-C", str(pretrained_dir)],
        check=True,
    )
    log.info("Pretrained weights extracted to %s", model_dir)
    return model_dir


def default_use_gpu() -> bool:
    """True only if Paddle was built with CUDA (typical Linux+GPU). False on macOS CPU."""
    try:
        import paddle

        return bool(paddle.device.is_compiled_with_cuda())
    except Exception:
        return False


def count_dict_chars(dict_path: Path) -> int:
    """Return the number of characters in the Yorùbá character dictionary."""
    if not dict_path.exists():
        raise FileNotFoundError(f"Character dictionary not found: {dict_path}")
    with dict_path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.rstrip("\n"))


def count_samples(label_path: Path) -> int:
    """Return the number of samples in a label file."""
    if not label_path.exists():
        return 500
    with label_path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def build_config(
    project_root: Path,
    data_dir: Path,
    pretrained_model_dir: Path,
    dict_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    use_gpu: bool,
) -> dict:
    """Construct the full PaddleOCR YAML config as a Python dict."""
    n_train = count_samples(data_dir / "labels" / "train.txt")
    eval_step = max(20, min(2000, (n_train // batch_size) * 2))
    
    return {
        "Global": {
            "use_gpu": use_gpu,
            "epoch_num": epochs,
            "log_smooth_window": 20,
            "print_batch_step": 10,
            "save_model_dir": str(output_dir / "finetuned"),
            "save_epoch_step": 10,
            "eval_batch_step": [0, eval_step],
            "cal_metric_during_train": True,
            "pretrained_model": str(pretrained_model_dir / "best_accuracy"),
            "checkpoints": None,
            "save_inference_dir": None,
            "use_visualdl": False,
            "character_dict_path": str(dict_path),
            "max_text_length": 160,
            "infer_mode": False,
            "use_space_char": True,
            "distributed": False,
            "save_res_path": str(
                project_root / "results" / "tables" / "rec_eval_output.txt"
            ),
        },
        "Optimizer": {
            "name": "Adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "lr": {
                "name": "MultiStepDecay",
                "learning_rate": lr,
                "milestones": [int(epochs * 0.6), int(epochs * 0.8)],
                "gamma": 0.1,
                "warmup_epoch": 5,
            },
            "regularizer": {"name": "L2", "factor": 3.0e-05},
        },
        "Architecture": {
            "model_type": "rec",
            "algorithm": "SVTR_LCNet",
            "Transform": None,
            "Backbone": {
                "name": "MobileNetV1Enhance",
                "scale": 0.5,
                "last_conv_stride": [1, 2],
                "last_pool_type": "avg",
                "last_pool_kernel_size": [2, 4],
            },
            "Neck": {
                "name": "SequenceEncoder",
                "encoder_type": "svtr",
                "dims": 64,
                "depth": 2,
                "hidden_dims": 120,
                "use_guide": True,
            },
            "Head": {"name": "CTCHead", "fc_decay": 0.00001},
        },
        "Loss": {"name": "CTCLoss"},
        "PostProcess": {"name": "CTCLabelDecode"},
        "Metric": {
            "name": "RecMetric",
            "main_indicator": "acc",
            "ignore_space": False,
        },
        "Train": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": str(data_dir),
                "label_file_list": [str(data_dir / "labels" / "train.txt")],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"RecAug": None},
                    {"CTCLabelEncode": None},
                    {"RecResizeImg": {"image_shape": [3, 48, 512]}},
                    {"KeepKeys": {"keep_keys": ["image", "label", "length"]}},
                ],
            },
            "loader": {
                "shuffle": True,
                "batch_size_per_card": batch_size,
                "drop_last": True,
                "num_workers": 4,
            },
        },
        "Eval": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": str(data_dir),
                "label_file_list": [str(data_dir / "labels" / "val.txt")],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"CTCLabelEncode": None},
                    {"RecResizeImg": {"image_shape": [3, 48, 512]}},
                    {"KeepKeys": {"keep_keys": ["image", "label", "length"]}},
                ],
            },
            "loader": {
                "shuffle": False,
                "drop_last": False,
                "batch_size_per_card": batch_size * 2,
                "num_workers": 4,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for config generation."""
    parser = argparse.ArgumentParser(
        description="Generate PaddleOCR fine-tuning config for Yorùbá OCR."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Set Global.use_gpu=false (required for macOS / CPU-only Paddle).",
    )
    parser.add_argument(
        "--force-gpu",
        action="store_true",
        help="Set Global.use_gpu=true for CUDA training (Linux).",
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        type=Path,
        default=Path("data/processed"),
        help="Consolidated dataset root.",
    )
    parser.add_argument(
        "--output-config",
        "--out_config",
        "--output_config",
        dest="output_config",
        type=Path,
        default=Path("configs/paddleocr_yoruba_rec.yml"),
        help="Path to write the generated YAML config.",
    )
    parser.add_argument(
        "--pretrained-dir",
        "--pretrained_dir",
        type=Path,
        default=Path("experiments/baseline/pretrained"),
        help="Directory to store/find pretrained weights.",
    )
    parser.add_argument(
        "--experiments-dir",
        "--experiments_dir",
        type=Path,
        default=Path("experiments"),
        help="Root for experiment output directories.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=64,
        help="Batch size per GPU card during training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--skip-download",
        "--skip_download",
        action="store_true",
        help="Skip pretrained weight download even if weights are absent.",
    )
    parser.add_argument(
        "--download-pretrained",
        "--download_pretrained",
        action="store_true",
        help="Optional. Weights are downloaded by default; use only to be explicit.",
    )
    parser.add_argument(
        "--log-file",
        "--log_file",
        type=Path,
        default=Path("results/tables/config_generation.json"),
        help="JSON log for this run.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate and write the fine-tuning config."""
    args = parse_args()
    project_root = Path(".").resolve()

    dict_path = args.data_dir / "dictionary" / "yoruba_char_dict.txt"
    try:
        n_chars = count_dict_chars(dict_path)
        log.info("Character dictionary: %d chars at %s", n_chars, dict_path)
    except FileNotFoundError as exc:
        log.error("%s — run 01_consolidate_data.py first.", exc)
        sys.exit(1)

    # Download pretrained weights unless instructed to skip
    if args.skip_download:
        pretrained_model_dir = args.pretrained_dir / PRETRAINED_DIR_NAME
        log.warning("Skipping download. Expecting weights at %s", pretrained_model_dir)
    else:
        try:
            pretrained_model_dir = download_pretrained(args.pretrained_dir)
        except (urllib.error.URLError, OSError) as exc:
            log.error("Pretrained download failed: %s", exc)
            log.error(
                "If weights are already local, re-run with --skip-download "
                "(expects %s/best_accuracy.pdparams).",
                args.pretrained_dir / PRETRAINED_DIR_NAME,
            )
            sys.exit(1)

    if args.cpu and args.force_gpu:
        log.error("Use only one of --cpu or --force-gpu.")
        sys.exit(1)
    if args.cpu:
        use_gpu = False
    elif args.force_gpu:
        use_gpu = True
    else:
        use_gpu = default_use_gpu()
    log.info("Global.use_gpu = %s", use_gpu)

    config = build_config(
        project_root=project_root,
        data_dir=args.data_dir.resolve(),
        pretrained_model_dir=pretrained_model_dir.resolve(),
        dict_path=dict_path.resolve(),
        output_dir=args.experiments_dir.resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_gpu=use_gpu,
    )

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as fh:
        yaml.dump(config, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)
    log.info("Config written to %s", args.output_config)

    # Persist run metadata
    report = {
        "stage": "generate_config",
        "config_path": str(args.output_config),
        "dict_size": n_chars,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "pretrained_model_dir": str(pretrained_model_dir),
        "use_gpu": use_gpu,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Next step — run fine-tuning from the PaddleOCR repo root:")
    print(
        f"  python tools/train.py -c {args.output_config.resolve()}"
    )
    print("Or use scripts/04_train_paddleocr.py for automated logging.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
