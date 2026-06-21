"""
End-to-end network and hub connectivity checks for the Yorùbá OCR pipeline.


Validates reachability and config fetch for Hugging Face models and Paddle
pretrained weights — without running full GPU eval unless ``--live`` is set.


Usage:
    python scripts/30_e2e_network_check.py
    python scripts/30_e2e_network_check.py --live --max-samples 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get("PYTHON", sys.executable)
SHELL_ENV = {**os.environ, "PYTHON": PY, "PROJECT_ROOT": str(ROOT)}

HF_MODELS = (
    ("paddleocr_vl16", "PaddlePaddle/PaddleOCR-VL-1.6", True),
    ("glm_ocr", "zai-org/GLM-OCR", True),
)

PADDLE_PRETRAINED_URL = (
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar"
)


class SkipCheck(Exception):
    """Environment not suitable for a live eval (not a network failure)."""


def step_result(name: str, fn) -> dict:
    """Run one check and return a serialisable result dict."""
    try:
        detail = fn()
        print(f"OK  {name}")
        return {"name": name, "status": "ok", "detail": detail or {}}
    except SkipCheck as exc:
        print(f"SKIP {name}: {exc}")
        return {"name": name, "status": "skip", "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {name}: {exc}")
        return {"name": name, "status": "fail", "error": str(exc)}


def check_hf_hub_api() -> dict:
    """Ping Hugging Face Hub model metadata API."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info("PaddlePaddle/PaddleOCR-VL-1.6")
    return {"model_id": info.id}


def check_hf_model_config(model_id: str, trust_remote_code: bool) -> dict:
    """Fetch ``AutoConfig`` without downloading full weights."""
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        return {"model_id": model_id, "config_class": type(cfg).__name__}
    except ValueError as exc:
        if "does not recognize this architecture" in str(exc) or "model_type" in str(exc):
            raise SkipCheck(
                f"Model architecture not recognized by current transformers version: {exc}. "
                f"Upgrade transformers or install from source."
            )
        raise


def check_paddle_pretrained_url() -> dict:
    """HEAD/GET probe for PP-OCRv3 English pretrained tarball."""
    req = urllib.request.Request(PADDLE_PRETRAINED_URL, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            code = resp.status
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 405):
            with urllib.request.urlopen(PADDLE_PRETRAINED_URL, timeout=60) as resp:
                code = resp.status
        else:
            raise
    return {"url": PADDLE_PRETRAINED_URL, "http_status": code}


def check_vl_trust_remote_defaults() -> dict:
    """Ensure PaddleOCR-VL trust_remote_code defaults are enabled."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from paddle_vl_shared import (  # noqa: E402
        hf_trust_remote_code_model,
        hf_trust_remote_code_processor,
    )

    os.environ.pop("HF_TRUST_REMOTE_CODE", None)
    model_ok = hf_trust_remote_code_model()
    proc_ok = hf_trust_remote_code_processor()
    if not model_ok or not proc_ok:
        raise RuntimeError(
            f"HF trust defaults wrong: model={model_ok} processor={proc_ok}"
        )
    return {"model_default": model_ok, "processor_default": proc_ok}




def _require_transformers_min(min_major: int = 5) -> None:
    """Raise SkipCheck if installed transformers is below ``min_major``."""
    import transformers

    major = int(transformers.__version__.split(".")[0])
    if major < min_major:
        raise SkipCheck(
            f"transformers>={min_major} required for live VL-1.6 eval "
            f"(found {transformers.__version__}; Colab: pip install -U 'transformers>=5')"
        )


def _require_cuda_for_live() -> None:
    """Raise SkipCheck when no CUDA device is available for large VLM loads."""
    try:
        import torch

        if not torch.cuda.is_available():
            raise SkipCheck("CUDA not available for live VL-1.6 eval")
    except ImportError as exc:
        raise SkipCheck("torch not installed for live VL-1.6 eval") from exc


def check_live_vl16_load(max_samples: int) -> dict:
    """Run one-sample VL-1.6 eval (GPU + large download)."""
    _require_transformers_min(5)
    _require_cuda_for_live()
    cmd = [
        PY,
        "scripts/15_baseline_paddleocr_vl16.py",
        "--split",
        "test",
        "--max-samples",
        str(max_samples),
        "--quantize-4bit",
        "--results-csv",
        "results/tables/metrics_e2e_scratch.csv",
        "--per-sample-log",
        "results/tables/paddleocr_vl16_e2e_scratch.jsonl",
    ]
    subprocess.check_call(cmd, cwd=ROOT, env={**SHELL_ENV, "HF_TRUST_REMOTE_CODE": "1"})
    return {"max_samples": max_samples, "metrics_csv": "results/tables/metrics_e2e_scratch.csv"}




def check_live_glm_ocr(max_samples: int) -> dict:
    """Run capped GLM-OCR zero-shot eval."""
    _require_cuda_for_live()
    from transformers import AutoConfig
    try:
        AutoConfig.from_pretrained("zai-org/GLM-OCR", trust_remote_code=True)
    except ValueError as exc:
        raise SkipCheck(f"GLM-OCR not supported by current transformers version: {exc}")

    cmd = [
        PY,
        "scripts/16_baseline_glm_ocr.py",
        "--split",
        "test",
        "--max-samples",
        str(max_samples),
        "--quantize-4bit",
        "--results-csv",
        "results/tables/metrics_e2e_scratch.csv",
        "--per-sample-log",
        "results/tables/glm_ocr_e2e_scratch.jsonl",
    ]
    subprocess.check_call(cmd, cwd=ROOT, env=SHELL_ENV)
    return {"max_samples": max_samples, "metrics_csv": "results/tables/metrics_e2e_scratch.csv"}


def main() -> None:
    """Run network e2e checks and write JSON report."""
    parser = argparse.ArgumentParser(description="Network e2e checks for OCR pipeline.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/tables/e2e_network_check.json"),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run capped live model evals (GPU + downloads; slow).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Sample cap for --live evals.",
    )
    args = parser.parse_args()

    results: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": PY,
        "steps": [],
        "ok": True,
    }

    checks = [
        ("huggingface_hub_api", check_hf_hub_api),
        ("paddle_pretrained_url", check_paddle_pretrained_url),
        ("vl_trust_remote_code_defaults", check_vl_trust_remote_defaults),
    ]
    for label, model_id, trust in HF_MODELS:
        checks.append(
            (
                f"hf_config_{label}",
                lambda mid=model_id, t=trust: check_hf_model_config(mid, t),
            )
        )

    def check_transformers_version() -> dict:
        import transformers

        major = int(transformers.__version__.split(".")[0])
        return {
            "version": transformers.__version__,
            "vl16_live_ready": major >= 5,
        }

    checks.append(("transformers_version", check_transformers_version))

    for name, fn in checks:
        row = step_result(name, fn)
        results["steps"].append(row)
        if row["status"] == "fail":
            results["ok"] = False

    if args.live:
        live_checks = [
            ("live_vl16", lambda: check_live_vl16_load(args.max_samples)),
            ("live_glm_ocr", lambda: check_live_glm_ocr(args.max_samples)),
        ]
        for name, fn in live_checks:
            row = step_result(name, fn)
            results["steps"].append(row)
            if row["status"] == "fail":
                results["ok"] = False

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport: {args.report}")
    print("E2E NETWORK:", "PASSED" if results["ok"] else "FAILED")
    if not results["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
