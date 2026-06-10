"""
Upload fine-tuned model artifacts to the Hugging Face Hub.

Supports:
  * PaddleOCR-VL-1.5 LoRA adapter (``experiments/paddleocr_vl15_lora/adapter``)
  * Surya Foundation fine-tuned checkpoint (``experiments/surya_finetune/``)

Usage:
    python scripts/29_upload_hf_models.py --dry-run
    python scripts/29_upload_hf_models.py --push \\
        --vl-repo-id USER/paddleocr-vl15-yoruba-lora \\
        --surya-repo-id USER/surya-yoruba-finetuned
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_VL_ADAPTER = Path("experiments/paddleocr_vl15_lora/adapter")
DEFAULT_SURYA_DIR = Path("experiments/surya_finetune")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Upload fine-tuned models to HF Hub.")
    parser.add_argument(
        "--vl-adapter-dir",
        type=Path,
        default=DEFAULT_VL_ADAPTER,
        help="PaddleOCR-VL-1.5 LoRA adapter folder.",
    )
    parser.add_argument(
        "--surya-checkpoint-dir",
        type=Path,
        default=DEFAULT_SURYA_DIR,
        help="Surya fine-tune output (checkpoint-* or root).",
    )
    parser.add_argument(
        "--vl-repo-id",
        type=str,
        default=None,
        help="Hub model repo for VL LoRA adapter.",
    )
    parser.add_argument(
        "--surya-repo-id",
        type=str,
        default=None,
        help="Hub model repo for Surya checkpoint.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update repos as private.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload (requires HF_TOKEN).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned uploads without pushing.",
    )
    parser.add_argument(
        "--skip-vl",
        action="store_true",
        help="Skip VL LoRA upload.",
    )
    parser.add_argument(
        "--skip-surya",
        action="store_true",
        help="Skip Surya upload.",
    )
    return parser.parse_args()


def resolve_surya_checkpoint(root: Path) -> Path | None:
    """Return newest checkpoint directory under ``root``."""
    if not root.is_dir():
        return None
    if (root / "adapter_config.json").exists():
        return None
    if (root / "pytorch_model.bin").exists() or any(root.glob("*.safetensors")):
        return root
    checkpoints = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def upload_folder(api, folder: Path, repo_id: str, private: bool, message: str) -> None:
    """Create repo if needed and upload folder."""
    from huggingface_hub import create_repo  # type: ignore

    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message=message,
    )
    log.info("Uploaded %s → %s", folder, repo_id)


def patch_vl_adapter_config(adapter_dir: Path) -> None:
    """Set ``task_type`` on LoRA adapter config for HF PEFT loaders."""
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["task_type"] = "CAUSAL_LM"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def write_manifest(entries: list[dict]) -> Path:
    """Write upload manifest JSON."""
    out = Path("results/tables/hf_models_upload.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uploads": entries,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    """Upload VL LoRA and/or Surya checkpoints."""
    args = parse_args()
    plan: list[dict] = []

    if not args.skip_vl and args.vl_adapter_dir.is_dir():
        if (args.vl_adapter_dir / "adapter_config.json").is_file():
            plan.append(
                {
                    "kind": "paddleocr_vl15_lora",
                    "local_path": str(args.vl_adapter_dir),
                    "repo_id": args.vl_repo_id,
                }
            )

    surya_ckpt = resolve_surya_checkpoint(args.surya_checkpoint_dir)
    if not args.skip_surya and surya_ckpt is not None:
        plan.append(
            {
                "kind": "surya_finetuned",
                "local_path": str(surya_ckpt),
                "repo_id": args.surya_repo_id,
            }
        )

    if not plan:
        log.warning("Nothing to upload — train models first or check paths.")
        return

    for item in plan:
        log.info(
            "Plan: %s  local=%s  repo=%s",
            item["kind"],
            item["local_path"],
            item.get("repo_id") or "(set --*-repo-id)",
        )

    if args.dry_run or not args.push:
        write_manifest([{**p, "status": "dry_run"} for p in plan])
        return

    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as exc:
        raise ImportError("Run: pip install huggingface_hub") from exc

    api = HfApi()
    results: list[dict] = []

    for item in plan:
        repo_id = item.get("repo_id")
        if not repo_id:
            log.error("Missing repo_id for %s", item["kind"])
            continue
        folder = Path(item["local_path"])
        if item["kind"] == "paddleocr_vl15_lora":
            patch_vl_adapter_config(folder)
        upload_folder(
            api,
            folder,
            repo_id,
            args.private,
            f"Yorùbá OCR {item['kind']} from pipeline",
        )
        results.append({**item, "status": "uploaded"})

    manifest = write_manifest(results)
    log.info("Manifest: %s", manifest)


if __name__ == "__main__":
    main()
