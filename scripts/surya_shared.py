"""
Shared Surya v2 inference backend resolution for baseline and Colab runs.
"""

from __future__ import annotations

import logging
import shutil

log = logging.getLogger(__name__)


def docker_available() -> bool:
    """Return True if a ``docker`` CLI is on PATH and responds to ``docker info``."""
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return False
    import subprocess

    try:
        proc = subprocess.run(
            [docker_bin, "info"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def llamacpp_available() -> bool:
    """Return True if llama.cpp server or CLI binaries appear installed."""
    return bool(shutil.which("llama-server") or shutil.which("llama-cli"))


def resolve_surya_inference_backend(explicit: str = "auto") -> str | None:
    """
    Resolve Surya v2 backend from CLI flag, env, and host capabilities.

    Returns ``None`` to let ``SuryaInferenceManager`` pick its default when no
    backend is suitable. On CUDA hosts without Docker, ``vllm`` is **not**
    selected (Colab fails with ``docker binary not found``).
    """
    import os

    env_backend = os.environ.get("SURYA_INFERENCE_BACKEND", "").strip().lower()
    choice = explicit if explicit and explicit != "auto" else env_backend
    if choice and choice not in ("auto", ""):
        if choice == "vllm" and not docker_available():
            log.warning(
                "SURYA_INFERENCE_BACKEND=vllm but Docker is unavailable; "
                "falling back to auto resolution"
            )
        else:
            return choice

    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except ImportError:
        cuda = False

    if cuda and docker_available():
        return "vllm"
    if cuda and not docker_available():
        log.warning(
            "CUDA detected but Docker unavailable — skipping vllm "
            "(install Docker or set SURYA_INFERENCE_BACKEND=llamacpp)"
        )
    if llamacpp_available():
        return "llamacpp"
    return None
