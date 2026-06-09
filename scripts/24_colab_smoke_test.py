"""
Colab / local smoke test for the Yorùbá OCR pipeline.

Validates data layout, config generation, VL export, analysis scripts,
table compilation, checkpoint audit, and research_approach.md — without
full GPU training or multi-hour eval runs.

Usage:
    python scripts/24_colab_smoke_test.py
    python scripts/24_colab_smoke_test.py --skip-config   # if weights already fetched
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SHELL_ENV = {**os.environ, "PYTHON": PY, "PROJECT_ROOT": str(ROOT)}


def run(cmd: list[str], *, cwd: Path = ROOT, env: dict | None = None) -> None:
    """Run subprocess and raise on failure."""
    print("$", " ".join(cmd))
    merged = {**SHELL_ENV, **(env or {})}
    subprocess.check_call(cmd, cwd=cwd, env=merged)


def check_path(path: Path, kind: str = "file") -> None:
    """Assert a required path exists."""
    ok = path.is_file() if kind == "file" else path.is_dir()
    if not ok:
        raise FileNotFoundError(f"Missing {kind}: {path}")


def count_lines(path: Path) -> int:
    """Return line count for a text file."""
    return sum(1 for _ in path.open(encoding="utf-8"))


def main() -> None:
    """Run smoke checks."""
    parser = argparse.ArgumentParser(description="Smoke-test Colab pipeline components.")
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Skip phase_03 (Paddle weight download).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/tables/colab_smoke_test.json"),
        help="JSON report path.",
    )
    args = parser.parse_args()

    results: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "steps": [],
        "ok": True,
    }

    def step(name: str, fn) -> None:
        """Run one check and record outcome."""
        try:
            fn()
            results["steps"].append({"name": name, "status": "ok"})
            print(f"OK  {name}")
        except Exception as exc:  # noqa: BLE001
            results["steps"].append({"name": name, "status": "fail", "error": str(exc)})
            results["ok"] = False
            print(f"FAIL {name}: {exc}")

    def data_checks() -> None:
        for rel in (
            "data/processed/labels/train.txt",
            "data/processed/labels/val.txt",
            "data/processed/labels/test.txt",
            "data/processed/dictionary/yoruba_char_dict.txt",
            "data/processed/images/test",
        ):
            p = ROOT / rel
            check_path(p, "file" if rel.endswith(".txt") else "dir")
        n_test = count_lines(ROOT / "data/processed/labels/test.txt")
        if n_test < 10:
            raise ValueError(f"test split too small: {n_test} lines")

    def audit_and_eda() -> None:
        run([PY, "scripts/02b_data_quality_audit.py",
             "--data-dir", "data/processed",
             "--out-json", "results/tables/data_quality.json"])
        check_path(ROOT / "results/tables/data_quality.json")

    def config_phase() -> None:
        if args.skip_config:
            check_path(ROOT / "configs/paddleocr_yoruba_rec.yml")
            return
        env = {**SHELL_ENV, "CONFIG_FORCE_GPU": "0"}
        run(["bash", "scripts/shell/phase_03_config.sh"], cwd=ROOT, env=env)

    def vl_export() -> None:
        run(["bash", "scripts/shell/phase_14_export_vl15.sh"])
        for split in ("train", "val", "test"):
            f = ROOT / "data/paddleocr_vl15_sft" / f"{split}.jsonl"
            check_path(f)
            if count_lines(f) < 1:
                raise ValueError(f"empty export: {f}")

    def analysis() -> None:
        for script in (
            "17_stratified_error_analysis.py",
            "18_der_universe_ablation.py",
            "19_bootstrap_metric_cis.py",
        ):
            run([PY, f"scripts/{script}"])
        for name in (
            "bootstrap_metric_cis.csv",
            "bootstrap_pairwise_comparison.csv",
            "stratified_der_by_density.csv",
            "der_universe_ablation.csv",
        ):
            check_path(ROOT / "results/tables" / name)

    def compile_tables() -> None:
        check_path(ROOT / "results/tables/metrics.csv")
        run(["bash", "scripts/shell/phase_09_compile.sh"])
        check_path(ROOT / "results/tables/table1_main_comparison.csv")
        check_path(ROOT / "results/tables/metrics_summary.csv")

    def checkpoint_audit() -> None:
        run([
            PY, "scripts/12_diagnose_hypotheses.py", "checkpoints",
            "--csv", "results/tables/metrics.csv",
            "--report", "results/tables/checkpoint_audit.json",
        ])
        check_path(ROOT / "results/tables/checkpoint_audit.json")

    def research_doc() -> None:
        run([PY, "scripts/23_write_research_approach.py", "--output", "research_approach.md"])
        check_path(ROOT / "research_approach.md")

    def notebook_syntax() -> None:
        import ast

        nb = json.loads((ROOT / "yor_ocr.ipynb").read_text(encoding="utf-8"))
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            ast.parse("".join(cell["source"]))

    def deps_checks() -> None:
        import editdistance  # noqa: F401

    step("data/processed layout", data_checks)
    step("python deps (editdistance)", deps_checks)
    step("02b data quality audit", audit_and_eda)
    step("phase_03 config", config_phase)
    step("phase_14 VL export", vl_export)
    step("analysis 17-19", analysis)
    step("phase_09 compile", compile_tables)
    step("checkpoint audit", checkpoint_audit)
    step("research_approach.md", research_doc)
    step("yor_ocr.ipynb syntax", notebook_syntax)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport: {args.report}")
    print("SMOKE TEST:", "PASSED" if results["ok"] else "FAILED")
    if not results["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
