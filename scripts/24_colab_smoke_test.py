"""
Colab / local smoke test for the Yorùbá OCR pipeline.

Two modes:

  --quick (default for Colab Step 5b)
      Data layout, Python deps, config/VL export sanity — **no eval JSONL required**.
      Use after Step 3 install, before GPU baselines.

  --full
      Also runs analysis 17–19, compile+alignment, checkpoint audit, HF card dry-run.
      Requires prior eval JSONL logs under results/tables/.

Usage:
    python scripts/24_colab_smoke_test.py --quick --skip-config
    python scripts/24_colab_smoke_test.py --full --skip-config
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
PY = os.environ.get("PYTHON", sys.executable)
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Pre-flight only: data, deps, VL export (no eval JSONL needed).",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Full pipeline smoke: includes analysis 17–19 and HF card dry-run.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/tables/colab_smoke_test.json"),
        help="JSON report path.",
    )
    args = parser.parse_args()
    if not args.quick and not args.full:
        args.quick = True

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

    def deps_full_checks() -> None:
        import datasets  # noqa: F401

        deps_checks()

    def refresh_report() -> None:
        run([PY, "scripts/02c_refresh_dataset_report.py"])

    def compile_tables_quick() -> None:
        metrics = ROOT / "results/tables/metrics.csv"
        if not metrics.is_file():
            print("metrics.csv missing — skipping compile (run baselines first)")
            return
        run([
            PY, "scripts/11_compile_results.py",
            "--results-csv", "results/tables/metrics.csv",
            "--output-dir", "results/tables",
        ])
        check_path(ROOT / "results/tables/table1_main_comparison.csv")

    def hf_dataset_card() -> None:
        run([PY, "scripts/25_upload_hf_dataset.py", "--dry-run"])
        check_path(ROOT / "data/hf_export/README.md")
        check_path(ROOT / "data/hf_export/LICENSE")
        if "cc-by-4.0" not in (ROOT / "data/hf_export/README.md").read_text(encoding="utf-8"):
            raise ValueError("HF dataset card missing cc-by-4.0 license")

    step("data/processed layout", data_checks)
    step("python deps (editdistance)", deps_checks)
    step("02c refresh dataset report", refresh_report)
    step("phase_03 config", config_phase)
    step("phase_14 VL export", vl_export)
    step("compile table1 (if metrics.csv exists)", compile_tables_quick)
    step("yor_ocr.ipynb syntax", notebook_syntax)

    if args.full:
        step("02b data quality audit", audit_and_eda)
        step("analysis 17-19", analysis)
        step("phase_09 compile + alignment", compile_tables)
        step("checkpoint audit", checkpoint_audit)
        step("research_approach.md", research_doc)
        step("python deps (datasets)", deps_full_checks)
        step("HF dataset card dry-run", hf_dataset_card)

    results["mode"] = "full" if args.full else "quick"
    results["python"] = PY

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport: {args.report}")
    print("SMOKE TEST:", "PASSED" if results["ok"] else "FAILED")
    if not results["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
