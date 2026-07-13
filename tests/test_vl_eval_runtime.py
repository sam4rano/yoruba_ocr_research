from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from vl_eval_runtime import evaluate_resumable, run_fingerprint  # noqa: E402


class ResumableEvaluationTests(unittest.TestCase):
    def test_completed_samples_are_not_recomputed(self) -> None:
        pairs = [(Path(f"images/test/{name}.png"), name) for name in ("a", "b", "c")]
        fingerprint = run_fingerprint({"run": 1})
        calls: list[list[str]] = []

        def infer(paths: list[Path]) -> list[str]:
            calls.append([path.stem for path in paths])
            return [path.stem.upper() for path in paths]

        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "predictions.jsonl.partial"
            first, rows = evaluate_resumable(
                pairs,
                transcribe_batch=infer,
                batch_size=2,
                partial_path=partial,
                fingerprint=fingerprint,
                resume=True,
                description="test",
            )
            self.assertEqual(first, [("A", "a"), ("B", "b"), ("C", "c")])
            self.assertTrue(all(row["status"] == "ok" for row in rows))
            calls.clear()

            second, _ = evaluate_resumable(
                pairs,
                transcribe_batch=infer,
                batch_size=2,
                partial_path=partial,
                fingerprint=fingerprint,
                resume=True,
                description="test",
            )
            self.assertEqual(second, first)
            self.assertEqual(calls, [])

    def test_failed_samples_are_retried_on_resume(self) -> None:
        pairs = [(Path(f"images/test/{name}.png"), name) for name in ("a", "b")]
        fingerprint = run_fingerprint({"run": 2})
        broken = True

        def infer(paths: list[Path]) -> list[str]:
            if broken and any(path.stem == "b" for path in paths):
                raise RuntimeError("simulated failure")
            return [path.stem.upper() for path in paths]

        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "predictions.jsonl.partial"
            _, rows = evaluate_resumable(
                pairs,
                transcribe_batch=infer,
                batch_size=2,
                partial_path=partial,
                fingerprint=fingerprint,
                resume=True,
                description="test",
            )
            self.assertEqual([row["status"] for row in rows], ["ok", "error"])

            broken = False
            predictions, rows = evaluate_resumable(
                pairs,
                transcribe_batch=infer,
                batch_size=2,
                partial_path=partial,
                fingerprint=fingerprint,
                resume=True,
                description="test",
            )
            self.assertEqual(predictions, [("A", "a"), ("B", "b")])
            self.assertTrue(all(row["status"] == "ok" for row in rows))


if __name__ == "__main__":
    unittest.main()
