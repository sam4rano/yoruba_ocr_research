from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from initialize_notebook_run import initialize_run  # noqa: E402


class NotebookRunInitializationTests(unittest.TestCase):
    def test_same_run_id_does_not_reset_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tables = Path(tmp)
            self.assertTrue(initialize_run("ocr-run-001", tables, reset=True))
            evidence = tables / "keep.jsonl.partial"
            evidence.write_text("progress\n", encoding="utf-8")
            self.assertFalse(initialize_run("ocr-run-001", tables, reset=True))
            self.assertTrue(evidence.is_file())

    def test_new_run_id_requires_explicit_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tables = Path(tmp)
            initialize_run("ocr-run-001", tables, reset=True)
            with self.assertRaises(RuntimeError):
                initialize_run("ocr-run-002", tables, reset=False)


if __name__ == "__main__":
    unittest.main()
