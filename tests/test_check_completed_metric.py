from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_completed_metric import is_complete  # noqa: E402


class CompletedMetricTests(unittest.TestCase):
    def test_requires_aligned_failure_free_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tables = root / "results/tables"
            data = root / "data/processed"
            (tables / "meta").mkdir(parents=True)
            (data / "labels").mkdir(parents=True)
            (data / "images/test").mkdir(parents=True)
            (data / "images/test/a.png").touch()
            (data / "images/test/b.png").touch()
            (data / "labels/test.txt").write_text(
                "images/test/a.png\ta\nimages/test/b.png\tb\n", encoding="utf-8"
            )
            fields = ["model", "split", "n"]
            with (tables / "metrics.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"model": "demo", "split": "test", "n": 2})
            (tables / "demo_test.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            meta = tables / "meta/demo_test.json"
            meta.write_text(
                json.dumps({"provenance": {"failure_count": 0}}), encoding="utf-8"
            )
            self.assertTrue(is_complete("demo", "test", tables, data))

            meta.write_text(
                json.dumps({"provenance": {"failure_count": 1}}), encoding="utf-8"
            )
            self.assertFalse(is_complete("demo", "test", tables, data))


if __name__ == "__main__":
    unittest.main()
