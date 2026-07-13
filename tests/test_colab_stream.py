import tempfile
import unittest
from pathlib import Path

from scripts.colab_stream import _validate_local_command


class ValidateLocalCommandTests(unittest.TestCase):
    def test_existing_script_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "scripts" / "smoke_test_runtime.py"
            script.parent.mkdir()
            script.touch()

            _validate_local_command(["python", "scripts/smoke_test_runtime.py"], root)

    def test_legacy_layout_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "scripts" / "24_colab_smoke_test.py"
            legacy.parent.mkdir()
            legacy.touch()

            with self.assertRaisesRegex(FileNotFoundError, "notebook is newer"):
                _validate_local_command(
                    ["python", "scripts/smoke_test_runtime.py"], root
                )


if __name__ == "__main__":
    unittest.main()
