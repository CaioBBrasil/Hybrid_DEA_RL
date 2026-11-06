from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rl.common.config import load_config


class ConfigTests(unittest.TestCase):
    def test_extends_deep_merges_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.yaml").write_text("a:\n  b: 1\n  c: 2\nx: 3\n", encoding="utf-8")
            (root / "child.yaml").write_text("extends: base.yaml\na:\n  b: 9\n", encoding="utf-8")
            config = load_config(root / "child.yaml")
            self.assertEqual(config["a"], {"b": 9, "c": 2})
            self.assertEqual(config["x"], 3)


if __name__ == "__main__":
    unittest.main()
