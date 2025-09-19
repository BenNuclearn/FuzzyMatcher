from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from fmatch.cli import main

try:
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
    HAS_RAPIDFUZZ = True
except Exception:  # pragma: no cover
    HAS_RAPIDFUZZ = False


@unittest.skipIf(not HAS_RAPIDFUZZ, "rapidfuzz not installed")
class TestCLI(unittest.TestCase):
    def test_cli_non_interactive_enriches_and_writes(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            base = pd.DataFrame({"Name": ["Alice", "Bob", "Carol"]})
            lookup = pd.DataFrame(
                {
                    "FullName": ["alice", "bob", "carol"],
                    "email": ["a@example.com", "b@example.com", "c@example.com"],
                    "company": ["ACo", "BCo", "CCo"],
                }
            )
            base_path = td / "base.csv"
            lookup_path = td / "lookup.csv"
            base.to_csv(base_path, index=False)
            lookup.to_csv(lookup_path, index=False)

            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(
                    [
                        str(base_path),
                        str(lookup_path),
                        "--base-key",
                        "Name",
                        "--lookup-key",
                        "FullName",
                        "--take",
                        "email,company",
                        "--threshold",
                        "85",
                    ]
                )
            self.assertEqual(code, 0)

            out_path = base_path.with_name("base.enriched.csv")
            self.assertTrue(out_path.exists())
            # Check that new columns were added
            df = pd.read_csv(out_path)
            self.assertIn("lk_email", df.columns)
            self.assertIn("lk_company", df.columns)

    def test_cli_resolve_ambiguous(self):
        # Duplicate lookup keys cause ambiguity; resolve by choosing first
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            base = pd.DataFrame({"Name": ["Alice"]})
            lookup = pd.DataFrame(
                {
                    "FullName": ["alice", "alice"],
                    "email": ["a1@example.com", "a2@example.com"],
                }
            )
            base_path = td / "base.csv"
            lookup_path = td / "lookup.csv"
            base.to_csv(base_path, index=False)
            lookup.to_csv(lookup_path, index=False)

            import builtins
            from unittest.mock import patch

            # First prompt is for ambiguous resolution -> choose index 0
            with patch.object(builtins, "input", side_effect=["0"]):
                code = main(
                    [
                        str(base_path),
                        str(lookup_path),
                        "--base-key",
                        "Name",
                        "--lookup-key",
                        "FullName",
                        "--take",
                        "email",
                        "--threshold",
                        "85",
                        "--resolve-ambiguous",
                    ]
                )
            self.assertEqual(code, 0)
            out_path = base_path.with_name("base.enriched.csv")
            df = pd.read_csv(out_path)
            self.assertEqual(df.loc[0, "lk_email"], "a1@example.com")

    def test_cli_multi_key_composite(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            base = pd.DataFrame({"Name": ["Alice"], "ID": ["123"]})
            lookup = pd.DataFrame(
                {
                    "FullName": ["Alice", "Alice"],
                    "ID": ["123", "999"],
                    "email": ["a123@example.com", "a999@example.com"],
                }
            )
            base_path = td / "base.csv"
            lookup_path = td / "lookup.csv"
            base.to_csv(base_path, index=False)
            lookup.to_csv(lookup_path, index=False)

            code = main(
                [
                    str(base_path),
                    str(lookup_path),
                    "--base-key",
                    "Name,ID",
                    "--lookup-key",
                    "FullName,ID",
                    "--take",
                    "email",
                ]
            )
            self.assertEqual(code, 0)
            out_path = base_path.with_name("base.enriched.csv")
            df = pd.read_csv(out_path)
            self.assertEqual(df.loc[0, "lk_email"], "a123@example.com")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
