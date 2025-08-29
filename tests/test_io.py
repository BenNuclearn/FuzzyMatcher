from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fmatch.io import default_output_path, list_columns, read_table, write_table


class TestIO(unittest.TestCase):
    def test_default_output_path(self):
        p = Path("/tmp/base.csv")
        self.assertEqual(default_output_path(p).name, "base.enriched.csv")
        p2 = Path("/tmp/book.xlsx")
        self.assertEqual(default_output_path(p2).name, "book.enriched.xlsx")

    def test_read_write_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            base_path = Path(td) / "base.csv"
            # write a simple CSV
            df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
            df.to_csv(base_path, index=False)

            read_df = read_table(str(base_path))
            self.assertEqual(read_df.shape, (2, 2))

            out_path = write_table(read_df, base_input_path=str(base_path))
            self.assertTrue(out_path.exists())

    def test_list_columns(self):
        df = pd.DataFrame({"c1": [], "c2": []})
        self.assertEqual(list_columns(df), ["c1", "c2"])

    def test_read_xlsx_defaults_first_sheet(self):
        try:
            import openpyxl  # noqa: F401
        except Exception:
            self.skipTest("openpyxl not installed")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "book.xlsx"
            # Write two sheets and ensure default read pulls the first
            with pd.ExcelWriter(p) as xw:
                pd.DataFrame({"a": [1]}).to_excel(xw, index=False, sheet_name="First")
                pd.DataFrame({"b": [2]}).to_excel(xw, index=False, sheet_name="Second")
            df = read_table(str(p))
            self.assertEqual(list(df.columns), ["a"])  # first sheet


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
