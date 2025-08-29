from __future__ import annotations

import sys
from pathlib import Path
import sys

import pandas as pd

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fmatch.cli import main


def run() -> int:
    here = Path(__file__).resolve().parent.parent
    base = here / "samples" / "base.csv"
    lookup = here / "samples" / "lookup.csv"
    if not base.exists() or not lookup.exists():
        print("Sample files not found. Expected in samples/ directory.")
        return 1
    code = main(
        [
            str(base),
            str(lookup),
            "--base-key",
            "Name",
            "--lookup-key",
            "FullName",
            "--take",
            "email,company",
        ]
    )
    out_path = base.with_name("base.enriched.csv")
    assert out_path.exists(), "Output file not written"
    df = pd.read_csv(out_path)
    assert "lk_email" in df.columns and "lk_company" in df.columns, "Missing enriched columns"
    print("Smoke test passed. Wrote:", out_path)
    return code


if __name__ == "__main__":
    raise SystemExit(run())
