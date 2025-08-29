from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd


def read_table(path: str | Path, sheet: Optional[str] = None) -> pd.DataFrame:
    """Read CSV or XLSX into a DataFrame. Uses first sheet by default.

    Args:
        path: Path to CSV/XLSX file.
        sheet: Sheet name/index for XLSX. Ignored for CSV.

    Raises:
        ValueError: If extension unsupported or file missing.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"File not found: {p}")
    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(p)
    if ext in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        # pandas returns a dict when sheet_name=None; default to first sheet instead
        sheet_name = 0 if sheet is None else sheet
        df_or_dict = pd.read_excel(p, sheet_name=sheet_name)
        if isinstance(df_or_dict, dict):
            # Fallback: pick the first sheet if a dict was returned
            if not df_or_dict:
                raise ValueError(f"Excel file has no sheets: {p}")
            first_key = next(iter(df_or_dict))
            return df_or_dict[first_key]
        return df_or_dict
    raise ValueError(f"Unsupported file extension: {ext}")


def write_table(
    df: pd.DataFrame,
    base_input_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Write DataFrame as CSV or XLSX to output_path or alongside base_input_path.

    Returns the written path.
    """
    in_path = Path(base_input_path)
    out_path = Path(output_path) if output_path else default_output_path(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ext = in_path.suffix.lower()
    if ext == ".csv":
        df.to_csv(out_path, index=False)
    elif ext in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        # Use openpyxl engine implicitly
        df.to_excel(out_path, index=False)
    else:
        raise ValueError(f"Unsupported base file extension: {ext}")
    return out_path


def default_output_path(base_input_path: Path) -> Path:
    """Return default enriched output path next to base input.

    e.g., base.csv -> base.enriched.csv; base.xlsx -> base.enriched.xlsx
    """
    return base_input_path.with_name(
        f"{base_input_path.stem}.enriched{base_input_path.suffix}"
    )


def list_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns]
