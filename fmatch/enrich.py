from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from .match import MatchResult


@dataclass
class EnrichOptions:
    prefix: str = "lk_"
    diagnostics: bool = False


def enrich(
    base_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    match_results: Sequence[MatchResult],
    lookup_take_cols: Sequence[str],
    opts: EnrichOptions,
) -> pd.DataFrame:
    """Append selected lookup columns to base_df where matches are unique.

    New columns are named with prefix + original lookup column name.
    If diagnostics is True, add match_score, matched_lookup_key, match_count.
    """

    if len(match_results) != len(base_df):
        raise ValueError("match_results length must equal base_df row count")

    # Prepare new columns with None/NaN
    out = base_df.copy()
    for c in lookup_take_cols:
        out[f"{opts.prefix}{c}"] = pd.NA

    if opts.diagnostics:
        out["match_score"] = pd.NA
        out["matched_lookup_key"] = pd.NA
        out["match_count"] = pd.NA

    # Efficiently fill via vectorized assignment using index lists
    matched_indices_base: List[int] = []
    matched_indices_lookup: List[int] = []
    scores: List[Optional[float]] = []
    keys: List[Optional[str]] = []
    counts: List[Optional[int]] = []

    for i, res in enumerate(match_results):
        if res.reason == "matched" and res.row_index is not None:
            matched_indices_base.append(i)
            matched_indices_lookup.append(res.row_index)
        if opts.diagnostics:
            scores.append(res.score)
            keys.append(res.matched_key)
            counts.append(res.candidate_count)

    if matched_indices_base:
        # Gather values for each requested column
        for c in lookup_take_cols:
            out.loc[matched_indices_base, f"{opts.prefix}{c}"] = (
                lookup_df.iloc[matched_indices_lookup][c].values
            )

    if opts.diagnostics:
        out.loc[:, "match_score"] = scores if scores else pd.NA
        out.loc[:, "matched_lookup_key"] = keys if keys else pd.NA
        out.loc[:, "match_count"] = counts if counts else pd.NA

    return out

