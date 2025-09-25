from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from .match import MatchResult


@dataclass
class EnrichOptions:
    prefix: str = "lk_"
    diagnostics: bool = False
    overwrite_base: bool = False
    overwrite_if_empty: bool = False
    # Field mapping and creation behavior
    # mappings: map source lookup column -> target base column name
    mappings: Optional[dict] = None
    # new_field_mode:
    #  - "default": if overwrite_base and target exists, write there; otherwise create prefixed new column
    #  - "always": always create a new prefixed column regardless of conflicts
    #  - "on_conflict": create target column if it doesn't exist; if it does, create prefixed new column
    new_field_mode: str = "default"


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
    # Decide target columns for each source
    col_plan: List[tuple] = []  # (src_lookup_col, target_col_name, write_mode)
    # write_mode: "existing" to write into existing/target; "new" to write into a new column name
    for src in lookup_take_cols:
        target = src
        if opts.mappings and src in opts.mappings:
            target = str(opts.mappings[src])
        mode = "new"
        new_name = f"{opts.prefix}{target}"
        if opts.new_field_mode == "always":
            mode = "new"
            dest = new_name
        elif opts.new_field_mode == "on_conflict":
            if target in out.columns:
                mode = "new"
                dest = new_name
            else:
                mode = "existing"
                dest = target
        else:  # default
            if opts.overwrite_base and target in out.columns:
                mode = "existing"
                dest = target
            else:
                mode = "new"
                dest = new_name
        col_plan.append((src, dest, mode))

    # Initialize any new columns in the plan
    for _, dest, mode in col_plan:
        if mode == "new" and dest not in out.columns:
            out[dest] = pd.NA

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
        # Gather values for each requested column based on plan
        for src, dest, mode in col_plan:
            lk_values = lookup_df.iloc[matched_indices_lookup][src].values
            if mode == "existing":
                if opts.overwrite_if_empty:
                    base_vals = out.loc[matched_indices_base, dest]
                    empty_mask = base_vals.isna() | (base_vals.astype(str).str.strip() == "")
                    if empty_mask.any():
                        idxs = [idx for idx, is_empty in zip(matched_indices_base, empty_mask.tolist()) if is_empty]
                        vals = [val for val, is_empty in zip(lk_values, empty_mask.tolist()) if is_empty]
                        if idxs:
                            out.loc[idxs, dest] = vals
                else:
                    out.loc[matched_indices_base, dest] = lk_values
            else:  # new column
                out.loc[matched_indices_base, dest] = lk_values

    if opts.diagnostics:
        out.loc[:, "match_score"] = scores if scores else pd.NA
        out.loc[:, "matched_lookup_key"] = keys if keys else pd.NA
        out.loc[:, "match_count"] = counts if counts else pd.NA

    return out
