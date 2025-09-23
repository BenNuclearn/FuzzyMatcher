from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd

from . import io as io_mod
from .enrich import EnrichOptions, enrich
from .match import (
    MatchPolicy,
    NormalizeOptions,
    normalize_series,
    build_lookup_index,
    match_one,
    top_candidates,
    MatchResult,
)


def _prompt_select(prompt: str, options: List[str], multi: bool = False) -> List[str]:
    print(prompt)
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    while True:
        raw = input("Enter index" + ("es (comma)" if multi else "") + ": ").strip()
        try:
            if multi:
                idxs = [int(x) for x in raw.split(",") if x != ""]
                return [options[i] for i in idxs]
            idx = int(raw)
            return [options[idx]]
        except (ValueError, IndexError):
            print("Invalid selection. Try again.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fmatch",
        description="Fuzzy match values from a lookup file into a base file",
    )
    p.add_argument("base_path", help="Base CSV/XLSX file path")
    p.add_argument("lookup_path", help="Lookup CSV/XLSX file path")
    p.add_argument(
        "--base-key",
        dest="base_key",
        help="Key column(s) in base (comma-separated for multiple)",
    )
    p.add_argument(
        "--lookup-key",
        dest="lookup_key",
        help="Key column(s) in lookup (comma-separated for multiple)",
    )
    p.add_argument(
        "--take",
        dest="take_cols",
        help="Comma-separated lookup columns to copy into base",
    )
    p.add_argument("--output", help="Output file path override")
    p.add_argument("--sheet", help="XLSX sheet name (for both files if applicable)")
    p.add_argument("--threshold", type=int, default=85, help="Match threshold (0-100)")
    p.add_argument(
        "--scorer",
        default="smart",
        choices=[
            "smart",
            "WRatio",
            "ratio",
            "token_set_ratio",
            "token_sort_ratio",
            "partial_ratio",
            "partial_token_set_ratio",
        ],
        help="Scorer (smart=acronym/segment aware)",
    )
    p.add_argument("--top-n", type=int, default=3, help="Top-N candidates to consider")
    p.add_argument("--margin", type=int, default=3, help="Tie-break margin over second best")
    p.add_argument("--prefix", default="lk_", help="Prefix for new columns")
    p.add_argument("--diagnostics", action="store_true", help="Add diagnostics columns")
    p.add_argument("--no-casefold", action="store_true", help="Disable case folding")
    p.add_argument("--no-trim", action="store_true", help="Disable whitespace trim/collapse")
    p.add_argument("--alnum-only", action="store_true", help="Keep only alphanumeric during normalization")
    # Ambiguity resolution on by default; allow explicit opt-out
    p.add_argument(
        "--resolve-ambiguous",
        dest="resolve_ambiguous",
        action="store_true",
        default=True,
        help="Interactively resolve ambiguous matches by choosing a candidate (default)",
    )
    p.add_argument(
        "--no-resolve-ambiguous",
        dest="resolve_ambiguous",
        action="store_false",
        help="Disable interactive resolution and leave ambiguous rows unresolved",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    base_df = io_mod.read_table(args.base_path, sheet=args.sheet)
    lookup_df = io_mod.read_table(args.lookup_path, sheet=args.sheet)

    base_cols = io_mod.list_columns(base_df)
    lookup_cols = io_mod.list_columns(lookup_df)

    def _parse_keys(arg_val: Optional[str]) -> Optional[List[str]]:
        if not arg_val:
            return None
        parts = [c.strip() for c in arg_val.split(",") if c.strip()]
        return parts or None

    base_keys = _parse_keys(args.base_key)
    if not base_keys or any(k not in base_cols for k in base_keys):
        base_keys = _prompt_select(
            "Select base key column(s):", base_cols, multi=True
        )

    lookup_keys = _parse_keys(args.lookup_key)
    if not lookup_keys or any(k not in lookup_cols for k in lookup_keys):
        lookup_keys = _prompt_select(
            "Select lookup key column(s):", lookup_cols, multi=True
        )

    take_cols: List[str]
    if args.take_cols:
        take_cols = [c.strip() for c in args.take_cols.split(",") if c.strip()]
    else:
        take_cols = _prompt_select("Select lookup columns to bring:", lookup_cols, multi=True)

    # Validate columns
    for c in take_cols:
        if c not in lookup_cols:
            parser.error(f"Lookup column not found: {c}")

    # Normalization and match policy
    norm = NormalizeOptions(
        casefold=not args.no_casefold,
        trim_space=not args.no_trim,
        collapse_space=not args.no_trim,
        alnum_only=args.alnum_only,
    )
    policy = MatchPolicy(
        threshold=args.threshold,
        top_n=args.top_n,
        tie_margin=args.margin,
        scorer=args.scorer,
    )

    # Build composite key series then normalize
    base_raw = _composite_key_series(base_df, base_keys)
    lookup_raw = _composite_key_series(lookup_df, lookup_keys)
    base_norm = normalize_series(base_raw, norm)
    lookup_norm = normalize_series(lookup_raw, norm)
    index = build_lookup_index(list(lookup_norm))
    results: List[MatchResult] = []
    for _, b in base_norm.items():
        res = match_one(b, index, policy)
        results.append(res)

    if args.resolve_ambiguous:
        _resolve_ambiguous_interactive(
            results,
            base_values=list(base_raw.astype(str).fillna("")),
            base_norm=list(base_norm),
            lookup_df=lookup_df,
            lookup_key_cols=lookup_keys,
            take_cols=take_cols,
            lookup_index=index,
            policy=policy,
        )

    # Enrich
    enriched = enrich(
        base_df=base_df,
        lookup_df=lookup_df,
        match_results=results,
        lookup_take_cols=take_cols,
        opts=EnrichOptions(prefix=args.prefix, diagnostics=args.diagnostics),
    )

    # Summary
    matched = sum(1 for r in results if r.reason == "matched")
    ambiguous = sum(1 for r in results if r.reason == "ambiguous")
    unmatched = sum(1 for r in results if r.reason == "unmatched")
    total = len(results)
    print(
        f"Processed {total} rows — matched: {matched}, ambiguous: {ambiguous}, unmatched: {unmatched}"
    )

    # Write
    out_path = io_mod.write_table(enriched, base_input_path=args.base_path, output_path=args.output)
    print(f"Wrote: {out_path}")
    return 0


def _resolve_ambiguous_interactive(
    results: List[MatchResult],
    base_values: Sequence[str],
    base_norm: Sequence[str],
    lookup_df: pd.DataFrame,
    lookup_key_cols: Sequence[str],
    take_cols: Sequence[str],
    lookup_index,
    policy: MatchPolicy,
) -> None:
    """Prompt user to resolve ambiguous rows by selecting a lookup row.

    Modifies `results` in place when user selects a candidate.
    """
    # Cache of user decisions: normalized base value -> chosen lookup row index
    decision_cache: dict[str, int] = {}

    for i, res in enumerate(results):
        if res.reason != "ambiguous":
            continue
        # Auto-apply prior decision for identical base values
        if base_norm[i] in decision_cache:
            row_idx = decision_cache[base_norm[i]]
            # Use the exact normalized key from lookup for diagnostics if available
            norm_key = None
            try:
                # Find the normalized key corresponding to this row index
                for k, rows in lookup_index.items():
                    if row_idx in rows:
                        norm_key = k
                        break
            except Exception:
                norm_key = None
            results[i] = MatchResult(
                row_index=row_idx,
                score=None,
                matched_key=norm_key,
                candidate_count=res.candidate_count,
                reason="matched",
            )
            print(f"Auto-resolved row {i} using prior decision for identical base value.")
            continue

        print("\nAmbiguous match for base row", i)
        print("  Base value:", base_values[i])
        print("  (normalized):", base_norm[i])

        cands = top_candidates(base_norm[i], lookup_index, policy)
        if not cands:
            print("  No candidates above threshold. Skipping.")
            continue

        def _print_candidates(tag: str, cand_list):
            print(tag)
            for j, (row_idx, norm_key, score) in enumerate(cand_list):
                preview = []
                try:
                    orig_key = " | ".join(
                        [str(lookup_df.iloc[row_idx][k]) for k in lookup_key_cols]
                    )
                except Exception:
                    orig_key = "<err>"
                for c in take_cols[:3]:  # limit preview to first 3 columns
                    try:
                        preview.append(f"{c}={lookup_df.iloc[row_idx][c]}")
                    except Exception:
                        preview.append(f"{c}=<na>")
                preview_str = ", ".join(preview)
                print(
                    f"  [{j}] score={score:.1f} key='{orig_key}' {('('+preview_str+')') if preview else ''}"
                )

        display_cands = list(cands)
        _print_candidates("  Candidates:", display_cands)
        print("  Commands: enter a number; 's <query>' to search; '/<query>' also searches; blank to skip")

        while True:
            raw = input("Select candidate index to accept (blank=skip): ").strip()
            if raw == "":
                print("  Skipped; leaving as ambiguous.")
                break
            # Search command
            if raw.startswith("s ") or raw.startswith("/"):
                query = raw[2:] if raw.startswith("s ") else raw[1:]
                if not query.strip():
                    print("  Provide a search query after 's '.")
                    continue
                try:
                    from rapidfuzz import process as _rf_process  # type: ignore
                    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
                except Exception:
                    print("  Search requires rapidfuzz installed.")
                    continue
                # Build list of all normalized keys
                all_keys = list(lookup_index.keys())
                # Normalize query similarly to base normalization assumptions: lower/trim
                q = query.strip().casefold()
                scorer = _rf_fuzz.token_set_ratio if policy.scorer != "smart" else _rf_fuzz.token_set_ratio
                found = _rf_process.extract(q, all_keys, scorer=scorer, limit=10)
                # Append unique row candidates from search to display list
                existing_rows = {r for r, _, _ in display_cands}
                added = 0
                for k, s, _ in found:
                    for row_idx in lookup_index.get(k, []):
                        if row_idx in existing_rows:
                            continue
                        display_cands.append((row_idx, k, float(s)))
                        existing_rows.add(row_idx)
                        added += 1
                if added == 0:
                    print("  No new results for your search.")
                else:
                    _print_candidates("  Updated candidates (includes search results):", display_cands)
                continue
            try:
                sel = int(raw)
                if 0 <= sel < len(display_cands):
                    row_idx, norm_key, score = display_cands[sel]
                    # Update result in place to a resolved match
                    results[i] = MatchResult(
                        row_index=row_idx,
                        score=score,
                        matched_key=norm_key,
                        candidate_count=res.candidate_count,
                        reason="matched",
                    )
                    # Cache for identical base values later in the run
                    decision_cache[base_norm[i]] = row_idx
                    print(f"  Accepted candidate at lookup row {row_idx}.")
                    break
            except ValueError:
                pass
            print("  Invalid selection. Enter a number or blank to skip.")


def _composite_key_series(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """Build a composite string key by joining multiple columns.

    Joins with " | " for readability; downstream normalization handles spacing/case.
    """
    if not cols:
        raise ValueError("At least one key column is required")
    s = df[cols[0]].astype(str).fillna("")
    for c in cols[1:]:
        s = s + " | " + df[c].astype(str).fillna("")
    return s
