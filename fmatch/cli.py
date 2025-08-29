from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from . import io as io_mod
from .enrich import EnrichOptions, enrich
from .match import MatchPolicy, NormalizeOptions, match_all, normalize_series, build_lookup_index, match_one, top_candidates, MatchResult


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
    p.add_argument("--base-key", dest="base_key", help="Key column in base file")
    p.add_argument("--lookup-key", dest="lookup_key", help="Key column in lookup file")
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
        default="WRatio",
        choices=["WRatio", "ratio", "token_set_ratio", "token_sort_ratio", "partial_ratio"],
        help="RapidFuzz scorer",
    )
    p.add_argument("--top-n", type=int, default=3, help="Top-N candidates to consider")
    p.add_argument("--margin", type=int, default=3, help="Tie-break margin over second best")
    p.add_argument("--prefix", default="lk_", help="Prefix for new columns")
    p.add_argument("--diagnostics", action="store_true", help="Add diagnostics columns")
    p.add_argument("--no-casefold", action="store_true", help="Disable case folding")
    p.add_argument("--no-trim", action="store_true", help="Disable whitespace trim/collapse")
    p.add_argument("--alnum-only", action="store_true", help="Keep only alphanumeric during normalization")
    p.add_argument(
        "--resolve-ambiguous",
        action="store_true",
        help="Interactively resolve ambiguous matches by choosing a candidate",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    base_df = io_mod.read_table(args.base_path, sheet=args.sheet)
    lookup_df = io_mod.read_table(args.lookup_path, sheet=args.sheet)

    base_cols = io_mod.list_columns(base_df)
    lookup_cols = io_mod.list_columns(lookup_df)

    base_key = args.base_key
    if not base_key or base_key not in base_cols:
        [base_key] = _prompt_select("Select base key column:", base_cols)

    lookup_key = args.lookup_key
    if not lookup_key or lookup_key not in lookup_cols:
        [lookup_key] = _prompt_select("Select lookup key column:", lookup_cols)

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

    # Run matching (keep normalized/index to optionally resolve ambiguous)
    base_norm = normalize_series(base_df[base_key], norm)
    lookup_norm = normalize_series(lookup_df[lookup_key], norm)
    index = build_lookup_index(list(lookup_norm))
    results: List[MatchResult] = []
    for _, b in base_norm.items():
        res = match_one(b, index, policy)
        results.append(res)

    if args.resolve_ambiguous:
        _resolve_ambiguous_interactive(
            results,
            base_values=list(base_df[base_key].astype(str).fillna("")),
            base_norm=list(base_norm),
            lookup_df=lookup_df,
            lookup_key_col=lookup_key,
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
    lookup_key_col: str,
    take_cols: Sequence[str],
    lookup_index,
    policy: MatchPolicy,
) -> None:
    """Prompt user to resolve ambiguous rows by selecting a lookup row.

    Modifies `results` in place when user selects a candidate.
    """
    for i, res in enumerate(results):
        if res.reason != "ambiguous":
            continue
        print("\nAmbiguous match for base row", i)
        print("  Base value:", base_values[i])
        print("  (normalized):", base_norm[i])

        cands = top_candidates(base_norm[i], lookup_index, policy)
        if not cands:
            print("  No candidates above threshold. Skipping.")
            continue

        # Display candidates with a small preview
        for j, (row_idx, norm_key, score) in enumerate(cands):
            preview = []
            try:
                orig_key = str(lookup_df.iloc[row_idx][lookup_key_col])
            except Exception:
                orig_key = "<err>"
            for c in take_cols[:3]:  # limit preview to first 3 columns
                try:
                    preview.append(f"{c}={lookup_df.iloc[row_idx][c]}")
                except Exception:
                    preview.append(f"{c}=<na>")
            preview_str = ", ".join(preview)
            print(f"  [{j}] score={score:.1f} key='{orig_key}' {('('+preview_str+')') if preview else ''}")

        while True:
            raw = input("Select candidate index to accept (blank=skip): ").strip()
            if raw == "":
                print("  Skipped; leaving as ambiguous.")
                break
            try:
                sel = int(raw)
                if 0 <= sel < len(cands):
                    row_idx, norm_key, score = cands[sel]
                    # Update result in place to a resolved match
                    results[i] = MatchResult(
                        row_index=row_idx,
                        score=score,
                        matched_key=norm_key,
                        candidate_count=res.candidate_count,
                        reason="matched",
                    )
                    print(f"  Accepted candidate at lookup row {row_idx}.")
                    break
            except ValueError:
                pass
            print("  Invalid selection. Enter a number or blank to skip.")
