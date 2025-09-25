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
    match_one_with_secondary,
    top_candidates,
    MatchResult,
)


def _prompt_select(prompt: str, options: List[str], multi: bool = False) -> List[str]:
    """Interactive picker supporting indices, ranges, and names.

    - Displays zero-based indices next to options.
    - Accepts comma-separated indices, ranges like "3-10" (inclusive), and/or names.
    """
    import re as _re

    print(prompt)
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    while True:
        suffix = "es, ranges (e.g., 3-10), or names" if multi else " or name"
        raw = input("Enter index" + suffix + ": ").strip()
        try:
            if multi:
                tokens = [t.strip() for t in raw.split(",") if t.strip()]
                idxs: List[int] = []
                for t in tokens:
                    if _re.fullmatch(r"\d+-\d+", t):
                        a_s, b_s = t.split("-", 1)
                        a, b = int(a_s), int(b_s)
                        if a > b or a < 0 or b >= len(options):
                            raise ValueError
                        idxs.extend(range(a, b + 1))
                    elif t.isdigit():
                        i = int(t)
                        if i < 0 or i >= len(options):
                            raise ValueError
                        idxs.append(i)
                    else:
                        i = options.index(t)
                        idxs.append(i)
                # Deduplicate while preserving order
                seen = set()
                ordered_idxs = []
                for i in idxs:
                    if i not in seen:
                        ordered_idxs.append(i)
                        seen.add(i)
                return [options[i] for i in ordered_idxs]
            # single selection
            if raw.isdigit():
                idx = int(raw)
            else:
                idx = options.index(raw)
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
        help=(
            "Lookup columns to copy into base. Accepts names or 1-based indexes "
            "and ranges, e.g. 'email,company' or '3-34' or '2,5,8-10'."
        ),
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
    p.add_argument("--overwrite-base", action="store_true", help="Write matched values into identically named base columns instead of adding prefixed columns")
    p.add_argument("--overwrite-if-empty", action="store_true", help="Only overwrite base columns where values are empty or NA")
    # Directional copy controls
    p.add_argument(
        "--copy-direction",
        choices=["to-base", "to-lookup", "both"],
        default="to-base",
        help="Direction to copy fields after matching: to-base (default), to-lookup, or both",
    )
    p.add_argument(
        "--overwrite-lookup",
        action="store_true",
        help="When copying to lookup, overwrite existing values in lookup columns",
    )
    p.add_argument(
        "--overwrite-lookup-if-empty",
        action="store_true",
        help="When copying to lookup, only write when lookup cells are empty/NA",
    )
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
    # Optional: update lookup with unmatched rows
    p.add_argument(
        "--update-lookup",
        action="store_true",
        help="Append unmatched base rows as new entries to the lookup file",
    )
    p.add_argument(
        "--lookup-output",
        help="Path to write the updated lookup file (defaults to <lookup>.updated.<ext>)",
    )
    p.add_argument(
        "--fill-take-from-base",
        action="store_true",
        help="When creating new lookup rows, copy values for --take columns from base if columns exist",
    )
    p.add_argument(
        "--append-unmatched-from-lookup",
        action="store_true",
        help="Append lookup rows that were not matched to any base row as new rows in the output",
    )
    # Mapping and new field creation behavior
    p.add_argument(
        "--map",
        dest="mappings",
        help="Column mappings in 'src:dst,src2:dst2' format. For to-base, src=lookup col, dst=base col. For to-lookup, src=base col, dst=lookup col.",
    )
    p.add_argument(
        "--new-field",
        choices=["default", "always", "on_conflict"],
        default="default",
        help="When creating destinations: default=use existing/overwrite rules; always=create new prefixed column; on_conflict=create target if missing else prefixed",
    )
    p.add_argument(
        "--dedupe-by",
        help="Comma-separated column names to drop duplicates by after enrichment/append",
    )
    p.add_argument(
        "--dedupe-keep",
        choices=["first", "last"],
        default="first",
        help="Which duplicate to keep when using --dedupe-by",
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

    def _parse_take(arg_val: Optional[str], cols: List[str]) -> List[str]:
        """Parse --take allowing names, 1-based indices and ranges like '3-34'."""
        if not arg_val:
            return _prompt_select("Select lookup columns to bring:", cols, multi=True)
        tokens = [t.strip() for t in arg_val.split(",") if t.strip()]
        out: List[str] = []
        seen = set()
        def add(col_name: str):
            if col_name not in seen:
                out.append(col_name)
                seen.add(col_name)
        for t in tokens:
            if "-" in t and not t.startswith("-"):
                a, b = t.split("-", 1)
                if a.isdigit() and b.isdigit():
                    start = int(a)
                    end = int(b)
                    if start < 1 or end < 1 or start > end or end > len(cols):
                        raise SystemExit(
                            f"Invalid --take range '{t}'. Use 1-based inclusive indices within 1..{len(cols)}."
                        )
                    for i in range(start - 1, end):
                        add(cols[i])
                    continue
            if t.isdigit():
                idx = int(t)
                if idx < 1 or idx > len(cols):
                    raise SystemExit(
                        f"Invalid --take index '{t}'. Use 1-based indices within 1..{len(cols)}."
                    )
                add(cols[idx - 1])
            else:
                if t not in cols:
                    raise SystemExit(f"Lookup column not found: {t}")
                add(t)
        return out

    take_cols: List[str] = _parse_take(args.take_cols, lookup_cols)

    # Validate columns (already validated in _parse_take for names and indices)

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

    # If multiple keys provided, treat the first as primary and the rest as secondary boosts
    use_secondary_boost = len(base_keys) >= 2 and len(lookup_keys) >= 2

    if use_secondary_boost:
        # Primary only for matching
        base_primary_raw = base_df[base_keys[0]].astype(str).fillna("")
        lookup_primary_raw = lookup_df[lookup_keys[0]].astype(str).fillna("")
        base_primary_norm = normalize_series(base_primary_raw, norm)
        lookup_primary_norm = normalize_series(lookup_primary_raw, norm)
        index = build_lookup_index(list(lookup_primary_norm))

        # Secondary tuples (normalized)
        sec_base_cols = base_keys[1:]
        sec_lookup_cols = lookup_keys[1:]
        def _norm_tuple_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[Tuple[str, ...]]:
            if not cols:
                return [("") for _ in range(len(df))]  # type: ignore
            normed = [normalize_series(df[c].astype(str).fillna(""), norm) for c in cols]
            out: List[Tuple[str, ...]] = []
            for i in range(len(df)):
                out.append(tuple(str(n.iloc[i]) for n in normed))
            return out
        base_secondary_norm_list = _norm_tuple_cols(base_df, sec_base_cols)
        lookup_secondary_norm_list = _norm_tuple_cols(lookup_df, sec_lookup_cols)

        # For display and decision cache, still build composite strings
        base_raw = _composite_key_series(base_df, base_keys)
        base_norm_display = normalize_series(base_raw, norm)

        results: List[MatchResult] = []
        for i, b in enumerate(base_primary_norm):
            res = match_one_with_secondary(
                base_primary_norm=str(b),
                base_secondary_norm=tuple(base_secondary_norm_list[i]) if base_secondary_norm_list else None,
                lookup_key_to_rows=index,
                lookup_secondary_norm_rows=lookup_secondary_norm_list,
                policy=policy,
                secondary_boost=10,
            )
            results.append(res)
    else:
        # Build composite key series then normalize
        base_raw = _composite_key_series(base_df, base_keys)
        lookup_raw = _composite_key_series(lookup_df, lookup_keys)
        base_norm_display = normalize_series(base_raw, norm)
        lookup_norm = normalize_series(lookup_raw, norm)
        index = build_lookup_index(list(lookup_norm))
        results: List[MatchResult] = []
        for _, b in base_norm_display.items():
            res = match_one(b, index, policy)
            results.append(res)

    if args.resolve_ambiguous:
        # Use primary normalization for candidate search when boosting is enabled
        base_norm_for_matching: Sequence[str]
        if use_secondary_boost:
            base_norm_for_matching = [str(x) for x in list(base_primary_norm)]  # type: ignore
        else:
            base_norm_for_matching = [str(x) for x in list(base_norm_display)]  # type: ignore
        _resolve_ambiguous_interactive(
            results,
            base_values=list(base_raw.astype(str).fillna("")),
            base_norm_display=list(base_norm_display),
            base_norm_for_matching=list(base_norm_for_matching),
            lookup_df=lookup_df,
            lookup_key_cols=lookup_keys,
            take_cols=take_cols,
            lookup_index=index,
            policy=policy,
        )

    # Copy to base (default)
    enriched = None
    if args.copy_direction in ("to-base", "both"):
        enriched = enrich(
            base_df=base_df,
            lookup_df=lookup_df,
            match_results=results,
            lookup_take_cols=take_cols,
            opts=EnrichOptions(
                prefix=args.prefix,
                diagnostics=args.diagnostics,
                overwrite_base=args.overwrite_base,
                overwrite_if_empty=args.overwrite_if_empty,
                mappings=mappings,
                new_field_mode=args.new_field,
            ),
        )

    # Optionally append unmatched rows from lookup into the enriched output
    if enriched is not None and args.append_unmatched_from_lookup:
        matched_lookup_rows = [r.row_index for r in results if r.reason == "matched" and r.row_index is not None]
        matched_set = set(matched_lookup_rows)
        if len(matched_set) < len(lookup_df):
            # Build rows aligned to base_df columns
            new_rows = []
            for idx in range(len(lookup_df)):
                if idx in matched_set:
                    continue
                row_dict = {str(c): pd.NA for c in enriched.columns}
                # Map lookup key columns to base key columns
                for b_col, l_col in zip(base_keys, lookup_keys):
                    if b_col in row_dict:
                        row_dict[b_col] = lookup_df.iloc[idx][l_col]
                # Copy identically named columns that exist in both
                for c in enriched.columns:
                    if c in lookup_df.columns:
                        row_dict[c] = lookup_df.iloc[idx][c]
                # Also copy requested take columns into identically named base columns when present
                for c in take_cols:
                    if c in enriched.columns and c in lookup_df.columns:
                        row_dict[c] = lookup_df.iloc[idx][c]
                new_rows.append(row_dict)
            if new_rows:
                add_df = pd.DataFrame(new_rows)
                enriched = pd.concat([enriched, add_df], ignore_index=True)
                print(f"Appended {len(new_rows)} unmatched lookup rows into the output.")
        else:
            print("No unmatched lookup rows to append.")

    # Optional de-duplication
    if enriched is not None and args.dedupe_by:
        dedupe_cols = [c.strip() for c in args.dedupe_by.split(",") if c.strip()]
        missing = [c for c in dedupe_cols if c not in enriched.columns]
        if missing:
            print(f"Warning: dedupe columns not in output: {', '.join(missing)}")
        enriched = enriched.drop_duplicates(subset=[c for c in dedupe_cols if c in enriched.columns], keep=args.dedupe_keep)

    # Summary
    matched = sum(1 for r in results if r.reason == "matched")
    ambiguous = sum(1 for r in results if r.reason == "ambiguous")
    unmatched = sum(1 for r in results if r.reason == "unmatched")
    total = len(results)
    print(
        f"Processed {total} rows — matched: {matched}, ambiguous: {ambiguous}, unmatched: {unmatched}"
    )

    # Write
    if enriched is not None:
        out_path = io_mod.write_table(enriched, base_input_path=args.base_path, output_path=args.output)
        print(f"Wrote: {out_path}")

    # Copy to lookup (update lookup rows with base values for selected columns)
    updated_lookup = lookup_df
    if args.copy_direction in ("to-lookup", "both"):
        updated_lookup = lookup_df.copy()
        def _is_empty(v) -> bool:
            if v is None:
                return True
            try:
                import pandas as _pd
                if _pd.isna(v):
                    return True
            except Exception:
                pass
            return str(v).strip() == ""
        updates = 0
        for i, res in enumerate(results):
            if res.row_index is None or res.reason != "matched":
                continue
            lk_i = int(res.row_index)
            for src in take_cols:
                # Map base source -> lookup target
                target = mappings.get(src, src) if mappings else src
                # Decide destination column name
                if args.new_field == "always":
                    dest = f"{args.prefix}{target}"
                    # ensure column exists
                    if dest not in updated_lookup.columns:
                        updated_lookup[dest] = pd.NA
                    value_col = dest
                    allow_overwrite = True
                elif args.new_field == "on_conflict":
                    if target in updated_lookup.columns:
                        dest = f"{args.prefix}{target}"
                        if dest not in updated_lookup.columns:
                            updated_lookup[dest] = pd.NA
                        value_col = dest
                        allow_overwrite = True
                    else:
                        if target not in updated_lookup.columns:
                            updated_lookup[target] = pd.NA
                        value_col = target
                        allow_overwrite = True
                else:  # default
                    if target in updated_lookup.columns:
                        value_col = target
                        allow_overwrite = args.overwrite_lookup
                    else:
                        dest = f"{args.prefix}{target}"
                        if dest not in updated_lookup.columns:
                            updated_lookup[dest] = pd.NA
                        value_col = dest
                        allow_overwrite = True

                if src in base_df.columns:
                    new_val = base_df.iloc[i][src]
                    if allow_overwrite:
                        if args.overwrite_lookup_if_empty:
                            if _is_empty(updated_lookup.iloc[lk_i][value_col]):
                                updated_lookup.at[lk_i, value_col] = new_val
                                updates += 1
                        else:
                            updated_lookup.at[lk_i, value_col] = new_val
                            updates += 1
                    else:
                        # Default (no overwrite): only fill empty
                        if _is_empty(updated_lookup.iloc[lk_i][value_col]):
                            updated_lookup.at[lk_i, value_col] = new_val
                            updates += 1
        print(f"Updated lookup cells: {updates}")

        # Optionally append unmatched base rows to lookup (continues to work on updated_lookup)
        if args.update_lookup:
            if len(base_keys) != len(lookup_keys):
                print(
                    f"Cannot update lookup: number of base-key columns ({len(base_keys)}) != lookup-key columns ({len(lookup_keys)})."
                )
            else:
                unmatched_indices = [i for i, r in enumerate(results) if r.reason == "unmatched"]
                if unmatched_indices:
                    new_rows = []
                    for i in unmatched_indices:
                        row = {str(c): pd.NA for c in updated_lookup.columns}
                        for b_col, l_col in zip(base_keys, lookup_keys):
                            try:
                                row[str(l_col)] = base_df.iloc[i][b_col]
                            except Exception:
                                row[str(l_col)] = pd.NA
                        if args.fill_take_from_base:
                            for c in take_cols:
                                if c in updated_lookup.columns and c in base_df.columns:
                                    try:
                                        row[c] = base_df.iloc[i][c]
                                    except Exception:
                                        row[c] = pd.NA
                        new_rows.append(row)
                    if new_rows:
                        new_df = pd.DataFrame(new_rows)
                        updated_lookup = pd.concat([updated_lookup, new_df], ignore_index=True)
                        print(f"Appended {len(new_rows)} unmatched rows to lookup.")

        # Write updated lookup
        lp = Path(args.lookup_path)
        default_out = lp.with_name(f"{lp.stem}.updated{lp.suffix}")
        lookup_out_path = args.lookup_output or str(default_out)
        out_lookup_path = io_mod.write_table(updated_lookup, base_input_path=args.lookup_path, output_path=lookup_out_path)
        print(f"Wrote updated lookup: {out_lookup_path}")

    # Optionally create/update a lookup file with unmatched entries appended
    if args.update_lookup and args.copy_direction not in ("to-lookup", "both"):
        if len(base_keys) != len(lookup_keys):
            print(
                f"Cannot update lookup: number of base-key columns ({len(base_keys)}) != lookup-key columns ({len(lookup_keys)})."
            )
            return 0
        unmatched_indices = [i for i, r in enumerate(results) if r.reason == "unmatched"]
        if unmatched_indices:
            # Build new rows as dicts matching lookup_df columns
            new_rows = []
            for i in unmatched_indices:
                row = {str(c): pd.NA for c in lookup_df.columns}
                # Map base key values into lookup key columns
                for b_col, l_col in zip(base_keys, lookup_keys):
                    try:
                        row[str(l_col)] = base_df.iloc[i][b_col]
                    except Exception:
                        row[str(l_col)] = pd.NA
                # Optionally copy take columns from base if present
                if args.fill_take_from_base:
                    for c in take_cols:
                        if c in lookup_df.columns and c in base_df.columns:
                            try:
                                row[c] = base_df.iloc[i][c]
                            except Exception:
                                row[c] = pd.NA
                new_rows.append(row)
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                updated_lookup = pd.concat([lookup_df, new_df], ignore_index=True)
                from pathlib import Path
                lp = Path(args.lookup_path)
                default_out = lp.with_name(f"{lp.stem}.updated{lp.suffix}")
                lookup_out_path = args.lookup_output or str(default_out)
                out_lookup_path = io_mod.write_table(
                    updated_lookup, base_input_path=args.lookup_path, output_path=lookup_out_path
                )
                print(
                    f"Appended {len(new_rows)} unmatched rows to lookup; wrote: {out_lookup_path}"
                )
        else:
            print("No unmatched rows; lookup was not updated.")
    return 0


def _resolve_ambiguous_interactive(
    results: List[MatchResult],
    base_values: Sequence[str],
    base_norm_display: Sequence[str],
    base_norm_for_matching: Sequence[str],
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
        if base_norm_display[i] in decision_cache:
            row_idx = decision_cache[base_norm_display[i]]
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
        print("  (normalized):", base_norm_display[i])

        cands = top_candidates(base_norm_for_matching[i], lookup_index, policy)
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
    # Parse mappings
    mappings: Optional[dict] = None
    if args.mappings:
        mappings = {}
        for tok in [t for t in args.mappings.split(",") if t.strip()]:
            if ":" not in tok:
                raise SystemExit(f"Invalid --map entry '{tok}'. Use src:dst format.")
            src, dst = tok.split(":", 1)
            src = src.strip()
            dst = dst.strip()
            if not src or not dst:
                raise SystemExit(f"Invalid --map entry '{tok}'. Use non-empty src:dst.")
            mappings[src] = dst
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
                    decision_cache[base_norm_display[i]] = row_idx
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
