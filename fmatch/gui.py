from __future__ import annotations

import sys
try:  # Provide a friendly error if Tk is unavailable
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception as _e:  # pragma: no cover
    raise SystemExit(
        "Tkinter is not available in this Python. Install a Python build with Tk support "
        "(e.g., python.org installer on macOS), or rebuild your pyenv Python with Tcl/Tk.\n"
        "Alternatively use the CLI: `python -m fmatch ...`"
    ) from _e
from contextlib import redirect_stdout
from io import StringIO
from typing import List

import pandas as pd

from . import io as io_mod
from .cli import build_parser, _composite_key_series
from .match import (
    NormalizeOptions,
    MatchPolicy,
    normalize_series,
    build_lookup_index,
    match_one,
    match_one_with_secondary,
    top_candidates,
    MatchResult,
)
from .enrich import EnrichOptions, enrich


SCORERS = [
    "smart",
    "WRatio",
    "ratio",
    "token_set_ratio",
    "token_sort_ratio",
    "partial_ratio",
    "partial_token_set_ratio",
]


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("fmatch — Fuzzy Match Enricher")
        self.geometry("900x650")

        self.base_path = tk.StringVar()
        self.lookup_path = tk.StringVar()
        self.sheet = tk.StringVar()
        self.output_path = tk.StringVar()
        self.lookup_output_path = tk.StringVar()
        self.prefix = tk.StringVar(value="lk_")

        self.threshold = tk.IntVar(value=85)
        self.top_n = tk.IntVar(value=3)
        self.margin = tk.IntVar(value=3)
        self.scorer = tk.StringVar(value="smart")

        self.diagnostics = tk.BooleanVar(value=False)
        self.casefold = tk.BooleanVar(value=True)
        self.trim = tk.BooleanVar(value=True)
        self.alnum_only = tk.BooleanVar(value=False)
        self.overwrite_base = tk.BooleanVar(value=False)
        self.overwrite_if_empty = tk.BooleanVar(value=False)
        self.update_lookup = tk.BooleanVar(value=False)
        self.fill_take_from_base = tk.BooleanVar(value=False)
        self.append_unmatched_from_lookup = tk.BooleanVar(value=False)
        self.dedupe_keep = tk.StringVar(value="first")
        self.dedupe_by = tk.StringVar(value="")
        # Copy direction controls
        self.copy_direction = tk.StringVar(value="to-base")  # to-base | to-lookup | both
        self.overwrite_lookup = tk.BooleanVar(value=False)
        self.overwrite_lookup_if_empty = tk.BooleanVar(value=False)

        self.base_cols: List[str] = []
        self.lookup_cols: List[str] = []

        # Primary selectors for interactive ordering
        self.base_primary = tk.StringVar(value="<auto>")
        self.lookup_primary = tk.StringVar(value="<auto>")
        self._om_base_primary = None  # type: ignore
        self._om_lookup_primary = None  # type: ignore

        self._build_ui()

    def _build_ui(self) -> None:
        frm = tk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # File pickers
        row = 0
        tk.Label(frm, text="Base file").grid(row=row, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.base_path, width=70).grid(row=row, column=1, sticky="we")
        tk.Button(frm, text="Browse", command=self._pick_base).grid(row=row, column=2)
        row += 1
        tk.Label(frm, text="Lookup file").grid(row=row, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.lookup_path, width=70).grid(row=row, column=1, sticky="we")
        tk.Button(frm, text="Browse", command=self._pick_lookup).grid(row=row, column=2)
        row += 1
        tk.Label(frm, text="Sheet (optional)").grid(row=row, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.sheet, width=30).grid(row=row, column=1, sticky="w")
        tk.Button(frm, text="Load Columns", command=self._load_columns).grid(row=row, column=2)

        # Column selectors
        row += 1
        col_frame = tk.Frame(frm)
        col_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(8, 4))
        frm.grid_rowconfigure(row, weight=1)
        for c in range(3):
            col_frame.grid_columnconfigure(c, weight=1)
        tk.Label(col_frame, text="Base key columns (multi-select)").grid(row=0, column=0)
        tk.Label(col_frame, text="Lookup key columns (multi-select)").grid(row=0, column=1)
        tk.Label(col_frame, text="Lookup take columns (multi-select)").grid(row=0, column=2)
        self.lb_base = tk.Listbox(col_frame, selectmode=tk.EXTENDED, exportselection=False)
        self.lb_lookup = tk.Listbox(col_frame, selectmode=tk.EXTENDED, exportselection=False)
        self.lb_take = tk.Listbox(col_frame, selectmode=tk.EXTENDED, exportselection=False)
        self.lb_base.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.lb_lookup.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.lb_take.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)

        # Primary key pickers (to set order without editing files)
        row += 1
        prim = tk.Frame(frm)
        prim.grid(row=row, column=0, columnspan=3, sticky="we", pady=(2, 2))
        tk.Label(prim, text="Primary base key").grid(row=0, column=0, sticky="w")
        self._om_base_primary = tk.OptionMenu(prim, self.base_primary, "<auto>")
        self._om_base_primary.grid(row=0, column=1, sticky="w")
        tk.Label(prim, text="Primary lookup key").grid(row=0, column=2, sticky="w")
        self._om_lookup_primary = tk.OptionMenu(prim, self.lookup_primary, "<auto>")
        self._om_lookup_primary.grid(row=0, column=3, sticky="w")
        tk.Button(prim, text="Sync from selection", command=self._sync_primary_options).grid(row=0, column=4, padx=6)

        # Options
        row += 1
        opt = tk.Frame(frm)
        opt.grid(row=row, column=0, columnspan=3, sticky="we", pady=(6, 6))
        # Row 1
        tk.Label(opt, text="Scorer").grid(row=0, column=0, sticky="w")
        tk.OptionMenu(opt, self.scorer, *SCORERS).grid(row=0, column=1, sticky="w")
        tk.Label(opt, text="Threshold").grid(row=0, column=2, sticky="w")
        tk.Spinbox(opt, from_=0, to=100, textvariable=self.threshold, width=5).grid(row=0, column=3, sticky="w")
        tk.Label(opt, text="Top-N").grid(row=0, column=4, sticky="w")
        tk.Spinbox(opt, from_=1, to=100, textvariable=self.top_n, width=5).grid(row=0, column=5, sticky="w")
        tk.Label(opt, text="Margin").grid(row=0, column=6, sticky="w")
        tk.Spinbox(opt, from_=0, to=20, textvariable=self.margin, width=5).grid(row=0, column=7, sticky="w")

        # Row 2
        tk.Checkbutton(opt, text="Diagnostics", variable=self.diagnostics).grid(row=1, column=0, sticky="w")
        tk.Checkbutton(opt, text="Casefold", variable=self.casefold).grid(row=1, column=1, sticky="w")
        tk.Checkbutton(opt, text="Trim", variable=self.trim).grid(row=1, column=2, sticky="w")
        tk.Checkbutton(opt, text="Alnum-only", variable=self.alnum_only).grid(row=1, column=3, sticky="w")
        tk.Label(opt, text="Prefix").grid(row=1, column=4, sticky="e")
        tk.Entry(opt, textvariable=self.prefix, width=8).grid(row=1, column=5, sticky="w")
        tk.Label(opt, text="New field").grid(row=1, column=6, sticky="e")
        tk.OptionMenu(opt, self.new_field_mode, "default", "always", "on_conflict").grid(row=1, column=7, sticky="w")

        # Row 3
        tk.Checkbutton(opt, text="Overwrite base", variable=self.overwrite_base).grid(row=2, column=0, sticky="w")
        tk.Checkbutton(opt, text="Only if empty", variable=self.overwrite_if_empty).grid(row=2, column=1, sticky="w")
        tk.Checkbutton(opt, text="Update lookup (append unmatched)", variable=self.update_lookup).grid(row=2, column=2, columnspan=2, sticky="w")
        tk.Checkbutton(opt, text="Fill take from base", variable=self.fill_take_from_base).grid(row=2, column=4, columnspan=2, sticky="w")
        tk.Checkbutton(opt, text="Append unmatched from lookup", variable=self.append_unmatched_from_lookup).grid(row=2, column=6, columnspan=2, sticky="w")
        tk.Label(opt, text="Copy dir").grid(row=2, column=8, sticky="w")
        tk.OptionMenu(opt, self.copy_direction, "to-base", "to-lookup", "both").grid(row=2, column=9, sticky="w")
        tk.Checkbutton(opt, text="Overwrite lookup", variable=self.overwrite_lookup).grid(row=2, column=10, sticky="w")
        tk.Checkbutton(opt, text="Lookup only if empty", variable=self.overwrite_lookup_if_empty).grid(row=2, column=11, sticky="w")

        # Row 4
        tk.Label(opt, text="Output (optional)").grid(row=3, column=0, sticky="w")
        tk.Entry(opt, textvariable=self.output_path, width=45).grid(row=3, column=1, columnspan=3, sticky="we")
        tk.Button(opt, text="Browse", command=self._pick_output).grid(row=3, column=4)
        tk.Label(opt, text="Lookup output (optional)").grid(row=3, column=5, sticky="w")
        tk.Entry(opt, textvariable=self.lookup_output_path, width=30).grid(row=3, column=6, sticky="we")
        tk.Button(opt, text="Browse", command=self._pick_lookup_output).grid(row=3, column=7)
        tk.Label(opt, text="Mappings (src:dst,...) ").grid(row=3, column=8, sticky="w")
        tk.Entry(opt, textvariable=self.mappings, width=28).grid(row=3, column=9, columnspan=2, sticky="we")

        # Row 5
        tk.Label(opt, text="Dedupe by (comma names)").grid(row=4, column=0, sticky="w")
        tk.Entry(opt, textvariable=self.dedupe_by, width=40).grid(row=4, column=1, columnspan=3, sticky="we")
        tk.Label(opt, text="Keep").grid(row=4, column=4, sticky="e")
        tk.OptionMenu(opt, self.dedupe_keep, "first", "last").grid(row=4, column=5, sticky="w")

        # Run and log
        run_frame = tk.Frame(frm)
        run_frame.grid(row=row + 1, column=0, columnspan=3, sticky="we")
        tk.Button(run_frame, text="Run", command=self._run).pack(side=tk.LEFT)
        tk.Label(run_frame, text="Logs").pack(side=tk.LEFT, padx=10)
        self.log = tk.Text(frm, height=12)
        self.log.grid(row=row + 2, column=0, columnspan=3, sticky="nsew")
        frm.grid_rowconfigure(row + 2, weight=1)

    def _pick_base(self) -> None:
        path = filedialog.askopenfilename(title="Select base CSV/XLSX", filetypes=[("Spreadsheet", "*.csv *.xlsx *.xlsm *.xltx *.xltm"), ("All", "*.*")])
        if path:
            self.base_path.set(path)

    def _pick_lookup(self) -> None:
        path = filedialog.askopenfilename(title="Select lookup CSV/XLSX", filetypes=[("Spreadsheet", "*.csv *.xlsx *.xlsm *.xltx *.xltm"), ("All", "*.*")])
        if path:
            self.lookup_path.set(path)

    def _pick_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Select output path", defaultextension=".xlsx")
        if path:
            self.output_path.set(path)

    def _pick_lookup_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Select updated lookup output path", defaultextension=".xlsx")
        if path:
            self.lookup_output_path.set(path)

    def _load_columns(self) -> None:
        try:
            sheet = self.sheet.get().strip() or None
            base_df = io_mod.read_table(self.base_path.get(), sheet=sheet)
            lookup_df = io_mod.read_table(self.lookup_path.get(), sheet=sheet)
        except Exception as e:
            messagebox.showerror("Error loading files", str(e))
            return
        self.base_cols = [str(c) for c in base_df.columns]
        self.lookup_cols = [str(c) for c in lookup_df.columns]
        # Populate listboxes
        self.lb_base.delete(0, tk.END)
        self.lb_lookup.delete(0, tk.END)
        self.lb_take.delete(0, tk.END)
        for c in self.base_cols:
            self.lb_base.insert(tk.END, c)
        for c in self.lookup_cols:
            self.lb_lookup.insert(tk.END, c)
            self.lb_take.insert(tk.END, c)
        # Reset primary menus
        self.base_primary.set("<auto>")
        self.lookup_primary.set("<auto>")
        self._sync_primary_options()

    def _get_selected(self, lb: tk.Listbox, cols: List[str]) -> List[str]:
        sel = lb.curselection()
        return [cols[i] for i in sel]

    # Utility: update primary OptionMenus to reflect current selection
    def _sync_primary_options(self) -> None:
        base_sel = self._get_selected(self.lb_base, self.base_cols)
        lookup_sel = self._get_selected(self.lb_lookup, self.lookup_cols)
        base_opts = base_sel[:] if len(base_sel) >= 1 else ["<auto>"]
        lookup_opts = lookup_sel[:] if len(lookup_sel) >= 1 else ["<auto>"]
        # Rebuild base OptionMenu
        try:
            self._om_base_primary["menu"].delete(0, "end")
        except Exception:
            pass
        for opt in base_opts:
            self._om_base_primary["menu"].add_command(
                label=opt, command=tk._setit(self.base_primary, opt)
            )
        if self.base_primary.get() not in base_opts:
            self.base_primary.set(base_opts[0])
        # Rebuild lookup OptionMenu
        try:
            self._om_lookup_primary["menu"].delete(0, "end")
        except Exception:
            pass
        for opt in lookup_opts:
            self._om_lookup_primary["menu"].add_command(
                label=opt, command=tk._setit(self.lookup_primary, opt)
            )
        if self.lookup_primary.get() not in lookup_opts:
            self.lookup_primary.set(lookup_opts[0])

    def _run(self) -> None:
        # In-GUI pipeline with ambiguity picker
        try:
            base_path = self.base_path.get().strip()
            lookup_path = self.lookup_path.get().strip()
            if not base_path or not lookup_path:
                raise ValueError("Please select both base and lookup files.")
            sheet = self.sheet.get().strip() or None
            base_df = io_mod.read_table(base_path, sheet=sheet)
            lookup_df = io_mod.read_table(lookup_path, sheet=sheet)

            base_keys = self._get_selected(self.lb_base, self.base_cols)
            lookup_keys = self._get_selected(self.lb_lookup, self.lookup_cols)
            take_cols = self._get_selected(self.lb_take, self.lookup_cols)
            if not base_keys or not lookup_keys:
                raise ValueError("Please select base and lookup key columns.")
            if not take_cols:
                raise ValueError("Please select at least one 'take' column.")

            # Ensure primary selection options reflect current selection
            self._sync_primary_options()
            # Reorder keys to place chosen primary first (if available)
            def _apply_primary_order(keys: List[str], primary_name: str) -> List[str]:
                if len(keys) >= 2 and primary_name and primary_name not in ("<auto>", "") and primary_name in keys:
                    return [primary_name] + [k for k in keys if k != primary_name]
                return keys
            base_keys = _apply_primary_order(base_keys, self.base_primary.get())
            lookup_keys = _apply_primary_order(lookup_keys, self.lookup_primary.get())

            norm = NormalizeOptions(
                casefold=self.casefold.get(),
                trim_space=self.trim.get(),
                collapse_space=self.trim.get(),
                alnum_only=self.alnum_only.get(),
            )
            policy = MatchPolicy(
                threshold=int(self.threshold.get()),
                top_n=int(self.top_n.get()),
                tie_margin=int(self.margin.get()),
                scorer=self.scorer.get(),
            )

            use_secondary_boost = len(base_keys) >= 2 and len(lookup_keys) >= 2
            if use_secondary_boost:
                # Primary-only matching with secondary boost
                base_primary_raw = base_df[base_keys[0]].astype(str).fillna("")
                lookup_primary_raw = lookup_df[lookup_keys[0]].astype(str).fillna("")
                base_primary_norm = normalize_series(base_primary_raw, norm)
                lookup_primary_norm = normalize_series(lookup_primary_raw, norm)
                index = build_lookup_index(list(lookup_primary_norm))

                def _norm_tuple_cols(df: pd.DataFrame, cols: List[str]) -> List[tuple]:
                    if not cols:
                        return [("") for _ in range(len(df))]  # type: ignore
                    normed = [normalize_series(df[c].astype(str).fillna(""), norm) for c in cols]
                    out: List[tuple] = []
                    for i in range(len(df)):
                        out.append(tuple(str(n.iloc[i]) for n in normed))
                    return out
                base_secondary_norm_list = _norm_tuple_cols(base_df, base_keys[1:])
                lookup_secondary_norm_list = _norm_tuple_cols(lookup_df, lookup_keys[1:])

                base_raw = _composite_key_series(base_df, base_keys)
                base_norm_display = normalize_series(base_raw, norm)

                results: List[MatchResult] = []
                for i, b in enumerate(base_primary_norm):
                    results.append(
                        match_one_with_secondary(
                            base_primary_norm=str(b),
                            base_secondary_norm=tuple(base_secondary_norm_list[i]) if base_secondary_norm_list else None,
                            lookup_key_to_rows=index,
                            lookup_secondary_norm_rows=lookup_secondary_norm_list,
                            policy=policy,
                            secondary_boost=10,
                        )
                    )
            else:
                base_raw = _composite_key_series(base_df, base_keys)
                lookup_raw = _composite_key_series(lookup_df, lookup_keys)
                base_norm_display = normalize_series(base_raw, norm)
                lookup_norm = normalize_series(lookup_raw, norm)
                index = build_lookup_index(list(lookup_norm))

                results: List[MatchResult] = []
                for _, b in base_norm_display.items():
                    results.append(match_one(b, index, policy))

            # Resolve ambiguous via GUI picker with decision cache
            self._resolve_ambiguous_gui(
                results,
                base_values=list(base_raw.astype(str).fillna("")),
                base_norm_display=list(base_norm_display),
                base_norm_for_matching=(list(base_primary_norm) if use_secondary_boost else list(base_norm_display)),
                lookup_df=lookup_df,
                lookup_key_cols=lookup_keys,
                take_cols=take_cols,
                lookup_index=index,
                policy=policy,
            )

            # Copy to base (default)
            enriched = None
            if self.copy_direction.get() in ("to-base", "both"):
                enriched = enrich(
                    base_df=base_df,
                    lookup_df=lookup_df,
                    match_results=results,
                    lookup_take_cols=take_cols,
                    opts=EnrichOptions(
                        prefix=self.prefix.get(),
                        diagnostics=self.diagnostics.get(),
                        overwrite_base=self.overwrite_base.get(),
                        overwrite_if_empty=self.overwrite_if_empty.get(),
                        mappings=self._parse_mappings(),
                        new_field_mode=self.new_field_mode.get(),
                    ),
                )

            # Append unmatched from lookup if requested
            if enriched is not None and self.append_unmatched_from_lookup.get():
                matched_lookup_rows = [
                    r.row_index for r in results if r.reason == "matched" and r.row_index is not None
                ]
                matched_set = set(matched_lookup_rows)
                if len(matched_set) < len(lookup_df):
                    new_rows = []
                    for idx in range(len(lookup_df)):
                        if idx in matched_set:
                            continue
                        row_dict = {str(c): pd.NA for c in enriched.columns}
                        for b_col, l_col in zip(base_keys, lookup_keys):
                            if b_col in row_dict:
                                row_dict[b_col] = lookup_df.iloc[idx][l_col]
                        for c in enriched.columns:
                            if c in lookup_df.columns:
                                row_dict[c] = lookup_df.iloc[idx][c]
                        for c in take_cols:
                            if c in enriched.columns and c in lookup_df.columns:
                                row_dict[c] = lookup_df.iloc[idx][c]
                        new_rows.append(row_dict)
                    if new_rows:
                        add_df = pd.DataFrame(new_rows)
                        enriched = pd.concat([enriched, add_df], ignore_index=True)
                        self._log(f"Appended {len(new_rows)} unmatched lookup rows into the output.\n")

            # Dedupe
            dcols = [c.strip() for c in self.dedupe_by.get().split(",") if c.strip()]
            if enriched is not None and dcols:
                missing = [c for c in dcols if c not in enriched.columns]
                if missing:
                    self._log(f"Warning: dedupe columns not in output: {', '.join(missing)}\n")
                enriched = enriched.drop_duplicates(
                    subset=[c for c in dcols if c in enriched.columns], keep=self.dedupe_keep.get()
                )

            # Summary
            matched = sum(1 for r in results if r.reason == "matched")
            ambiguous = sum(1 for r in results if r.reason == "ambiguous")
            unmatched = sum(1 for r in results if r.reason == "unmatched")
            total = len(results)
            self._log(f"Processed {total} rows — matched: {matched}, ambiguous: {ambiguous}, unmatched: {unmatched}\n")

            # Write enriched output
            if enriched is not None:
                out_path = io_mod.write_table(
                    enriched, base_input_path=base_path, output_path=(self.output_path.get().strip() or None)
                )
                self._log(f"Wrote: {out_path}\n")

            # Copy to lookup when requested (update lookup rows with base values for selected columns)
            if self.copy_direction.get() in ("to-lookup", "both"):
                updated_lookup = lookup_df.copy()
                def _is_empty(v) -> bool:
                    try:
                        import pandas as _pd
                        if _pd.isna(v):
                            return True
                    except Exception:
                        pass
                    return str(v).strip() == ""
                updates = 0
                mappings = self._parse_mappings()
                for i, res in enumerate(results):
                    if res.row_index is None or res.reason != "matched":
                        continue
                    lk_i = int(res.row_index)
                    for src in take_cols:
                        target = mappings.get(src, src) if mappings else src
                        # Decide destination column name
                        mode = self.new_field_mode.get()
                        if mode == "always":
                            dest = f"{self.prefix.get()}{target}"
                            if dest not in updated_lookup.columns:
                                updated_lookup[dest] = pd.NA
                            value_col = dest
                            allow_overwrite = True
                        elif mode == "on_conflict":
                            if target in updated_lookup.columns:
                                dest = f"{self.prefix.get()}{target}"
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
                                allow_overwrite = self.overwrite_lookup.get()
                            else:
                                dest = f"{self.prefix.get()}{target}"
                                if dest not in updated_lookup.columns:
                                    updated_lookup[dest] = pd.NA
                                value_col = dest
                                allow_overwrite = True

                        if src in base_df.columns:
                            new_val = base_df.iloc[i][src]
                            if allow_overwrite:
                                if self.overwrite_lookup_if_empty.get():
                                    if _is_empty(updated_lookup.iloc[lk_i][value_col]):
                                        updated_lookup.at[lk_i, value_col] = new_val
                                        updates += 1
                                else:
                                    updated_lookup.at[lk_i, value_col] = new_val
                                    updates += 1
                            else:
                                if _is_empty(updated_lookup.iloc[lk_i][value_col]):
                                    updated_lookup.at[lk_i, value_col] = new_val
                                    updates += 1
                self._log(f"Updated lookup cells: {updates}\n")

                # Optionally append unmatched base rows to lookup
                if self.update_lookup.get():
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
                            if self.fill_take_from_base.get():
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

                # Write updated lookup
                from pathlib import Path
                lp = Path(lookup_path)
                default_out = lp.with_name(f"{lp.stem}.updated{lp.suffix}")
                lookup_out_path = self.lookup_output_path.get().strip() or str(default_out)
                out_lookup_path = io_mod.write_table(
                    updated_lookup, base_input_path=lookup_path, output_path=lookup_out_path
                )
                self._log(f"Wrote updated lookup: {out_lookup_path}\n")

            # Optionally update lookup with unmatched base rows
            if self.update_lookup.get():
                unmatched_indices = [i for i, r in enumerate(results) if r.reason == "unmatched"]
                if unmatched_indices:
                    new_rows = []
                    for i in unmatched_indices:
                        row = {str(c): pd.NA for c in lookup_df.columns}
                        for b_col, l_col in zip(base_keys, lookup_keys):
                            try:
                                row[str(l_col)] = base_df.iloc[i][b_col]
                            except Exception:
                                row[str(l_col)] = pd.NA
                        if self.fill_take_from_base.get():
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
                        lp = Path(lookup_path)
                        default_out = lp.with_name(f"{lp.stem}.updated{lp.suffix}")
                        lookup_out_path = self.lookup_output_path.get().strip() or str(default_out)
                        out_lookup_path = io_mod.write_table(
                            updated_lookup, base_input_path=lookup_path, output_path=lookup_out_path
                        )
                        self._log(
                            f"Appended {len(new_rows)} unmatched rows to lookup; wrote: {out_lookup_path}\n"
                        )
                else:
                    self._log("No unmatched rows; lookup was not updated.\n")

        except ValueError as ve:
            messagebox.showerror("Invalid mapping", str(ve))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _log(self, text: str) -> None:
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

    def _parse_mappings(self):
        s = (self.mappings.get() or "").strip()
        if not s:
            return None
        out = {}
        toks = [t for t in s.split(",") if t.strip()]
        for tok in toks:
            if ":" not in tok:
                raise ValueError(f"Invalid mapping '{tok}'. Use src:dst.")
            src, dst = tok.split(":", 1)
            src = src.strip()
            dst = dst.strip()
            if not src or not dst:
                raise ValueError(f"Invalid mapping '{tok}'. Use non-empty src:dst.")
            out[src] = dst
        return out

    def _resolve_ambiguous_gui(
        self,
        results: List[MatchResult],
        base_values: List[str],
        base_norm_display: List[str],
        base_norm_for_matching: List[str],
        lookup_df: pd.DataFrame,
        lookup_key_cols: List[str],
        take_cols: List[str],
        lookup_index,
        policy: MatchPolicy,
    ) -> None:
        decision_cache: dict[str, int] = {}
        for i, res in enumerate(results):
            if res.reason != "ambiguous":
                continue
            if base_norm_display[i] in decision_cache:
                row_idx = decision_cache[base_norm_display[i]]
                norm_key = None
                for k, rows in lookup_index.items():
                    if row_idx in rows:
                        norm_key = k
                        break
                results[i] = MatchResult(
                    row_index=row_idx,
                    score=None,
                    matched_key=norm_key,
                    candidate_count=res.candidate_count,
                    reason="matched",
                )
                continue
            dlg = CandidateDialog(
                self,
                base_values[i],
                base_norm_display[i],
                base_norm_for_matching[i],
                lookup_df,
                lookup_key_cols,
                take_cols,
                lookup_index,
                policy,
            )
            sel = dlg.show()
            if sel is None:
                continue
            row_idx, norm_key, score = sel
            results[i] = MatchResult(
                row_index=row_idx,
                score=score,
                matched_key=norm_key,
                candidate_count=res.candidate_count,
                reason="matched",
            )
            decision_cache[base_norm_display[i]] = row_idx


class CandidateDialog(tk.Toplevel):
    def __init__(
        self,
        master: tk.Misc,
        base_value: str,
        base_norm_display: str,
        base_norm_for_matching: str,
        lookup_df: pd.DataFrame,
        lookup_key_cols: List[str],
        take_cols: List[str],
        lookup_index,
        policy: MatchPolicy,
    ) -> None:
        super().__init__(master)
        self.title("Resolve Ambiguous Match")
        self.transient(master)
        self.grab_set()
        self.resizable(True, True)

        self.lookup_df = lookup_df
        self.lookup_key_cols = lookup_key_cols
        self.take_cols = take_cols
        self.lookup_index = lookup_index
        self.policy = policy
        self.base_norm_display = base_norm_display
        self.base_norm_for_matching = base_norm_for_matching

        tk.Label(self, text=f"Base: {base_value}").pack(anchor="w", padx=10, pady=(10, 0))
        tk.Label(self, text=f"(normalized): {base_norm_display}").pack(anchor="w", padx=10)

        srch_frame = tk.Frame(self)
        srch_frame.pack(fill=tk.X, padx=10, pady=(6, 4))
        tk.Label(srch_frame, text="Search").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        tk.Entry(srch_frame, textvariable=self.search_var, width=40).pack(side=tk.LEFT, padx=5)
        tk.Button(srch_frame, text="Search", command=self._search).pack(side=tk.LEFT)

        self.list = tk.Listbox(self, width=100, height=12)
        self.list.pack(fill=tk.BOTH, expand=True, padx=10)

        btns = tk.Frame(self)
        btns.pack(fill=tk.X, padx=10, pady=10)
        tk.Button(btns, text="Accept", command=self._accept).pack(side=tk.LEFT)
        tk.Button(btns, text="Skip", command=self._skip).pack(side=tk.RIGHT)

        self._candidates = self._get_candidates()
        self._render(self._candidates)
        self._selection = None  # type: ignore

    def _get_candidates(self):
        return top_candidates(self.base_norm_for_matching, self.lookup_index, self.policy)

    

    def _render(self, cands):
        self.list.delete(0, tk.END)
        for j, (row_idx, norm_key, score) in enumerate(cands):
            preview = []
            try:
                orig_key = " | ".join([str(self.lookup_df.iloc[row_idx][k]) for k in self.lookup_key_cols])
            except Exception:
                orig_key = "<err>"
            for c in self.take_cols[:3]:
                try:
                    preview.append(f"{c}={self.lookup_df.iloc[row_idx][c]}")
                except Exception:
                    preview.append(f"{c}=<na>")
            preview_str = ", ".join(preview)
            self.list.insert(tk.END, f"[{j}] score={score:.1f} key='{orig_key}' {('('+preview_str+')') if preview else ''}")
        self._display = list(cands)

    def _search(self):
        try:
            from rapidfuzz import process as _rf_process  # type: ignore
            from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
        except Exception:
            messagebox.showwarning("Search unavailable", "rapidfuzz not installed; search disabled")
            return
        all_keys = list(self.lookup_index.keys())
        q = (self.search_var.get() or "").strip().casefold()
        if not q:
            return
        found = _rf_process.extract(q, all_keys, scorer=_rf_fuzz.token_set_ratio, limit=15)
        existing_rows = {r for r, _, _ in getattr(self, "_display", [])}
        for k, s, _ in found:
            for row_idx in self.lookup_index.get(k, []):
                if row_idx in existing_rows:
                    continue
                self._display.append((row_idx, k, float(s)))
                existing_rows.add(row_idx)
        self._render(self._display)

    def _accept(self):
        sel = self.list.curselection()
        if not sel:
            messagebox.showinfo("Select a candidate", "Please select a candidate to accept.")
            return
        idx = sel[0]
        row_idx, norm_key, score = self._display[idx]
        self._selection = (row_idx, norm_key, score)
        self.destroy()

    def _skip(self):
        self._selection = None
        self.destroy()

    def show(self):
        self.wait_window(self)
        return self._selection


def main() -> int:
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
