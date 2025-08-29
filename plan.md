# CSV/XLSX Fuzzy Match Enricher — Plan

## Goal
- Take a base CSV/XLSX file and a lookup CSV/XLSX file.
- Choose a key column in the base and a key column in the lookup.
- Fuzzy match base keys to lookup keys.
- If there is exactly one sufficiently good match, copy selected lookup columns into new columns in the base.
- Produce an enriched base file plus optional diagnostics (match score, matched value, match count).

## Scope & Assumptions
- Personal-use CLI, not production; happy path prioritized.
- Small to medium datasets (up to a few hundred thousand rows) are expected; we optimize for clarity over maximum performance.
- Headers are present on the first row; first sheet used for XLSX unless specified.
- Matching is done on string representations of the key columns.
- Dependencies: Python 3.10+, `pandas`, `openpyxl` (for XLSX), `rapidfuzz` for fuzzy matching.

## Inputs
- Base file: CSV or XLSX path.
- Lookup file: CSV or XLSX path.
- Base key column name.
- Lookup key column name.
- Columns to bring from lookup (one or more).
- Options: match threshold (default 85), top-N candidate window (default 3), case/whitespace normalization toggles.

## Outputs
- Enriched base file written alongside the input (e.g., `base.enriched.csv` or `base.enriched.xlsx`).
- Added columns include selected lookup columns with a configurable prefix (default `lk_`).
- Optional diagnostics columns: `match_score`, `matched_lookup_key`, `match_count`.
- A report summary printed to stdout (rows matched, rows ambiguous, rows unmatched).

## CLI Design (simple and interactive-friendly)
- Command: `fmatch <base_path> <lookup_path> \
  --base-key <name> --lookup-key <name> \
  --take <col1,col2,...> [--threshold 85] [--prefix lk_] [--xlsx-sheet SHEET] \
  [--normalize case,space,alnum] [--output <path>] [--diagnostics]`.
- If required args are missing, launch interactive prompts to list columns and accept choices.
- Detect format by extension; for XLSX, allow `--xlsx-sheet` (default first sheet).

## Matching Algorithm
- Preprocess keys (based on flags):
  - Trim whitespace; collapse multiple spaces.
  - Case-fold or lower.
  - Optional: keep only alphanumeric.
- Use `rapidfuzz.process.extract` or `extractOne` with a scorer (e.g., `WRatio` or `token_set_ratio`).
- Strategy:
  - For each base key, compute top-N matches from lookup keys and scores.
  - Accept if exactly one candidate >= threshold AND the next-best is sufficiently lower (tie-break guard), or if only one candidate exists.
  - Otherwise mark as ambiguous or unmatched; do not enrich.
- Store the matched lookup row index for fast column retrieval.

## Data Loading & Writing
- Use `pandas.read_csv` / `pandas.read_excel` for input; infer dtypes but treat keys as string via `astype(str)` post-load.
- For writing:
  - CSV by default if base is CSV; XLSX if base is XLSX.
  - Preserve original column order; append new columns at the end.

## Edge Cases & Handling
- Missing base key values: treat as unmatched.
- Duplicate keys in lookup: will create ambiguity; unless a single best score emerges with clear margin, mark ambiguous.
- Multiple matches with same top score: ambiguous.
- NaNs and non-strings: cast to string after fillna("") for matching; retain original when writing.
- Non-existent column names: fail fast with helpful error.

## Minimal Architecture
- `cli.py`: argument parsing + interactive prompts.
- `io.py`: load/save CSV/XLSX, sheet selection, output naming.
- `match.py`: normalization utilities, fuzzy matching, selection logic.
- `enrich.py`: join selected columns based on match results.
- `__main__.py`: entrypoint wiring CLI to functions.

## Implementation Steps
1. CLI skeleton and argument parsing.
2. IO module: load CSV/XLSX and list column names; write enriched output.
3. Normalization functions and matcher using `rapidfuzz`.
4. Enrichment: apply matches and append selected columns with prefix; diagnostics.
5. Interactive prompts for missing args (list columns, confirm threshold).
6. Smoke tests with tiny sample files and a few edge cases.

## Defaults & Config
- Threshold: 85; scorer: `WRatio` (or `token_set_ratio` for mixed token order).
- Top-N: 3; tie-break margin: 3–5 points.
- Prefix: `lk_`; diagnostics: off by default, toggle with `--diagnostics`.

## Example Usage
- Non-interactive:
  - `fmatch base.xlsx lookup.csv --base-key Name --lookup-key FullName --take email,company --threshold 88 --diagnostics`
- Interactive (only paths given):
  - `fmatch base.csv lookup.xlsx`
  - Tool lists columns for selection and asks for threshold.

## Future Enhancements (nice-to-have)
- Output ambiguous/unmatched rows to separate files for manual review.
- Caching index or n-gram filter for large lookups.
- Simple GUI wrapper.
- Config file support for repeated runs.

