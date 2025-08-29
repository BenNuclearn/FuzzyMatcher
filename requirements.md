# Requirements — CSV/XLSX Fuzzy Match Enricher

## Functional Requirements
- Load a base file and a lookup file in CSV or XLSX formats.
- Allow user to select a key column in base and a key column in lookup.
- Allow user to choose one or more columns from the lookup to bring into base.
- Perform fuzzy matching of base key values to lookup key values.
- Apply matches only when there is exactly one acceptable match (per policy):
  - Score >= threshold; and
  - No tie on top score; and
  - Optional margin over next-best candidate.
- Append selected lookup columns as new columns in the base with a prefix.
- Write an enriched output file next to the base file (CSV→CSV, XLSX→XLSX) unless an explicit output path is provided.
- Optionally add diagnostics columns (match score, matched key, match count).
- Print a summary of counts: matched, ambiguous, unmatched.

## Non-Functional Requirements
- Local CLI tool; no network access required.
- Prioritize simplicity and clarity over performance; acceptable for up to a few hundred thousand rows.
- Deterministic selection policy and consistent output order.
- Clear, actionable errors for missing columns, unreadable files, or empty inputs.
- Python 3.10+.

## Inputs
- Base file path: `.csv` or `.xlsx`.
- Lookup file path: `.csv` or `.xlsx`.
- Base key column name (string).
- Lookup key column name (string).
- Columns to take from lookup: one or many (comma-separated or repeated flags).
- Options:
  - Threshold (default 85)
  - Prefix for new columns (default `lk_`)
  - Sheet name/index for XLSX (default first sheet)
  - Normalization toggles: case-fold, trim/collapse spaces, optional alnum-only
  - Diagnostics on/off
  - Output path override

## Outputs
- Enriched base file with appended columns and same format as input base.
- Optional diagnostics columns: `match_score`, `matched_lookup_key`, `match_count`.
- Console summary of matched/ambiguous/unmatched counts.

## CLI Requirements
- Non-interactive usage with flags for all required inputs.
- Interactive prompts when key arguments are missing:
  - List columns to select base/lookup keys.
  - List columns to select which lookup columns to bring.
  - Confirm or adjust threshold.

## Matching Policy
- Normalization prior to matching: trim, collapse multiple spaces, lowercase; optional alnum-only.
- Fuzzy scorer: RapidFuzz (e.g., WRatio or token_set_ratio).
- Candidate selection:
  - Top-N candidates (default 3) from lookup for each base key.
  - Accept match if one candidate >= threshold and not tied at top; optionally require margin (3–5) over second best.
  - Otherwise mark as ambiguous; if none >= threshold, mark unmatched.

## Data Handling
- Read via pandas (`read_csv`, `read_excel`) and write via pandas.
- Treat key columns as strings for matching (`astype(str)`, `fillna("")`).
- Preserve original column order; append new columns at end.

## Error Handling
- Fail fast with clear messages when:
  - Paths don’t exist or are unreadable.
  - Column names are not found.
  - Files are empty or contain no rows/headers.
- Continue processing rows independently; never crash due to a single bad row.

## Out of Scope (Initial)
- Database/storage integration; web/GUI frontend.
- Multi-process parallelism or advanced indexing.
- Sophisticated disambiguation (beyond score/threshold/margin policy).

