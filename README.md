# fmatch — CSV/XLSX Fuzzy Match Enricher

Small CLI to enrich a “base” CSV/XLSX with columns from a “lookup” file using fuzzy matching of key values.

- Reads CSV or XLSX (first sheet by default or `--sheet`)
- Normalizes text (trim/collapse/casefold; optional alnum-only)
- Uses RapidFuzz with a deterministic selection policy; default scorer `smart` (acronym/segment aware)
- Appends selected lookup columns to base, with prefix (default `lk_`)
- Optional diagnostics columns and concise summary output

## Requirements

- Python 3.13.7
- Pipenv

Install deps into a Pipenv environment pinned to Python 3.13.7:

```bash
pipenv --python 3.13.7 install --deploy
```

## Quick Start

Show help:

```bash
pipenv run python -m fmatch --help
```

Try the included sample data and write an enriched CSV next to it:

```bash
pipenv run python scripts/smoke_e2e.py
# or directly:
pipenv run python -m fmatch samples/base.csv samples/lookup.csv \
  --base-key Name --lookup-key FullName --take email,company
```

Typical usage with your own files:

```bash
# Single key
pipenv run python -m fmatch base.csv lookup.xlsx \
  --base-key Name --lookup-key FullName \
  --take email,company --threshold 88

# Multiple keys (composite match): comma-separate columns
pipenv run python -m fmatch base.csv lookup.xlsx \
  --base-key Name,ID --lookup-key FullName,ID \
  --take email,company

# Resolve ambiguous rows interactively (default)
pipenv run python -m fmatch base.csv lookup.xlsx \
  --base-key Name --lookup-key FullName --take email
  # add --no-resolve-ambiguous to disable prompts
```

If you omit `--base-key`, `--lookup-key` or `--take`, the CLI will list columns and prompt you to select them interactively.

## Output

- Writes next to the base input by default, preserving extension:
  - `base.csv` → `base.enriched.csv`
  - `base.xlsx` → `base.enriched.xlsx`
- Override with `--output <path>`.
- New columns are prefixed (default `lk_`), e.g. `lk_email`.
- Enable diagnostics with `--diagnostics` to add: `match_score`, `matched_lookup_key`, `match_count`.
- Interactive resolution (default):
  - Prompts you to choose a candidate for rows that were ambiguous under the current policy.
  - Use `--no-resolve-ambiguous` to disable and leave ambiguous rows unresolved.

## Ambiguity Resolution

- Enabled by default; disable with `--no-resolve-ambiguous`. For each ambiguous base row the CLI shows:
  - Base value and its normalized form.
  - A numbered list of candidate lookup rows with scores and a short preview of the first few `--take` columns.
- Choose a number to accept a candidate; press Enter to skip and leave ambiguous.
- Repeated identical base values: after you resolve once, identical future rows auto-resolve to the same choice in this run.
- Inline search: at the prompt, type `s <query>` (or `/query`) to search the lookup and add results to the candidate list.
- Tips if you see too many/too few candidates:
  - Lower `--threshold` (e.g., 80) to include more candidates.
  - Adjust `--top-n` to search more keys; set `--margin 0` to relax the tie-break.

## Matching Policy (Defaults)

- Normalization: trim, collapse spaces, lowercase; `--alnum-only` to keep only letters/digits.
- Scorer: `smart` (options: `smart`, `WRatio`, `ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio`, `partial_token_set_ratio`).
- smart scoring combines token-set and partial matching, adds hyphen/segment awareness, and boosts candidates whose initials match acronym-like tokens in the base (e.g., `NPPD` ↔ `Nebraska Public Power District`).
- Threshold: `--threshold 85`.
- Candidate set: `--top-n 3` from RapidFuzz extract.
- Tie handling: accept a match only if there’s no tie at the top score and (by default) the top score exceeds the second-best by `--margin 3` points; otherwise mark as ambiguous.
- Duplicate lookup keys: if the accepted normalized key maps to multiple lookup rows, mark as ambiguous.
- Multiple keys: the specified columns are joined into a composite key before normalization and matching.

## Multiple Keys

- Provide comma-separated columns for `--base-key` and `--lookup-key` when you need a stricter match such as `Name` + `ID`.
- The composite key uses `" | "` as a joiner for readability; normalization then trims, collapses spaces, and case-folds.

## XLSX Notes

- Use `--sheet <name>` to select a sheet for both base and lookup when reading Excel files.

## Testing

Run the unit tests:

```bash
pipenv run python -m unittest discover -s tests -p "test_*.py" -q
```

Run the end-to-end smoke test on the sample CSVs:

```bash
pipenv run python scripts/smoke_e2e.py
```

## Repository Layout

- `fmatch/cli.py` — argument parsing and interactive prompts
- `fmatch/io.py` — CSV/XLSX read/write helpers
- `fmatch/match.py` — normalization and fuzzy matching/selection policy
- `fmatch/enrich.py` — apply matches and append columns
- `fmatch/__main__.py` — package entrypoint (`python -m fmatch`)
- `samples/` — minimal CSV sample data
- `scripts/smoke_e2e.py` — quick end-to-end smoke script
- `tests/` — small unittest suite

## Troubleshooting

- “No module named fmatch”: run commands via `pipenv run ...` from the repo root, or ensure the project root is on `PYTHONPATH`.
- “rapidfuzz is required”: dependencies are in the Pipenv, ensure `pipenv install` succeeded.
- Excel read/write requires `openpyxl` (included in deps).

No telemetry. Local-only tool by design.
