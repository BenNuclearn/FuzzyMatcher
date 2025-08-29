# AGENTS

Guidelines for collaborating with a coding agent (Codex CLI) on the CSV/XLSX Fuzzy Match Enricher.

## Purpose
- Keep contributions focused, simple, and traceable.
- Ensure the tool remains easy to use locally without production overhead.

## Repo Conventions
- Language: Python 3.10+.
- Style: match existing code; prefer clear function names; avoid premature abstraction.
- Dependencies: `pandas`, `openpyxl` (XLSX), `rapidfuzz` (fuzzy matching), `typer` or `argparse` for CLI.
- Entrypoint: package-style `__main__.py` and a thin `cli.py`.
- File layout (target):
  - `fmatch/cli.py` — parse args, interactive prompts.
  - `fmatch/io.py` — read/write CSV/XLSX.
  - `fmatch/match.py` — normalization + fuzzy match selection.
  - `fmatch/enrich.py` — apply matches and append columns.
  - `fmatch/__main__.py` — call CLI main.

## Working Agreement
- Keep changes minimal and scoped to the requested task.
- Prefer small, composable functions with docstrings over comments in-line.
- Surface assumptions and edge cases in PR/commit messages or in `plan.md` updates.
- Do not introduce network calls or telemetry.

## Tooling & Commands
- Local run examples:
  - `python -m fmatch --help`
  - `python -m fmatch base.csv lookup.xlsx --base-key Name --lookup-key FullName --take email,company --threshold 88`
- Development helpers (optional): `uv`/`pipx`/`pip` for env; no lockfile requirements for personal scope.

## Matching Defaults
- Normalization: trim, collapse spaces, lowercase; optional alnum-only.
- Scorer: `rapidfuzz.fuzz.WRatio` or `token_set_ratio` for order-insensitivity.
- Threshold: 85; Top-N: 3; tie-break margin: 3–5.
- Diagnostics columns off by default; enable with `--diagnostics`.

## Review Checklist
- Inputs validated: base/lookup paths exist; columns found.
- Memory-conscious on larger files (avoid unnecessary copies).
- Clear errors for missing columns, empty files, or unreadable formats.
- Output naming sensible: `base.enriched.csv|xlsx` unless `--output` provided.
- Logs/prints concise: counts for matched/ambiguous/unmatched.

## Prompts to Drive Next Steps
- “Create project skeleton with `fmatch/` modules and a CLI.”
- “Implement loader for CSV/XLSX and list columns interactively.”
- “Add fuzzy matching with RapidFuzz and a deterministic selection policy.”
- “Wire enrichment and write output with diagnostics columns.”
- “Provide a minimal sample dataset and smoke tests.”

## Non-Goals
- No database, web service, or GUI in initial scope.
- No heavy config systems or plugin architecture.

