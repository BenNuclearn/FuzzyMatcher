from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from rapidfuzz import fuzz, process
except Exception as e:  # pragma: no cover - provide friendly error at runtime
    fuzz = None
    process = None


@dataclass
class NormalizeOptions:
    casefold: bool = True
    trim_space: bool = True
    collapse_space: bool = True
    alnum_only: bool = False


def _normalize_text(s: str, opts: NormalizeOptions) -> str:
    t = s
    if opts.trim_space:
        t = t.strip()
    if opts.collapse_space:
        # collapse multiple spaces/tabs into single space
        t = " ".join(t.split())
    if opts.casefold:
        t = t.casefold()
    if opts.alnum_only:
        t = "".join(ch for ch in t if ch.isalnum())
    return t


def normalize_series(series: pd.Series, opts: NormalizeOptions) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .map(lambda x: _normalize_text(x, opts))
    )


@dataclass
class MatchPolicy:
    threshold: int = 85
    top_n: int = 3
    tie_margin: int = 3  # minimum lead over second-best; set 0 to disable
    scorer: str = "WRatio"  # or "token_set_ratio"


def _get_scorer(name: str):
    if fuzz is None:
        raise RuntimeError(
            "rapidfuzz is required. Please `pip install rapidfuzz`."
        )
    # Map simple names to scorer functions
    mapping = {
        "WRatio": fuzz.WRatio,
        "ratio": fuzz.ratio,
        "token_set_ratio": fuzz.token_set_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
        "partial_ratio": fuzz.partial_ratio,
    }
    return mapping.get(name, fuzz.WRatio)


@dataclass
class MatchResult:
    row_index: Optional[int]
    score: Optional[float]
    matched_key: Optional[str]
    candidate_count: int
    reason: str  # "matched", "ambiguous", "unmatched"


def build_lookup_index(lookup_keys_norm: Sequence[str]) -> Dict[str, List[int]]:
    """Map normalized lookup key -> list of row indices where it appears."""
    index: Dict[str, List[int]] = {}
    for i, key in enumerate(lookup_keys_norm):
        index.setdefault(key, []).append(i)
    return index


def top_candidates(
    base_key_norm: str,
    lookup_key_to_rows: Dict[str, List[int]],
    policy: MatchPolicy,
) -> List[Tuple[int, str, float]]:
    """Return expanded candidate rows for a normalized base key.

    Each entry is (lookup_row_index, normalized_key, score). Only includes
    candidates with score >= threshold. Rows are ordered by their key's score.
    Limited by policy.top_n at the key level (rows expanded from those keys).
    """
    if process is None:
        raise RuntimeError(
            "rapidfuzz is required. Please `pip install rapidfuzz`."
        )
    choices = list(lookup_key_to_rows.keys())
    if not base_key_norm or not choices:
        return []
    scorer = _get_scorer(policy.scorer)
    candidates = process.extract(
        base_key_norm,
        choices,
        scorer=scorer,
        limit=max(policy.top_n, 1),
    )
    valid = [(c, float(s)) for c, s, _ in candidates if s >= policy.threshold]
    rows_expanded: List[Tuple[int, str, float]] = []
    for key, score in sorted(valid, key=lambda x: x[1], reverse=True):
        for row_idx in lookup_key_to_rows.get(key, []):
            rows_expanded.append((row_idx, key, score))
    return rows_expanded


def match_one(
    base_key_norm: str,
    lookup_key_to_rows: Dict[str, List[int]],
    policy: MatchPolicy,
) -> MatchResult:
    if process is None:
        raise RuntimeError(
            "rapidfuzz is required. Please `pip install rapidfuzz`."
        )
    choices = list(lookup_key_to_rows.keys())
    if not base_key_norm:
        return MatchResult(None, None, None, 0, "unmatched")
    if not choices:
        return MatchResult(None, None, None, 0, "unmatched")

    scorer = _get_scorer(policy.scorer)
    candidates = process.extract(
        base_key_norm,
        choices,
        scorer=scorer,
        limit=max(policy.top_n, 1),
    )

    # Filter by threshold
    valid = [(c, float(s)) for c, s, _ in candidates if s >= policy.threshold]
    if not valid:
        return MatchResult(None, None, None, 0, "unmatched")

    # Expand candidate normalized keys to total row instances
    total_matches = sum(len(lookup_key_to_rows[c]) for c, _ in valid)

    # Check tie/margin among candidate keys (not expanded rows)
    valid_sorted = sorted(valid, key=lambda x: x[1], reverse=True)
    top_key, top_score = valid_sorted[0]

    # Detect tie at top score
    ties_at_top = [c for c, s in valid_sorted if s == top_score]
    if len(ties_at_top) > 1:
        return MatchResult(None, top_score, top_key, total_matches, "ambiguous")

    # Optional margin over second best
    if policy.tie_margin and len(valid_sorted) > 1:
        second_score = valid_sorted[1][1]
        if (top_score - second_score) < policy.tie_margin:
            return MatchResult(None, top_score, top_key, total_matches, "ambiguous")

    # Must resolve to exactly one row instance
    rows = lookup_key_to_rows.get(top_key, [])
    if len(rows) != 1:
        return MatchResult(None, top_score, top_key, total_matches, "ambiguous")

    return MatchResult(rows[0], top_score, top_key, total_matches, "matched")


def match_all(
    base_keys: pd.Series,
    lookup_keys: pd.Series,
    norm_opts: NormalizeOptions,
    policy: MatchPolicy,
) -> List[MatchResult]:
    base_norm = normalize_series(base_keys, norm_opts)
    lookup_norm = normalize_series(lookup_keys, norm_opts)
    index = build_lookup_index(list(lookup_norm))

    results: List[MatchResult] = []
    for _, b in base_norm.items():
        res = match_one(b, index, policy)
        results.append(res)
    return results
