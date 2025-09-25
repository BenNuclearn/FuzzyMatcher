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
    scorer: str = "smart"  # or "token_set_ratio"/"WRatio"


def _get_scorer(name: str):
    if fuzz is None:
        raise RuntimeError(
            "rapidfuzz is required. Please `pip install rapidfuzz`."
        )
    # Helper utilities for smart scoring
    def _tokens(text: str) -> List[str]:
        return [t for t in __import__("re").split(r"[^0-9a-zA-Z]+", text) if t]

    # Very small default stopword list for organization/company suffixes and generics
    STOPWORDS = {"inc", "inc.", "llc", "l.l.c", "co", "co.", "company", "corp", "corporation", "station", "plant", "the"}

    def _initials(text: str) -> str:
        toks = _tokens(text)
        out = "".join(t[0] for t in toks if t and t[0].isalpha())
        return out

    def _smart_scorer(a: str, b: str, *, processor=None, score_cutoff: float = 0.0, score_hint=None) -> float:
        # Apply optional processor consistent with RapidFuzz API
        if processor is not None:
            try:
                a_proc = processor(a)
                b_proc = processor(b)
            except Exception:
                a_proc, b_proc = a, b
        else:
            a_proc, b_proc = a, b

        # Base scores from robust metrics
        ts = fuzz.token_set_ratio(a_proc, b_proc)
        pr = fuzz.partial_ratio(a_proc, b_proc)
        # Segment-aware: if base has hyphens or pipes, compare segments
        seg_scores = []
        for seg in __import__("re").split(r"[-|]", a_proc):
            seg = seg.strip()
            if seg:
                seg_scores.append(fuzz.token_set_ratio(seg, b_proc))
        seg_best = max(seg_scores) if seg_scores else 0

        base_best = max(ts, pr, seg_best)

        # Acronym/initialism awareness
        a_inits = _initials(a_proc)
        b_inits = _initials(b_proc)
        bonus = 0

        # If a contains a short token equal to b's initials, strongly prefer
        a_tokens = _tokens(a_proc)
        a_acronyms = {t for t in a_tokens if t.isalpha() and 2 <= len(t) <= 6}
        if b_inits and b_inits.lower() in {x.lower() for x in a_acronyms}:
            base_best = max(base_best, 92.0)
        # If overall initials match, add a smaller boost
        if a_inits and b_inits and a_inits[0:3].lower() == b_inits[0:3].lower():
            bonus = 5

        # Prefer matches that contain all tokens from the left-most segment (pre-hyphen)
        import re as _re
        parts = [p.strip() for p in _re.split(r"[-|]", a_proc) if p.strip()]
        if parts:
            left_tokens = [t.lower() for t in _tokens(parts[0]) if t]
            left_tokens_nostop = [t for t in left_tokens if t not in STOPWORDS]
            b_tok_set = {t.lower() for t in _tokens(b_proc)}
            if left_tokens_nostop and set(left_tokens_nostop).issubset(b_tok_set):
                # Strong boost when the candidate fully contains the left segment
                base_best = max(base_best, 96.0)

        score = min(100.0, float(base_best + bonus))
        # Respect score_cutoff per RapidFuzz protocol
        if score < score_cutoff:
            return 0.0
        return score

    # Map simple names to scorer functions
    mapping = {
        "WRatio": fuzz.WRatio,
        "ratio": fuzz.ratio,
        "token_set_ratio": fuzz.token_set_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
        "partial_ratio": fuzz.partial_ratio,
        "partial_token_set_ratio": fuzz.partial_token_set_ratio,
        "smart": _smart_scorer,
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
    # Adaptive limit: for short or segmented keys, consider a few more candidates
    import re as _re
    toks = [t for t in _re.split(r"[^0-9a-zA-Z]+", base_key_norm) if t]
    adaptive_limit = max(policy.top_n, 1)
    if ("-" in base_key_norm or "|" in base_key_norm) or len(toks) <= 3:
        adaptive_limit = max(adaptive_limit, 10)
    candidates = process.extract(
        base_key_norm,
        choices,
        scorer=scorer,
        limit=adaptive_limit,
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
    # Adaptive limit mirroring top_candidates
    import re as _re
    toks = [t for t in _re.split(r"[^0-9a-zA-Z]+", base_key_norm) if t]
    adaptive_limit = max(policy.top_n, 1)
    if ("-" in base_key_norm or "|" in base_key_norm) or len(toks) <= 3:
        adaptive_limit = max(adaptive_limit, 10)
    candidates = process.extract(
        base_key_norm,
        choices,
        scorer=scorer,
        limit=adaptive_limit,
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


def match_one_with_secondary(
    base_primary_norm: str,
    base_secondary_norm: Optional[Tuple[str, ...]],
    lookup_key_to_rows: Dict[str, List[int]],
    lookup_secondary_norm_rows: Sequence[Optional[Tuple[str, ...]]],
    policy: MatchPolicy,
    secondary_boost: int = 10,
) -> MatchResult:
    """Match using primary key for candidate search and secondary for boosting.

    - Uses fuzzy matching on the normalized primary key to get candidate keys.
    - Expands to rows and adds a fixed boost to the row's score when all
      secondary values exactly match (normalized, non-empty).
    - Applies threshold/tie-margin on the adjusted row-level scores.
    """
    if process is None:
        raise RuntimeError(
            "rapidfuzz is required. Please `pip install rapidfuzz`."
        )
    choices = list(lookup_key_to_rows.keys())
    if not base_primary_norm:
        return MatchResult(None, None, None, 0, "unmatched")
    if not choices:
        return MatchResult(None, None, None, 0, "unmatched")

    scorer = _get_scorer(policy.scorer)
    import re as _re
    toks = [t for t in _re.split(r"[^0-9a-zA-Z]+", base_primary_norm) if t]
    adaptive_limit = max(policy.top_n, 1)
    if ("-" in base_primary_norm or "|" in base_primary_norm) or len(toks) <= 3:
        adaptive_limit = max(adaptive_limit, 10)

    candidates = process.extract(
        base_primary_norm,
        choices,
        scorer=scorer,
        limit=adaptive_limit,
    )

    # Filter by threshold at key-level first
    valid = [(c, float(s)) for c, s, _ in candidates if s >= policy.threshold]
    if not valid:
        return MatchResult(None, None, None, 0, "unmatched")

    # Expand to rows and compute adjusted row-level scores
    row_scores: List[Tuple[int, str, float]] = []  # (row_idx, norm_key, adj_score)
    for key, score in sorted(valid, key=lambda x: x[1], reverse=True):
        for row_idx in lookup_key_to_rows.get(key, []):
            adj = float(score)
            # Apply secondary boost when all secondary fields match and are not empty
            if base_secondary_norm is not None:
                try:
                    cand_sec = lookup_secondary_norm_rows[row_idx]
                except Exception:
                    cand_sec = None
                if cand_sec is not None and len(cand_sec) == len(base_secondary_norm):
                    if all((a != "" and a == b) for a, b in zip(base_secondary_norm, cand_sec)):
                        adj = min(100.0, adj + float(secondary_boost))
            row_scores.append((row_idx, key, adj))

    if not row_scores:
        return MatchResult(None, None, None, 0, "unmatched")

    # Sort by adjusted score
    row_scores.sort(key=lambda x: x[2], reverse=True)

    # Apply margin/tie rules on row-level scores
    top_row_idx, top_key, top_score = row_scores[0]
    # Ties at top
    top_peers = [r for r in row_scores if r[2] == top_score]
    if len(top_peers) > 1:
        # Multiple rows at identical top score
        return MatchResult(None, top_score, top_key, len(row_scores), "ambiguous")

    # Margin over second best
    if policy.tie_margin and len(row_scores) > 1:
        second_score = row_scores[1][2]
        if (top_score - second_score) < policy.tie_margin:
            return MatchResult(None, top_score, top_key, len(row_scores), "ambiguous")

    return MatchResult(top_row_idx, top_score, top_key, len(row_scores), "matched")
