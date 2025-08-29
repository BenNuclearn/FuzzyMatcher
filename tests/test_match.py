from __future__ import annotations

import unittest

import pandas as pd

from fmatch.match import (
    MatchPolicy,
    MatchResult,
    build_lookup_index,
    match_all,
    match_one,
    normalize_series,
    NormalizeOptions,
)

try:
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
    HAS_RAPIDFUZZ = True
except Exception:  # pragma: no cover
    HAS_RAPIDFUZZ = False


class TestNormalize(unittest.TestCase):
    def test_normalize_basic(self):
        s = pd.Series(["  Foo   Bar  ", "BAZ\tQux", None])
        norm = NormalizeOptions()
        out = normalize_series(s, norm).tolist()
        self.assertEqual(out[0], "foo bar")
        self.assertEqual(out[1], "baz qux")
        self.assertEqual(out[2], "")

    def test_alnum_only(self):
        s = pd.Series(["A-C_M.E 1"]) 
        norm = NormalizeOptions(alnum_only=True)
        out = normalize_series(s, norm).tolist()
        self.assertEqual(out[0], "acme1")


@unittest.skipIf(not HAS_RAPIDFUZZ, "rapidfuzz not installed")
class TestMatch(unittest.TestCase):
    def test_match_one_unique(self):
        # base: alice -> lookup unique 'alice' at row 1
        index = build_lookup_index(["bob", "alice", "carol"])
        policy = MatchPolicy(threshold=85, top_n=3, tie_margin=3, scorer="WRatio")
        res = match_one("alice", index, policy)
        self.assertEqual(res.reason, "matched")
        self.assertEqual(res.row_index, 1)
        self.assertIsNotNone(res.score)

    def test_match_one_duplicate_lookup_rows_ambiguous(self):
        # Two rows with same normalized key should be ambiguous even if top score is clear
        index = build_lookup_index(["alice", "alice"])  # duplicates
        policy = MatchPolicy(threshold=80, top_n=3, tie_margin=3, scorer="WRatio")
        res = match_one("alice", index, policy)
        self.assertEqual(res.reason, "ambiguous")
        self.assertIsNotNone(res.score)

    def test_unmatched_when_no_choices_or_empty_base(self):
        index = build_lookup_index([])
        policy = MatchPolicy()
        self.assertEqual(match_one("alice", index, policy).reason, "unmatched")
        # empty base key
        index2 = build_lookup_index(["alice"]) 
        self.assertEqual(match_one("", index2, policy).reason, "unmatched")

    def test_match_all_runs(self):
        base = pd.Series(["Alice", "", "X"])  # second is empty => unmatched
        lookup = pd.Series(["alice"])  # only one choice
        norm = NormalizeOptions()
        policy = MatchPolicy(threshold=80)
        results = match_all(base, lookup, norm, policy)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].reason, "matched")
        self.assertEqual(results[1].reason, "unmatched")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
