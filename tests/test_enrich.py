from __future__ import annotations

import unittest

import pandas as pd

from fmatch.enrich import EnrichOptions, enrich
from fmatch.match import MatchResult


class TestEnrich(unittest.TestCase):
    def test_enrich_populates_only_matched(self):
        base = pd.DataFrame({"Name": ["alice", "bob", "carol"]})
        lookup = pd.DataFrame(
            {"FullName": ["alice", "carol"], "email": ["a@example.com", "c@example.com"]}
        )
        # results: first matched row 0->0, second unmatched, third matched 2->1
        results = [
            MatchResult(row_index=0, score=99.0, matched_key="alice", candidate_count=1, reason="matched"),
            MatchResult(row_index=None, score=None, matched_key=None, candidate_count=0, reason="unmatched"),
            MatchResult(row_index=1, score=98.0, matched_key="carol", candidate_count=1, reason="matched"),
        ]
        out = enrich(
            base_df=base,
            lookup_df=lookup,
            match_results=results,
            lookup_take_cols=["email"],
            opts=EnrichOptions(prefix="lk_", diagnostics=True),
        )
        self.assertIn("lk_email", out.columns)
        self.assertEqual(out.loc[0, "lk_email"], "a@example.com")
        self.assertTrue(pd.isna(out.loc[1, "lk_email"]))
        self.assertEqual(out.loc[2, "lk_email"], "c@example.com")
        # diagnostics present and aligned
        self.assertIn("match_score", out.columns)
        self.assertEqual(len(out), 3)

    def test_enrich_validates_length(self):
        base = pd.DataFrame({"Name": ["x"]})
        lookup = pd.DataFrame({"FullName": ["x"], "email": ["e"]})
        results = []  # wrong length
        with self.assertRaises(ValueError):
            enrich(base, lookup, results, ["email"], EnrichOptions())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

