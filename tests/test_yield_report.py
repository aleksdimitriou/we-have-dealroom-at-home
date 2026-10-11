"""
Tests for src/yield_report.py. Run from the project root:  python -m unittest discover tests
"""

import unittest
from datetime import datetime

from schema import (
    Action, ActionKind, BatchRow, Claim, ClaimStatus, ClaimType, FoundVia, Run, SearchResult, Source, SourceOrigin,
)
from src.yield_report import Yield, classify, search_types

T = datetime(2026, 10, 9, 12, 0)


def make_run(with_events=True):
    run = Run(run_id="r", model="m", started_at=T,
              input=BatchRow(company_name="Acme", company_url="https://acme.example", founder_names=["Jane Doe"]))
    queries = [('"Jane Doe" "Acme"', "Jane Doe"), ('"Jane Doe" "Initech"', "Jane Doe"), ('"Jane Doe" "Orion"', "Jane Doe"),
               ('"Acme" funding investors', None), ('"Acme" news', None), ('"Acme" funding [sites: tracxn.com]', None)]
    for i, (q, who) in enumerate(queries, start=1):
        run.actions.append(Action(action_id=f"a{i:03d}", kind=ActionKind.SEARCH, query_or_url=q, timestamp=T, for_subject=who))
    if with_events:
        run.events += ["search a001 type=name+company", "search a002 type=name+organisation position=1",
                       "search a003 type=name+organisation position=2", "search a004 type=company funding/investors",
                       "search a005 type=company news", "search a006 type=database funding"]
    run.search_results = [
        SearchResult(action_id="a001", rank=1, url="https://press.example/jane", title="t", snippet="s"),
        SearchResult(action_id="a002", rank=1, url="https://press.example/jane", title="t", snippet="s"),     # also returned by a002
        SearchResult(action_id="a002", rank=2, url="https://initech.example/jane", title="t", snippet="s"),
        SearchResult(action_id="a006", rank=1, url="https://www.tracxn.com/acme", title="t", snippet="Acme raised $1M"),
    ]
    run.sources = [Source(source_id="s1", url="https://acme.example", fetched_at=T, fetched_by_action="a0", content_hash="h", found_via=FoundVia.INPUT),
                   Source(source_id="s2", url="https://press.example/jane", fetched_at=T, fetched_by_action="a0", content_hash="h", found_via=FoundVia.SEARCH),
                   Source(source_id="s3", url="https://initech.example/jane", fetched_at=T, fetched_by_action="a0", content_hash="h", found_via=FoundVia.SEARCH)]

    def claim(i, value, url, status=ClaimStatus.SOURCED, subject="Jane Doe", ctype=ClaimType.PRIOR_ROLE):
        return Claim(claim_id=f"c{i}", subject=subject, claim_type=ctype, value=value, source_origin=SourceOrigin.MEDIA,
                     status=status, source_url=url, quote="q")

    run.claims = [claim(1, "Engineer at Initech", "https://acme.example"),                  # from an anchor: not credited
                  claim(2, "Engineer at Initech", "https://press.example/jane"),           # credited to a001, not new
                  claim(3, "Built Orion", "https://press.example/jane"),                   # credited to a001, new
                  claim(4, "Manager at Initech", "https://initech.example/jane"),         # credited to a002 (position 1), new
                  claim(5, "$1M", "https://www.tracxn.com/acme", ClaimStatus.SNIPPET, "company", ClaimType.FUNDING)]   # snippet of a006
    return run


class YieldTests(unittest.TestCase):
    def test_types_and_positions_come_from_the_logged_events(self):
        types = search_types(make_run())
        self.assertEqual(types["a002"], ("name+organisation", 1))
        self.assertEqual(types["a003"], ("name+organisation", 2))
        self.assertEqual(types["a006"], ("database funding", 0))

    def test_runs_without_events_are_classified_from_the_query_text(self):
        self.assertEqual(search_types(make_run(with_events=False)), search_types(make_run()))
        self.assertEqual(classify('"Acme" founders', False, "Acme"), "company founders")
        self.assertEqual(classify('"acme.example" funding [sites: tracxn.com]', False, "Acme"), "database funding by domain")
        self.assertEqual(classify('"Acme" funding [sites: tracxn.com]', False, "Acme"), "database funding")
        self.assertEqual(classify('"acme.example" [sites: tracxn.com]', False, "Acme"), "database tracxn.com (domain)")
        self.assertEqual(classify('"Acme" [sites: dealroom.co]', False, "Acme"), "database dealroom.co (name)")
        self.assertEqual(classify('"Acme" "acme.example" [sites: linkedin.com]', False, "Acme"), "linkedin company")
        self.assertEqual(classify('"Jane Doe" "Acme" [sites: linkedin.com]', True, "Acme"), "linkedin founder")

    def test_credits_pages_claims_and_new_facts_per_type_and_position(self):
        y = Yield()
        y.add(make_run())
        t = y.by_type
        self.assertEqual((t["name+company"]["credits"], t["name+company"]["pages"], t["name+company"]["claims"], t["name+company"]["new"]), (1, 1, 2, 1))
        self.assertEqual((t["name+organisation"]["credits"], t["name+organisation"]["claims"], t["name+organisation"]["new"]), (2, 1, 1))
        self.assertEqual((t["database funding"]["claims"], t["database funding"]["new"], t["database funding"]["pages"]), (1, 1, 0))
        self.assertEqual((y.by_position[1]["credits"], y.by_position[1]["claims"], y.by_position[2]["credits"], y.by_position[2]["claims"]), (1, 1, 1, 0))
        self.assertEqual(t["company news"]["claims"], 0)
        self.assertEqual((t["name+company"]["results"], t["name+organisation"]["results"], t["database funding"]["results"]), (1, 2, 1))
        self.assertEqual((t["database funding"]["snippet"], t["name+company"]["snippet"]), (1, 0))
        self.assertIn("1st organisation", y.format())
        self.assertIn("credited to the first one", y.format())

    def test_new_search_types_are_listed_after_the_known_ones(self):
        run = make_run()
        run.events.append("search a007 type=database tracxn.com (domain)")
        run.actions.append(Action(action_id="a007", kind=ActionKind.SEARCH, query_or_url='"acme.example" [sites: tracxn.com]', timestamp=T))
        y = Yield()
        y.add(run)
        self.assertEqual(y.by_type["database tracxn.com (domain)"]["credits"], 1)
        self.assertIn("database tracxn.com (domain)", y.format())

    def test_runs_add_up(self):
        y = Yield()
        y.add(make_run())
        y.add(make_run())
        self.assertEqual(y.by_type["name+company"]["credits"], 2)


if __name__ == "__main__":
    unittest.main()
