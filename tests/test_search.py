"""
Tests for web search: src/search.py with fake HTTP, the snippet check and the identity anchor in verify.py, and
how snippet claims flow through evaluate, assemble, export, summary and view. No network, no API calls.
Run from the project root:  python -m unittest discover tests
"""

import tempfile
import unittest
import urllib.error
from pathlib import Path

from schema import ClaimStatus, ClaimType, RejectionReason, SourceOrigin
from src.agent import is_unfetchable_social
from src.evaluate import GoldenFact, evaluate, recall_without_linkedin
from src.llm import read_env_key
from src.search import CREDIT_PRICE_USD, search

PRESS = "https://news.example.org/jane-doe-profile"
STRAY = "https://other.example.net/jane-doe"
LINKEDIN = "https://www.linkedin.com/in/jane-doe-123"


class SearchClientTests(unittest.TestCase):
    def test_request_and_parsing(self):
        sent = {}

        def fake_post(url, body, key):
            sent.update(url=url, body=body, key=key)
            return {"results": [{"url": "https://a.com/x", "title": "A", "content": "snippet one"},
                                {"url": "", "title": "no url"}, {"url": "https://b.com", "content": "two"}],
                    "usage": {"credits": 1}}

        r = search('"Jane Doe" "Example Robotics"', site="acme.com", http_post=fake_post, key="k")
        self.assertEqual(sent["body"]["include_domains"], ["acme.com"])
        self.assertEqual(sent["body"]["search_depth"], "basic")
        self.assertEqual([(h.rank, h.url, h.snippet) for h in r.hits], [(1, "https://a.com/x", "snippet one"), (3, "https://b.com", "two")])
        self.assertEqual((r.credits, r.error), (1, None))
        self.assertAlmostEqual(r.cost_usd, CREDIT_PRICE_USD)
        search("q", http_post=fake_post, key="k")
        self.assertNotIn("include_domains", sent["body"])

    def test_errors_are_returned_not_raised(self):
        def http_error(url, body, key):
            raise urllib.error.HTTPError(url, 429, "Too Many", {}, None)

        def boom(url, body, key):
            raise TimeoutError("timed out")

        self.assertEqual((search("q", http_post=http_error, key="k").error, search("q", http_post=http_error, key="k").credits), ("HTTP 429", 0))
        self.assertIn("timed out", search("q", http_post=boom, key="k").error)

    def test_key_name_case_falls_back_after_exact_match(self):
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / ".env"
            env.write_text("tavily_api_key=lower\nOTHER=x\n")
            self.assertEqual(read_env_key("TAVILY_API_KEY", env), "lower")
            env.write_text("tavily_api_key=lower\nTAVILY_API_KEY=exact\n")
            self.assertEqual(read_env_key("TAVILY_API_KEY", env), "exact")


class SocialUrlTests(unittest.TestCase):
    def test_social_urls(self):
        self.assertTrue(is_unfetchable_social(LINKEDIN))
        self.assertTrue(is_unfetchable_social("https://x.com/janedoe"))
        self.assertFalse(is_unfetchable_social(PRESS))


class EvaluateSearchTests(unittest.TestCase):
    def test_surfaced_misses_and_recall_without_linkedin(self):
        from tests.test_evaluate import make_run, fake_judge
        golden = [GoldenFact("Acme", "Jane Doe", "technical", "Paper", "https://arxiv.org/abs/1", "founder", False),
                  GoldenFact("Acme", "Jane Doe", "education", "PhD", "https://www.linkedin.com/in/jane", "founder", True)]
        run = make_run([])
        run.search_result_urls = ["https://arxiv.org/abs/1"]
        ev = evaluate(golden, {"Acme": run}, fake_judge)
        by_type = {m["claim_type"]: m for m in ev.misses}
        self.assertTrue(by_type["technical"]["surfaced"])
        self.assertFalse(by_type["education"]["surfaced"])
        self.assertEqual(recall_without_linkedin(ev.groups)["golden"], 1)


class SnippetTierTests(unittest.TestCase):
    """Option B: snippet-checked claims are visible, lowest tier, never confirmed, never in the CRM's primary fields."""

    def setUp(self):
        from schema import SearchResult
        from tests.test_assemble import Builder
        self.b = Builder()
        self.b.add("Jane Doe", ClaimType.ROLE, "Co-Founder & CEO", SourceOrigin.COMPANY, "https://acme.com/about",
                   attribute=__import__("schema").Attribute.TITLE)
        self.b.add("Jane Doe", ClaimType.PRIOR_ROLE, "Engineer at Acme Corp", SourceOrigin.FOUNDER, LINKEDIN,
                   status=ClaimStatus.SNIPPET)
        self.b.add("Jane Doe", ClaimType.PRIOR_ROLE, "Founded Orion", SourceOrigin.MEDIA, PRESS, status=ClaimStatus.SNIPPET)
        self.b.add("Jane Doe", ClaimType.ROLE, "Co-Founder & CEO", SourceOrigin.FOUNDER, LINKEDIN, status=ClaimStatus.SNIPPET,
                   attribute=__import__("schema").Attribute.TITLE)         # a page-sourced fact already says it
        self.run = self.b.run

    def test_assemble_keeps_snippet_facts_separate_labelled_and_deduplicated(self):
        from schema import ConfirmationStatus
        from src.assemble import assemble
        a = assemble(self.run)
        jane = a.dossier.founders[0]
        roles = jane.facts[ClaimType.PRIOR_ROLE]
        self.assertEqual([f.confirmation for f in roles], [ConfirmationStatus.SNIPPET, ConfirmationStatus.SNIPPET])
        self.assertTrue(all(f.fact_key.endswith("#snippet") for f in roles))
        self.assertEqual(len(jane.facts[ClaimType.ROLE]), 1)                       # the duplicate snippet claim was dropped
        self.assertTrue(any("dropped" in n and "snippet" in n for n in a.notes))
        self.assertFalse(jane.identity_confirmed)
        self.assertFalse(jane.background_self_reported_only)                       # snippet claims are not page background

    def test_export_keeps_snippet_facts_out_of_primary_fields_but_in_claims_csv(self):
        import csv, tempfile
        from src.export import export_runs
        with tempfile.TemporaryDirectory() as d:
            export_runs([self.run], Path(d))
            with (Path(d) / "people.csv").open(encoding="utf-8") as f1, (Path(d) / "claims.csv").open(encoding="utf-8") as f2:
                people, claims = list(csv.DictReader(f1)), list(csv.DictReader(f2))
        self.assertEqual(people[0]["Prior roles"], "")
        self.assertEqual(people[0]["Job title"], "Co-Founder & CEO")
        snippet_rows = [c for c in claims if c["Status"] == "snippet"]
        self.assertEqual(len(snippet_rows), 3)
        self.assertEqual({c["Confirmation"] for c in snippet_rows}, {"snippet", ""})   # the dropped duplicate has none
        self.assertIn("snippet", {c["Visibility"] for c in snippet_rows})

    def test_summary_input_excludes_snippet_claims(self):
        from src.assemble import assemble
        from src.summary import summary_input
        values = [r["value"] for r in summary_input(assemble(self.run).dossier)]
        self.assertEqual(values, ["Co-Founder & CEO"])

    def test_page_shows_a_snippet_pill_and_explains_it_once(self):
        from src.view import build_page, company_view
        page = build_page([company_view(self.run)])
        self.assertIn('"label": "search snippet", "kind": "snippet"', page)
        self.assertIn("Only a search result's text was read", page)
        self.assertEqual(__import__("re").findall(r"verif", page, __import__("re").I), [])


class OrganisationVariantAnchorTests(unittest.TestCase):
    def test_a_snippet_naming_a_spelling_variant_of_a_sourced_organisation_is_anchored(self):
        from schema import ProposedClaim, SearchResult
        from src.verify import verify_claims
        from tests.test_verify import make_run
        run = make_run()
        run.search_results = [SearchResult(action_id="a3", rank=1, url="https://example.net/jane", title="Jane Doe",
                                           snippet="Jane Doe studied at the University of California, Berkeley. PhD in 2020.")]
        sourced = ProposedClaim(subject="Jane Doe", claim_type=ClaimType.EDUCATION, attribute="phd", value="PhD, UC Berkeley",
                                value_normalized={"organization": "UC Berkeley"}, source_url="https://www.ycombinator.com/companies/example-robotics",
                                quote="Jane Doe, Founder/CEO", source_origin=SourceOrigin.COUNTERPARTY)
        snippet = ProposedClaim(subject="Jane Doe", claim_type=ClaimType.EDUCATION, attribute="phd", value="PhD, Berkeley",
                                source_url="https://example.net/jane", quote="PhD in 2020.", source_origin=SourceOrigin.MEDIA)
        claims, _ = verify_claims([sourced, snippet], run, {"s2": "Founded:2026 Team Size:6 Location:San Francisco Jane Doe, Founder/CEO"})
        self.assertEqual([c.status for c in claims], [ClaimStatus.SOURCED, ClaimStatus.SNIPPET])


class SnippetVerifyTests(unittest.TestCase):
    def run_with(self, snippet, claim_extra=None):
        from schema import Action, ActionKind, SearchResult
        from src.verify import verify_claims
        from tests.test_verify import make_run
        from schema import ProposedClaim
        run = make_run()
        run.search_results = [SearchResult(action_id="a3", rank=1, url="https://example.net/jane", title="Jane Doe", snippet=snippet)]
        base = dict(subject="Jane Doe", claim_type=ClaimType.PRIOR_ROLE, value="Engineer at Acme Corp",
                    source_url="https://example.net/jane", quote="Engineer at Acme Corp", source_origin=SourceOrigin.MEDIA)
        base.update(claim_extra or {})
        claims, _ = verify_claims([ProposedClaim(**base)], run, {})
        return claims[0]

    def test_quote_in_a_snippet_naming_the_company_gets_snippet_status(self):
        c = self.run_with("Jane Doe, Engineer at Acme Corp, now at Example Robotics.")
        self.assertEqual((c.status, c.rejection_reason), (ClaimStatus.SNIPPET, None))
        self.assertIsNone(c.evidence_id)

    def test_quote_not_in_the_snippet_is_rejected(self):
        c = self.run_with("Jane Doe at Example Robotics.")
        self.assertEqual((c.status, c.rejection_reason), (ClaimStatus.REJECTED, RejectionReason.QUOTE_NOT_FOUND))

    def test_a_snippet_naming_no_anchor_is_unanchored(self):
        c = self.run_with("Jane Doe, Engineer at Acme Corp, chess champion.")
        self.assertEqual((c.status, c.rejection_reason), (ClaimStatus.REJECTED, RejectionReason.IDENTITY_UNANCHORED))

    def test_snippet_claims_still_need_a_valid_attribute(self):
        c = self.run_with("Jane Doe, Engineer at Acme Corp, Example Robotics.", {"attribute": "hq"})
        self.assertEqual(c.rejection_reason, RejectionReason.INVALID_ATTRIBUTE)


class EvaluateSnippetTests(unittest.TestCase):
    def runs(self):
        from tests.test_assemble import Builder, SITE
        b = (Builder(founders=("Jane Doe",))
             .add("Jane Doe", ClaimType.ROLE, "CEO", SourceOrigin.COMPANY, SITE)                       # page-sourced
             .add("Jane Doe", ClaimType.PRIOR_ROLE, "Engineer at Acme", SourceOrigin.FOUNDER, LINKEDIN, status=ClaimStatus.SNIPPET)
             .add("Jane Doe", ClaimType.EDUCATION, "PhD", SourceOrigin.FOUNDER, LINKEDIN, status=ClaimStatus.REJECTED,
                  reason=RejectionReason.SNIPPET_ONLY))                                                  # an older run's snippet lead
        b.run.input.company_name = "Acme"
        return {"Acme": b.run}

    GOLDEN = [GoldenFact("Acme", "Jane Doe", "role", "CEO", "https://acme.com", "company", False),
              GoldenFact("Acme", "Jane Doe", "prior_role", "Engineer at Acme", "https://www.linkedin.com/in/jane", "founder", True),
              GoldenFact("Acme", "Jane Doe", "education", "PhD", "https://www.linkedin.com/in/jane", "founder", True)]

    def judge(self, golden, claims):
        from src.evaluate import Judgment
        return [Judgment("same", next(i for i, g in enumerate(golden) if g.claim_type == c.claim_type), "ok") for c in claims]

    def test_recall_is_counted_with_and_without_snippet_claims(self):
        from src.evaluate import format_report, tally, total
        ev = evaluate(self.GOLDEN, self.runs(), self.judge)
        t = total(tally(ev.groups))
        self.assertEqual((t["golden"], t["strict"], t["strict_page"], t["claims_snippet"]), (3, 3, 1, 2))
        report = format_report(ev)
        self.assertIn("page-sourced claims only 1 of 3 (33%); page-sourced plus snippet-checked 3 (100%)", report)
        self.assertIn("SNIPPET CLAIMS: 2 (same 2", report)
        self.assertIn("exact, page-sourced only 1 of 1 (100%); page-sourced plus snippet-checked 1", report)   # the one non-LinkedIn fact

    def test_a_claim_only_matches_golden_facts_of_its_own_type(self):
        from src.evaluate import Judgment

        def cross(golden, claims):             # the judge wrongly matches every claim to the first golden fact
            return [Judgment("same", 0, "wrong type") for _ in claims]

        ev = evaluate(self.GOLDEN, self.runs(), cross)
        verdicts = {c.claim_type: j.verdict for g in ev.groups for c, j in zip(g.claims, g.judgments)}
        self.assertEqual(verdicts["role"], "same")                                                      # same type: kept
        self.assertEqual(verdicts["prior_role"], "not_in_key")                                          # cross-type: ignored
        self.assertEqual(verdicts["education"], "not_in_key")
