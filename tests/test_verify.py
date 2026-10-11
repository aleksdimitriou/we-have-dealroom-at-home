"""
Tests for src/verify.py. Run from the project root:
    python -m unittest discover tests

Fixtures are small and hand-made.
"""

import unittest
from datetime import datetime

from schema import (
    Action, ActionKind, ActionOutcome, Attribute, BatchRow, ClaimStatus, ClaimType,
    ProposedClaim, RejectionReason, Run, Source, SourceOrigin,
)
from src.verify import normalise_text, normalise_url, verify_claims

T = datetime(2026, 10, 3, 12, 0)

COMPANY_URL = "https://www.example-robotics.com/"
COMPANY_PAGE = "https://www.example-robotics.com/about"
YC_PAGE = "https://www.ycombinator.com/companies/example-robotics"
PRESS_PAGE = "https://news.example.org/story"
LINKEDIN_URL = "https://www.linkedin.com/in/jane-doe"

PAGE_TEXTS = {
    "s1": "Example Robotics builds   welding robots.\n We&rsquo;re based in Austin, Texas. "
          "Investor Partners: Goose Capital",
    "s2": "Founded:2026 Team Size:6 Location:San Francisco Jane Doe, Founder/CEO",
    "s3": "Jane Doe Experience Example Robotics Co-Founder &amp; CEO December 2025 - Present "
          "Acme Corp Senior Engineer 2019 - 2025 Education University of Texas at Austin",
}


def make_run() -> Run:
    """A run that fetched the company page and the YC page, failed on two pages, saw one URL only in
    search results."""
    actions = [
        Action(action_id="a1", kind=ActionKind.FETCH, query_or_url=COMPANY_PAGE, timestamp=T),
        Action(action_id="a2", kind=ActionKind.FETCH, query_or_url=YC_PAGE, timestamp=T),
        Action(action_id="a3", kind=ActionKind.SEARCH, query_or_url='"Jane Doe" "Example Robotics"', timestamp=T),
        Action(action_id="a4", kind=ActionKind.FETCH, query_or_url=PRESS_PAGE, timestamp=T,
               outcome=ActionOutcome.FETCH_FAILED),
        Action(action_id="a5", kind=ActionKind.FETCH, query_or_url="https://blocked.example.net/p", timestamp=T,
               outcome=ActionOutcome.ROBOTS_BLOCKED),
        Action(action_id="a6", kind=ActionKind.FETCH, query_or_url=LINKEDIN_URL, timestamp=T),
    ]
    sources = [
        Source(source_id="s1", url=COMPANY_PAGE, fetched_at=T, fetched_by_action="a1", content_hash="h1"),
        Source(source_id="s2", url=YC_PAGE, fetched_at=T, fetched_by_action="a2", content_hash="h2"),
        Source(source_id="s3", url=LINKEDIN_URL, fetched_at=T, fetched_by_action="a6", content_hash="h3"),
    ]
    return Run(
        run_id="test",
        input=BatchRow(company_name="Example Robotics", company_url=COMPANY_URL, founder_names=["Jane Doe"]),
        model="none",
        started_at=T,
        actions=actions,
        sources=sources,
        search_result_urls=["https://snippets.example.com/jane-doe"],
    )


def claim(**overrides) -> ProposedClaim:
    """A valid company-description claim from the company page; override any field."""
    fields = dict(
        subject="company",
        claim_type=ClaimType.COMPANY_DESCRIPTION,
        value="Builds welding robots",
        source_url=COMPANY_PAGE,
        quote="Example Robotics builds welding robots.",
        source_origin=SourceOrigin.COMPANY,
    )
    fields.update(overrides)
    return ProposedClaim(**fields)


def verify_one(p: ProposedClaim, run: Run | None = None, page_texts: dict | None = None):
    claims, evidence = verify_claims([p], run or make_run(), PAGE_TEXTS if page_texts is None else page_texts)
    return claims[0], evidence


class NormalisationTests(unittest.TestCase):
    def test_entities_curly_quotes_nfkc_and_whitespace(self):
        self.assertEqual(normalise_text("We&rsquo;re  here"), "We're here")
        self.assertEqual(normalise_text("“quoted”"), '"quoted"')
        self.assertEqual(normalise_text("eﬃcient team"), "efficient team")   # ligature, nbsp
        self.assertEqual(normalise_text(" a\n\tb  "), "a b")

    def test_case_is_kept(self):
        self.assertNotEqual(normalise_text("Founder"), normalise_text("founder"))

    def test_url_normalisation(self):
        self.assertEqual(normalise_url("http://WWW.Example.com/about/#team"), normalise_url("https://example.com/about"))
        self.assertNotEqual(normalise_url("https://example.com/p?id=1"), normalise_url("https://example.com/p?id=2"))


class SourcedTests(unittest.TestCase):
    def test_quote_on_page_is_sourced_with_evidence(self):
        c, evidence = verify_one(claim())
        self.assertEqual(c.status, ClaimStatus.SOURCED)
        self.assertIsNone(c.rejection_reason)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(c.evidence_id, evidence[0].evidence_id)
        self.assertEqual(evidence[0].source_id, "s1")

    def test_quote_matches_after_normalisation(self):
        c, _ = verify_one(claim(quote="We’re based in Austin, Texas."))
        self.assertEqual(c.status, ClaimStatus.SOURCED)

    def test_url_variant_still_matches(self):
        c, _ = verify_one(claim(source_url="http://example-robotics.com/about/"))
        self.assertEqual(c.status, ClaimStatus.SOURCED)


class RejectionTests(unittest.TestCase):
    def assert_rejected(self, p: ProposedClaim, reason: RejectionReason):
        c, _ = verify_one(p)
        self.assertEqual(c.status, ClaimStatus.REJECTED)
        self.assertEqual(c.rejection_reason, reason)

    def test_no_source(self):
        self.assert_rejected(claim(quote="  "), RejectionReason.NO_SOURCE)
        self.assert_rejected(claim(source_url=""), RejectionReason.NO_SOURCE)

    def test_fetch_failed(self):
        self.assert_rejected(claim(source_url=PRESS_PAGE, source_origin=SourceOrigin.MEDIA),
                             RejectionReason.FETCH_FAILED)

    def test_robots_blocked(self):
        self.assert_rejected(claim(source_url="https://blocked.example.net/p", source_origin=SourceOrigin.MEDIA),
                             RejectionReason.ROBOTS_BLOCKED)

    def test_url_seen_in_results_without_a_stored_snippet_is_quote_not_found(self):
        self.assert_rejected(claim(source_url="https://snippets.example.com/jane-doe", source_origin=SourceOrigin.MEDIA),
                             RejectionReason.QUOTE_NOT_FOUND)

    def test_source_not_fetched(self):
        self.assert_rejected(claim(source_url="https://invented.example.com/", source_origin=SourceOrigin.MEDIA),
                             RejectionReason.SOURCE_NOT_FETCHED)

    def test_quote_not_found(self):
        self.assert_rejected(claim(quote="Example Robotics builds welding drones."), RejectionReason.QUOTE_NOT_FOUND)

    def test_quote_match_is_case_sensitive(self):
        self.assert_rejected(claim(quote="example robotics builds welding robots."), RejectionReason.QUOTE_NOT_FOUND)

    def test_invalid_sector(self):
        self.assert_rejected(claim(claim_type=ClaimType.SECTOR, attribute=Attribute.PRIMARY, value="robotics"),
                             RejectionReason.INVALID_SECTOR)

    def test_valid_sector(self):
        c, _ = verify_one(claim(claim_type=ClaimType.SECTOR, attribute=Attribute.PRIMARY, value="industrial_hardware"))
        self.assertEqual(c.status, ClaimStatus.SOURCED)

    def test_attribute_missing_for_conflict_prone_type(self):
        self.assert_rejected(claim(claim_type=ClaimType.LOCATION, value="Austin, Texas",
                                   quote="based in Austin, Texas"),
                             RejectionReason.INVALID_ATTRIBUTE)

    def test_attribute_from_another_type(self):
        self.assert_rejected(claim(claim_type=ClaimType.LOCATION, attribute=Attribute.INVESTOR,
                                   value="Austin, Texas", quote="based in Austin, Texas"),
                             RejectionReason.INVALID_ATTRIBUTE)

    def test_attribute_on_type_without_attributes(self):
        self.assert_rejected(claim(attribute=Attribute.HQ), RejectionReason.INVALID_ATTRIBUTE)

    def test_valid_attribute(self):
        c, _ = verify_one(claim(claim_type=ClaimType.LOCATION, attribute=Attribute.HQ,
                                value="Austin, Texas", quote="based in Austin, Texas"))
        self.assertEqual(c.status, ClaimStatus.SOURCED)


class SnippetOverFailedFetchTests(unittest.TestCase):
    """A page that could not be fetched (robots.txt, an error) still gives a snippet claim when the quote is in the
    stored search snippet, instead of a lead."""

    def run_with_snippet(self, url, outcome):
        from schema import SearchResult
        run = make_run()
        run.actions.append(Action(action_id="a9", kind=ActionKind.FETCH, query_or_url=url, timestamp=T, outcome=outcome))
        run.search_results = [SearchResult(action_id="a3", rank=1, url=url, title="Example Robotics funding",
                                           snippet="Example Robotics raised $750K in a seed round led by Inuka Capital.")]
        return run

    def test_robots_blocked_page_with_the_quote_in_its_snippet_gets_the_snippet_status(self):
        url = "https://tracxn.com/d/companies/example-robotics"
        run = self.run_with_snippet(url, ActionOutcome.ROBOTS_BLOCKED)
        c, _ = verify_one(claim(source_url=url, source_origin=SourceOrigin.DATABASE, claim_type=ClaimType.FUNDING,
                                attribute=Attribute.TOTAL_RAISED, value="$750K", quote="raised $750K in a seed round"), run)
        self.assertEqual((c.status, c.rejection_reason), (ClaimStatus.SNIPPET, None))

    def test_a_quote_not_in_the_snippet_is_still_a_lead_for_a_failed_fetch(self):
        url = "https://tracxn.com/d/companies/example-robotics"
        run = self.run_with_snippet(url, ActionOutcome.ROBOTS_BLOCKED)
        c, _ = verify_one(claim(source_url=url, source_origin=SourceOrigin.DATABASE, claim_type=ClaimType.FUNDING,
                                attribute=Attribute.TOTAL_RAISED, value="$5M", quote="raised $5M"), run)
        self.assertEqual(c.rejection_reason, RejectionReason.ROBOTS_BLOCKED)


class LinkedInEducationDefaultTests(unittest.TestCase):
    URL = "https://de.linkedin.com/in/jane-doe"

    def snippet_claim(self, value, quote, attribute=None, url=None):
        from schema import SearchResult
        run = make_run()
        run.search_results = [SearchResult(action_id="a3", rank=1, url=url or self.URL, title="Jane Doe - Example Robotics",
                                           snippet=f"Jane Doe, founder at Example Robotics. Education: {quote}")]
        return verify_one(claim(subject="Jane Doe", claim_type=ClaimType.EDUCATION, attribute=attribute, value=value,
                                source_url=url or self.URL, quote=quote, source_origin=SourceOrigin.FOUNDER), run)[0]

    def test_the_level_is_read_from_the_text(self):
        for quote, level in [("University of Toronto, Master’s Degree Management", Attribute.MASTERS),
                             ("Carleton University, Bachelor’s Degree Engineering", Attribute.BACHELORS),
                             ("Emerson College, BS Political Science", Attribute.BACHELORS),
                             ("ETH Zurich, Doctor of Philosophy - PhD, Robotics", Attribute.PHD),
                             ("New York University, Master of Arts - MA, Comparative Politics", Attribute.MASTERS)]:
            c = self.snippet_claim(quote, quote)
            self.assertEqual((c.status, c.attribute), (ClaimStatus.SNIPPET, level), quote)

    def test_without_a_readable_level_the_claim_gets_degree_level_not_stated_not_a_rejection(self):
        c = self.snippet_claim("Stanford University", "Stanford University")
        self.assertEqual((c.status, c.attribute, c.rejection_reason), (ClaimStatus.SNIPPET, Attribute.DEGREE_LEVEL_NOT_STATED, None))

    def test_degree_level_not_stated_is_set_valued_and_never_makes_sources_disagree(self):
        from schema import SET_VALUED_ATTRIBUTES, ATTRIBUTES_BY_TYPE
        self.assertIn(Attribute.DEGREE_LEVEL_NOT_STATED, SET_VALUED_ATTRIBUTES)
        self.assertIn(Attribute.DEGREE_LEVEL_NOT_STATED, ATTRIBUTES_BY_TYPE[ClaimType.EDUCATION])

    def test_an_attribute_the_model_gave_is_kept_and_other_sources_are_unchanged(self):
        c = self.snippet_claim("PhD, X University", "X University, PhD", attribute=Attribute.MASTERS)
        self.assertEqual(c.attribute, Attribute.MASTERS)
        other = self.snippet_claim("Stanford University", "Stanford University", url="https://news.example.org/jane")
        self.assertEqual((other.status, other.rejection_reason), (ClaimStatus.REJECTED, RejectionReason.INVALID_ATTRIBUTE))


class ApiSourceTests(unittest.TestCase):
    def test_a_claim_matched_to_an_api_response_is_corrected_to_database(self):
        from schema import FoundVia
        run = make_run()
        url = "https://api.openalex.org/works?filter=authorships.author.id:A1"
        run.sources.append(Source(source_id="s9", url=url, fetched_at=T, fetched_by_action="a6", content_hash="h",
                                  found_via=FoundVia.API))
        texts = dict(PAGE_TEXTS, s9='{"works": [{"title": "Closed-chain planning for robots"}]}')
        c, _ = verify_one(claim(subject="Jane Doe", claim_type=ClaimType.TECHNICAL, value="Closed-chain planning (2025)",
                                source_url=url, quote="Closed-chain planning for robots", source_origin=SourceOrigin.FOUNDER), run, texts)
        self.assertEqual((c.status, c.source_origin, c.origin_corrected_from), (ClaimStatus.SOURCED, SourceOrigin.DATABASE, SourceOrigin.FOUNDER))


class OriginTests(unittest.TestCase):
    def test_company_domain_counterparty_is_corrected_to_company(self):
        c, _ = verify_one(claim(source_origin=SourceOrigin.COUNTERPARTY))
        self.assertEqual(c.status, ClaimStatus.SOURCED)
        self.assertEqual(c.source_origin, SourceOrigin.COMPANY)
        self.assertEqual(c.origin_corrected_from, SourceOrigin.COUNTERPARTY)

    def test_subdomain_counts_as_company_domain(self):
        c, _ = verify_one(claim(source_url="https://blog.example-robotics.com/post", source_origin=SourceOrigin.MEDIA))
        self.assertEqual(c.source_origin, SourceOrigin.COMPANY)
        self.assertEqual(c.origin_corrected_from, SourceOrigin.MEDIA)

    def test_correction_applies_to_rejected_claims_too(self):
        c, _ = verify_one(claim(quote="not on the page", source_origin=SourceOrigin.DATABASE))
        self.assertEqual(c.status, ClaimStatus.REJECTED)
        self.assertEqual(c.source_origin, SourceOrigin.COMPANY)

    def test_founder_on_company_domain_is_kept(self):
        c, _ = verify_one(claim(source_origin=SourceOrigin.FOUNDER))
        self.assertEqual(c.source_origin, SourceOrigin.FOUNDER)
        self.assertIsNone(c.origin_corrected_from)

    def test_counterparty_off_company_domain_is_kept(self):
        c, _ = verify_one(claim(source_url=YC_PAGE, quote="Team Size:6", claim_type=ClaimType.STAGE_SIGNAL,
                                attribute=Attribute.TEAM_SIZE, value="6", source_origin=SourceOrigin.COUNTERPARTY))
        self.assertEqual(c.status, ClaimStatus.SOURCED)
        self.assertEqual(c.source_origin, SourceOrigin.COUNTERPARTY)
        self.assertIsNone(c.origin_corrected_from)

    def test_lookalike_domain_is_not_company_domain(self):
        c, _ = verify_one(claim(source_url="https://notexample-robotics.com/x", source_origin=SourceOrigin.MEDIA))
        self.assertEqual(c.source_origin, SourceOrigin.MEDIA)


class ClaimRecordTests(unittest.TestCase):
    def test_claims_keep_their_source_url_and_quote_even_when_rejected(self):
        ok, _ = verify_one(claim())
        bad, _ = verify_one(claim(source_url=PRESS_PAGE, source_origin=SourceOrigin.MEDIA))
        self.assertEqual((ok.source_url, ok.quote), (COMPANY_PAGE, "Example Robotics builds welding robots."))
        self.assertEqual(bad.status, ClaimStatus.REJECTED)
        self.assertEqual((bad.source_url, bad.quote), (PRESS_PAGE, "Example Robotics builds welding robots."))


class BatchTests(unittest.TestCase):
    def test_no_claim_disappears_and_order_is_kept(self):
        proposed = [claim(), claim(quote=""), claim(source_url=PRESS_PAGE, source_origin=SourceOrigin.MEDIA),
                    claim(quote="We’re based in Austin, Texas.")]
        claims, evidence = verify_claims(proposed, make_run(), PAGE_TEXTS)
        self.assertEqual(len(claims), len(proposed))
        self.assertEqual([c.claim_id for c in claims], ["c001", "c002", "c003", "c004"])
        self.assertEqual([c.status for c in claims],
                         [ClaimStatus.SOURCED, ClaimStatus.REJECTED, ClaimStatus.REJECTED, ClaimStatus.SOURCED])
        self.assertEqual(len(evidence), 2)
