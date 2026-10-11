"""
Tests for src/agent.py, the fixed research workflow, with a scripted fake model, fake pages and fake search.
No network, no API calls.
Run from the project root:  python -m unittest discover tests
"""

import json
import tempfile
import unittest
from pathlib import Path

from schema import ActionOutcome, BatchRow, ClaimStatus, FoundVia, RejectionReason
from src.agent import (
    ASSUMED_FOUNDERS, pick_site_pages, page_kind, MAX_SEARCHES, SUBMIT_TOOL, SYSTEM_PROMPT, Researcher, claims_report, find_row, names_person,
    page_is_anchored, pass_prefilter, skip_reason, summarise, task_header, without_founder_names,
)
from src.cache import PageCache, SearchCache, cached_search
from src.fetch import FetchResult
from src.llm import ToolCall, Turn, Usage
from src.openalex import OpenAlexResult, Work
from src.search import CREDIT_PRICE_USD, SearchHit, SearchResponse

ROW = BatchRow(company_name="Example Robotics", company_url="https://example-robotics.com",
               accelerator_profile_url="https://www.ycombinator.com/companies/example-robotics",
               founder_names=["Jane Doe", "John Roe"])
RAW_FOUNDERS = ["Jane Doe", "John (Johnny) Roe"]

COMPANY = "https://example-robotics.com"
YC = "https://www.ycombinator.com/companies/example-robotics"
TEAM = "https://www.example-robotics.com/team"
JANE_SITE = "https://janedoe.dev/"
PRESS = "https://news.example.org/jane-doe-profile"
STRAY = "https://other.example.net/jane-doe"
ACME = "https://acme.example/blog/jane-doe"
FUNDING = "https://funds.example.org/example-robotics"
LINKEDIN = "https://www.linkedin.com/in/jane-doe-123"
VIDEO = "https://www.youtube.com/watch?v=abc"
PAPER = "https://arxiv.org/pdf/1234.5678"


def page(url, text, links=(), title="Page"):
    return FetchResult(requested_url=url, outcome=ActionOutcome.OK, final_url=url, status=200, title=title,
                       text=text, links=list(links))


PAGES = {
    COMPANY: page(COMPANY, "Example Robotics builds welding robots.\nBook a demo",
                  ["https://www.example-robotics.com/team", JANE_SITE, LINKEDIN, "https://example-robotics.com/careers"]),
    YC: page(YC, "Founded:2026 Team Size:2 Location:San Francisco\nJane Doe Founder/CEO\nJohn Roe Founder/CTO"),
    TEAM: page(TEAM, "Our team. Jane Doe, CEO. John Roe, CTO."),
    JANE_SITE: page(JANE_SITE, "Jane Doe. Before Example Robotics I built Orion at Acme Corp. PhD, University of Texas at Austin."),
    PRESS: page(PRESS, "Jane Doe, founder of Example Robotics, previously built Orion."),
    STRAY: page(STRAY, "Jane Doe won the county chess cup."),
    ACME: page(ACME, "Jane Doe was a senior engineer at Acme Corp before founding Example Robotics with John Roe."),
    FUNDING: page(FUNDING, "Example Robotics raised $2M from Goose Capital. CEO Jane Doe said so."),
}


def fake_fetch_for(calls):
    def fetch(url):
        calls.append(url)
        return PAGES.get(url, FetchResult(requested_url=url, outcome=ActionOutcome.FETCH_FAILED, error="HTTP 404"))
    return fetch


def hit(rank, url, title, snippet):
    return SearchHit(rank, url, title, snippet)


def fake_search_for(calls, answers=None, sites=None):
    """Answers by a word in the query; every other query gets one unrelated hit."""
    answers = answers if answers is not None else {
        '"Jane Doe" "Example Robotics"': [
            hit(1, LINKEDIN, "Jane Doe - Founder", "Founder at Example Robotics. Previously Senior Engineer at Acme Corp."),
            hit(2, PRESS, "Jane Doe profile", "Jane Doe leads Example Robotics.")],
        '"Jane Doe" "Orion"': [hit(1, STRAY, "Jane Doe", "A Jane Doe, engineer.")],
        '"John Roe" "Example Robotics"': [hit(1, "https://x.com/johnroe", "John Roe", "John Roe, CTO at Example Robotics")],
        '"Jane Doe" "Acme Corp"': [hit(1, ACME, "Jane Doe at Acme", "Jane Doe, engineer at Acme Corp.")],
        '"Example Robotics" funding investors': [hit(1, FUNDING, "Example Robotics funding", "Example Robotics raised $2M.")],
    }

    def search(query, site=None):
        calls.append(query)
        if sites is not None:
            sites.append(site)
        return SearchResponse(query=query, site=site, credits=1, hits=answers.get(query, []))
    return search


def claim(**kw):
    base = {"subject": "company", "claim_type": "company_description", "value": "Builds welding robots",
            "source_url": COMPANY, "quote": "Example Robotics builds welding robots.", "source_origin": "company"}
    base.update(kw)
    return base


JANE_ID = claim(subject="Jane Doe", claim_type="identity", value="Founder/CEO of Example Robotics", source_url=YC,
                quote="Jane Doe Founder/CEO", source_origin="counterparty")


class FakeClient:
    """Plays back scripted turns; records what the workflow sends."""
    provider = "fake"
    model = "fake-model"
    input_price_per_token = 2 / 1e6
    output_price_per_token = 10 / 1e6

    def __init__(self, turns):
        self.turns = list(turns)
        self.user_texts, self.tool_results = [], []

    def add_user_text(self, text):
        self.user_texts.append(text)

    def add_tool_results(self, results):
        self.tool_results.append(results)

    def step(self):
        if not self.turns:
            return Turn(text="(nothing more)", usage=Usage(input_tokens=10, output_tokens=5))
        return self.turns.pop(0)


def submit(claims=(), unknowns=(), entities=(), follow_links=(), cost=0.01, call_id="s", input_tokens=1000):
    call = ToolCall(id=call_id, name="submit_claims", input={
        "claims": list(claims), "unknowns": list(unknowns), "entities": list(entities),
        "follow_links": list(follow_links)})
    return Turn(tool_calls=[call], cost_usd=cost, model="fake-model",
                usage=Usage(input_tokens=input_tokens, cache_read_tokens=500, output_tokens=100))


def entity(subject, name, kind="employer"):
    return {"subject": subject, "name": name, "kind": kind}


class WorkflowTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fetched, self.searched = [], []

    def tearDown(self):
        self.tmp.cleanup()

    def research(self, turns, row=ROW, raw=RAW_FOUNDERS, name="20261010_120000", answers=None, **kw):
        client = FakeClient(turns)
        kw.setdefault("fetch", fake_fetch_for(self.fetched))
        kw.setdefault("search", fake_search_for(self.searched, answers))
        kw.setdefault("openalex", lambda name, institutions, coauthors, **kw_: OpenAlexResult())
        kw.setdefault("robots", lambda url: True)
        kw.setdefault("sitemap", None)
        run = Researcher(row, raw, client, self.root / name, **kw).research()
        return run, client


class WorkflowTests(WorkflowTestCase):
    def full_turns(self):
        first = submit(
            claims=[claim(), JANE_ID],
            unknowns=[{"subject": "Jane Doe", "claim_type": "education", "searched_for": "not on the anchors"}],
            entities=[entity("Jane Doe", "Acme Corp"), entity("Jane Doe", "University of Texas at Austin", "university"),
                      entity("Jane Doe", "Orion", "venture"), entity("Jane Doe", "A fourth organisation"),
                      entity("Jane Doe", "Example Robotics")],                      # the company itself is not searched
            follow_links=[{"url": JANE_SITE, "for_subject": "Jane Doe"},
                          {"url": "https://not-shown.example/page", "for_subject": "Jane Doe"},
                          {"url": LINKEDIN, "for_subject": "Jane Doe"}])
        second = submit(
            claims=[claim(),                                                    # an exact duplicate of the first reading
                    claim(subject="Jane Doe", claim_type="prior_venture", value="Built Orion", source_url=PRESS,
                          quote="previously built Orion", source_origin="media"),
                    claim(subject="Jane Doe", claim_type="public_presence", value="Chess cup", source_url=STRAY,
                          quote="won the county chess cup", source_origin="media"),            # a dropped page, not in its snippet
                    claim(subject="Jane Doe", claim_type="prior_role", value="Engineer, Acme Corp",
                          source_url=LINKEDIN, quote="Previously Senior Engineer at Acme Corp.",
                          source_origin="founder", value_normalized={"organization": "Acme Corp"})],  # snippet only
            unknowns=[], call_id="s2")
        return [first, second]

    def test_the_whole_workflow(self):
        run, client = self.research(self.full_turns())

        # order: anchors, the two name searches, then (after the first reading) links, entity and company searches
        kinds = [(a.kind.value, a.query_or_url, a.for_subject) for a in run.actions]
        self.assertEqual(kinds[:6], [
            ("fetch", COMPANY, None), ("fetch", YC, None), ("fetch", TEAM, None),
            ("search", '"Jane Doe" "Example Robotics"', "Jane Doe"), ("search", '"John Roe" "Example Robotics"', "John Roe"),
            ("fetch", JANE_SITE, "Jane Doe")])
        self.assertEqual(self.searched, [
            '"Jane Doe" "Example Robotics"', '"John Roe" "Example Robotics"',
            '"Jane Doe" "Acme Corp"', '"Jane Doe" "University of Texas at Austin"', '"Jane Doe" "Orion"',
            '"Example Robotics" funding investors',
            # the snippet budget: each database by domain, by name when the domain found nothing, then LinkedIn
            '"example-robotics.com"', '"Example Robotics"', '"example-robotics.com"', '"Example Robotics"', '"example-robotics.com"', '"Example Robotics"', '"example-robotics.com"', '"Example Robotics"',
            '"Jane Doe" "Example Robotics"', '"John Roe" "Example Robotics"'])
        self.assertEqual(run.budget.searches_used, 6 + 10)                               # 3 organisation queries, no 4th
        self.assertEqual(run.budget.max_searches, 9 + 10)

        # fetches: anchors, the chosen link, results by rank across queries (LinkedIn and X are never fetched; the
        # namesake page is fetched but dropped; the cap of 2 kept pages per founder is reached by ACME and PRESS)
        self.assertEqual(self.fetched, [COMPANY, YC, TEAM, JANE_SITE, ACME, STRAY, PRESS, FUNDING])
        self.assertEqual({s.url: s.found_via for s in run.sources}, {
            COMPANY: FoundVia.INPUT, YC: FoundVia.INPUT, TEAM: FoundVia.LINK, JANE_SITE: FoundVia.LINK,
            PRESS: FoundVia.SEARCH, ACME: FoundVia.SEARCH, FUNDING: FoundVia.SEARCH})
        events = " | ".join(run.events)
        self.assertIn(f"dropped (names neither the company nor a co-founder): {STRAY}", events)
        self.assertIn("link refused (not on a page shown to the model): https://not-shown.example/page", events)
        self.assertIn(f"link refused (LinkedIn/X): {LINKEDIN}", events)

        # what the model saw
        first_message = client.user_texts[0]
        self.assertIn("FIRST READING", first_message)
        self.assertIn("Example Robotics builds welding robots.", first_message)
        self.assertIn("Our team. Jane Doe, CEO.", first_message)
        self.assertIn("Links on this page (follow_links may only use these):", first_message)
        self.assertIn("Previously Senior Engineer at Acme Corp.", first_message)                  # a snippet
        self.assertIn("[not fetched: LinkedIn/X]", first_message)
        self.assertNotIn("Before Example Robotics I built Orion", first_message)                  # not fetched yet
        reply = client.tool_results[0][0]
        self.assertEqual(reply.tool_call_id, "s")
        second_message = reply.content
        self.assertIn("SECOND READING", second_message)
        self.assertIn("Before Example Robotics I built Orion", second_message)                   # the chosen link
        self.assertIn("previously built Orion", second_message)
        self.assertIn("raised $2M from Goose Capital", second_message)
        self.assertNotIn("chess cup", second_message)                                            # dropped, never shown
        self.assertIn("A Jane Doe, engineer.", second_message)                                   # a dropped page's snippet is shown

        # the merge, verification and bookkeeping
        self.assertEqual(run.stop_reason, "submitted")
        self.assertEqual(len(run.calls), 2)
        self.assertEqual(len(run.claims), 5)                                                     # 2 + 3 new, 1 duplicate dropped
        self.assertIn("dropped 1 duplicate claims from the second reading", events)
        outcome = [(c.subject, c.claim_type.value, c.status, c.rejection_reason) for c in run.claims]
        self.assertEqual(outcome, [
            ("company", "company_description", ClaimStatus.SOURCED, None),
            ("Jane Doe", "identity", ClaimStatus.SOURCED, None),
            ("Jane Doe", "prior_venture", ClaimStatus.SOURCED, None),
            ("Jane Doe", "public_presence", ClaimStatus.REJECTED, RejectionReason.QUOTE_NOT_FOUND),
            ("Jane Doe", "prior_role", ClaimStatus.SNIPPET, None)])
        self.assertEqual(run.unknowns, [])                                                       # the second list replaces
        self.assertAlmostEqual(run.cost_usd, 0.02 + 16 * CREDIT_PRICE_USD)
        self.assertEqual(len(run.search_results), 2 + 1 + 1 + 1 + 1 + 2 + 1)      # the LinkedIn searches answer too
        per_subject = {a.for_subject for a in run.actions}
        self.assertEqual(per_subject, {None, "Jane Doe", "John Roe"})
        saved = json.loads((self.root / "20261010_120000" / "submission.json").read_text())
        self.assertEqual(len(saved["claims"]), 5)
        self.assertTrue((self.root / "20261010_120000" / "reading_2.txt").exists())

    def test_a_search_page_is_kept_only_if_it_names_the_company_or_a_cofounder(self):
        self.assertTrue(page_is_anchored("Jane Doe leads Example Robotics.", "Jane Doe", ROW, ROW.founder_names))
        self.assertTrue(page_is_anchored("Jane Doe works with John Roe.", "Jane Doe", ROW, ROW.founder_names))
        self.assertFalse(page_is_anchored("Jane Doe won the chess cup.", "Jane Doe", ROW, ROW.founder_names))
        self.assertFalse(page_is_anchored("Example Robotics hires.", "Jane Doe", ROW, ROW.founder_names))
        self.assertTrue(page_is_anchored("Example Robotics, led by John Roe.", None, ROW, ROW.founder_names))
        self.assertTrue(page_is_anchored("Example Robotics (example-robotics.com) raised money.", None, ROW, ROW.founder_names))
        self.assertFalse(page_is_anchored("Example Robotics is a name used by several firms.", None, ROW, ROW.founder_names))

    def test_filters_before_fetching(self):
        self.assertEqual(skip_reason(LINKEDIN), "LinkedIn/X")
        self.assertEqual(skip_reason("https://twitter.com/jane"), "LinkedIn/X")
        self.assertEqual(skip_reason(VIDEO), "video or social")
        self.assertEqual(skip_reason(PAPER), "PDF")
        self.assertEqual(skip_reason("https://example.org/paper.pdf"), "PDF")
        self.assertIsNone(skip_reason(PRESS))
        self.assertTrue(pass_prefilter(hit(1, PRESS, "Jane Doe profile", ""), "Jane Doe", "Example Robotics"))
        self.assertFalse(pass_prefilter(hit(1, PRESS, "Founders", "A Jane met a Doe."), "Jane Doe", "Example Robotics"))
        self.assertTrue(pass_prefilter(hit(1, FUNDING, "Funding", "Example Robotics raised $2M"), None, "Example Robotics"))
        self.assertFalse(pass_prefilter(hit(1, FUNDING, "Funding", "Jane Doe raised $2M"), None, "Example Robotics"))
        self.assertTrue(names_person("Mira Q. Lindqvist joined", "Mira Lindqvist"))                   # a middle initial is fine
        self.assertFalse(names_person("Mira met a friend, Lindqvist", "Mira Lindqvist"))              # two words that are far apart
        self.assertFalse(names_person("Maya Lindqvist joined", "Mira Lindqvist"))

    def test_results_that_fail_the_prefilter_are_never_fetched(self):
        answers = {'"Jane Doe" "Example Robotics"': [
            hit(1, VIDEO, "Jane Doe talk", "Jane Doe on stage"),
            hit(2, PAPER, "Jane Doe paper", "Jane Doe, Example Robotics"),
            hit(3, "https://nobody.example/p", "Founders to watch", "A list of founders."),
            hit(4, PRESS, "Jane Doe profile", "Jane Doe leads Example Robotics.")]}
        run, _ = self.research([submit(claims=[claim()]), submit()], answers=answers)
        self.assertEqual(self.fetched, [COMPANY, YC, TEAM, PRESS])
        self.assertEqual(run.budget.pages_fetched, 4)

    def test_link_picks_are_capped_validated_and_not_refetched(self):
        links = ["https://a.example/jane", "https://b.example/jane", "https://c.example/jane", TEAM]
        pages = dict(PAGES)
        pages[COMPANY] = page(COMPANY, "Example Robotics builds welding robots.", links + [LINKEDIN])
        for u in links[:3]:
            pages[u] = page(u, "Jane Doe of Example Robotics")
        picks = [{"url": u, "for_subject": "Jane Doe"} for u in links] + [{"url": links[0], "for_subject": "Nobody Else"}]
        calls = []

        def fetch(url):
            calls.append(url)
            return pages.get(url, FetchResult(requested_url=url, outcome=ActionOutcome.FETCH_FAILED, error="HTTP 404"))

        run, _ = self.research([submit(follow_links=picks), submit()], fetch=fetch, answers={})
        self.assertEqual(calls, [COMPANY, YC, TEAM, "https://a.example/jane", "https://b.example/jane"])
        events = " | ".join(run.events)
        self.assertIn("link refused (cap of 2 for Jane Doe): https://c.example/jane", events)
        self.assertIn("link refused (not a founder of this company): https://a.example/jane", events)

    def test_the_model_may_also_pick_company_pages_up_to_a_cap(self):
        links = ["https://example-robotics.com/imprint", "https://example-robotics.com/press", "https://example-robotics.com/customers"]
        pages = dict(PAGES)
        pages[COMPANY] = page(COMPANY, "Example Robotics builds welding robots.", links)
        for u in links:
            pages[u] = page(u, "Example Robotics, Jane Doe, managing director.")
        calls = []

        def fetch(url):
            calls.append(url)
            return pages.get(url, FetchResult(requested_url=url, outcome=ActionOutcome.FETCH_FAILED, error="HTTP 404"))

        picks = [{"url": u, "for_subject": "company"} for u in links]
        run, _ = self.research([submit(follow_links=picks), submit()], fetch=fetch, answers={})
        self.assertEqual(calls, [COMPANY, YC, *links[:2]])
        self.assertIn("link refused (cap of 2 for the company): https://example-robotics.com/customers", " | ".join(run.events))
        self.assertEqual([a.for_subject for a in run.actions if a.query_or_url in links[:2]], [None, None])

    def test_a_page_naming_the_founder_and_an_organisation_sourced_on_an_anchor_page_is_kept(self):
        row, founders = ROW, ROW.founder_names
        page_text = "Jane Doe led robotics research at Acme Corp for six years."
        self.assertFalse(page_is_anchored(page_text, "Jane Doe", row, founders))
        self.assertTrue(page_is_anchored(page_text, "Jane Doe", row, founders, ("Acme Corp",)))
        self.assertFalse(page_is_anchored("Acme Corp hired someone.", "Jane Doe", row, founders, ("Acme Corp",)))   # no founder name
        # end to end: the organisation is claimed from an anchor page (the team page), so a page about Acme is kept
        acme_only = "https://acme.example/people/jane"
        pages = dict(PAGES)
        pages[acme_only] = page(acme_only, "Jane Doe led robotics research at Acme Corp for six years.")
        calls = []

        def fetch(url):
            calls.append(url)
            return pages.get(url, FetchResult(requested_url=url, outcome=ActionOutcome.FETCH_FAILED, error="HTTP 404"))

        orion = claim(subject="Jane Doe", claim_type="prior_role", value="Engineer, Acme Corp", source_url=TEAM,
                      quote="Jane Doe, CEO.", source_origin="company", value_normalized={"organization": "Acme Corp"})
        answers = {'"Jane Doe" "Acme Corp"': [hit(1, acme_only, "Jane Doe at Acme", "Jane Doe, Acme Corp.")]}
        first = submit(claims=[orion], entities=[entity("Jane Doe", "Acme Corp")])
        run, client = self.research([first, submit()], fetch=fetch, answers=answers)
        self.assertIn(acme_only, calls)
        self.assertIn("led robotics research at Acme Corp", client.tool_results[0][0].content)
        self.assertNotIn("dropped", " ".join(run.events))

    def test_search_caps_per_founder_and_per_company(self):
        many = [entity(who, f"Org {i}") for who in ("Jane Doe", "John Roe") for i in range(6)]
        run, _ = self.research([submit(entities=many), submit()], answers={})
        searches = [a for a in run.actions if a.kind.value == "search"]
        budgeted = [a for a in searches if "[sites:" in a.query_or_url]
        self.assertEqual(len(searches) - len(budgeted), 9)             # 2 x (1 name + 3 organisations) + 1 company
        self.assertEqual(len(budgeted), 10)                            # the snippet budget is counted apart: 8 + 2
        self.assertEqual(sum(a.for_subject == "Jane Doe" and "[sites:" not in a.query_or_url for a in searches), 4)
        self.assertLessEqual(run.budget.searches_used, run.budget.max_searches)

    def test_no_second_reading_when_nothing_new_was_collected(self):
        run, client = self.research([submit(claims=[claim()])], answers={})
        self.assertEqual(len(run.calls), 1)
        self.assertEqual(client.tool_results, [])
        self.assertIn("second reading skipped: nothing new was collected", run.events)
        self.assertEqual((run.stop_reason, len(run.claims)), ("submitted", 1))

    def test_cost_guard_stops_searching_then_skips_the_second_reading(self):
        run, _ = self.research([submit(claims=[claim()], entities=[entity("Jane Doe", "Acme Corp")], cost=0.45)],
                               cost_limit=0.5)
        self.assertEqual(self.searched, ['"Jane Doe" "Example Robotics"', '"John Roe" "Example Robotics"'])  # none after the call
        self.assertTrue(any(e.startswith("search skipped (cost near the guard)") for e in run.events))
        self.assertEqual(run.budget.cost_limit_usd, 0.5)
        self.assertEqual(run.stop_reason, "submitted")                 # nothing new to read, so no second call

    def test_projected_cost_blocks_the_second_reading_and_keeps_the_first_claims(self):
        turns = [submit(claims=[claim()], entities=[entity("Jane Doe", "Acme Corp")], cost=0.05, input_tokens=120_000),
                 submit()]
        run, _ = self.research(turns, cost_limit=0.30, answers={'"Jane Doe" "Acme Corp"': [
            hit(1, ACME, "Jane Doe at Acme", "Jane Doe, engineer at Acme Corp.")]})
        self.assertEqual(run.stop_reason, "cost_limit")
        self.assertEqual(len(run.calls), 1)
        self.assertEqual([c.status for c in run.claims], [ClaimStatus.SOURCED])
        self.assertTrue(any(e.startswith("second reading skipped: projected cost") for e in run.events))

    def test_a_cost_limit_below_the_first_reading_makes_no_model_call(self):
        run, client = self.research([submit(claims=[claim()])], cost_limit=0.001)
        self.assertEqual((run.stop_reason, run.calls, client.user_texts), ("cost_limit", [], []))

    def test_invalid_submission_gets_an_error_then_a_valid_one_passes(self):
        bad = Turn(tool_calls=[ToolCall(id="b1", name="submit_claims", input={"claims": "nope", "unknowns": []})],
                   cost_usd=0.01, model="fake-model", usage=Usage(input_tokens=10, output_tokens=5))
        run, client = self.research([bad, submit(claims=[claim()])], answers={})
        self.assertTrue(client.tool_results[0][0].is_error)
        self.assertIn("did not validate", client.tool_results[0][0].content)
        self.assertEqual((run.stop_reason, len(run.claims)), ("submitted", 1))

    def test_a_model_that_never_submits_stops_with_an_error(self):
        run, client = self.research([], answers={})
        self.assertEqual(run.stop_reason, "error")
        self.assertEqual(len(client.user_texts), 3)                    # the reading, then two reminders
        self.assertEqual(run.claims, [])

    def test_a_failed_second_reading_keeps_the_first_claims(self):
        turns = [submit(claims=[claim()]), Turn(error="APIError: overloaded")]
        run, _ = self.research(turns, answers={'"Jane Doe" "Example Robotics"': [
            hit(1, PRESS, "Jane Doe profile", "Jane Doe leads Example Robotics.")]})
        self.assertEqual(run.stop_reason, "error")
        self.assertEqual([c.status for c in run.claims], [ClaimStatus.SOURCED])
        self.assertTrue(any(e.startswith("second reading failed") for e in run.events))

    def test_a_failed_anchor_fetch_is_logged_and_the_run_goes_on(self):
        row = ROW.model_copy(update={"company_url": "https://down.example"})
        run, client = self.research([submit()], row=row, answers={})
        self.assertEqual([a.outcome for a in run.actions[:2]], [ActionOutcome.FETCH_FAILED, ActionOutcome.OK])
        self.assertIn("FIRST READING", client.user_texts[0])

    def test_the_prompt_and_tool(self):
        self.assertEqual(SUBMIT_TOOL.name, "submit_claims")
        props = SUBMIT_TOOL.input_schema["properties"]
        self.assertEqual(set(props), {"claims", "unknowns", "entities", "follow_links"})
        self.assertNotIn("$ref", json.dumps(SUBMIT_TOOL.input_schema))
        self.assertIn("## The two readings", SYSTEM_PROMPT)
        for removed in ("fetch_page", "web_search", "budget", "Analyst-provided", "Never LinkedIn: it cannot"):
            self.assertNotIn(removed, SYSTEM_PROMPT)
        for rule in ("Titles, dates and degrees", "left_without_degree", "tagline and a longer description",
                     "one claim per thing", "<Name>, Founder/CEO"):
            self.assertIn(rule, SYSTEM_PROMPT)

    def test_a_reply_without_a_submission_is_nudged_and_the_reading_goes_on(self):
        text_only = Turn(text="Let me think.", usage=Usage(input_tokens=10, output_tokens=5))
        run, client = self.research([text_only, submit(claims=[claim()]), submit()], answers={})
        self.assertIn("Call submit_claims with your claims and unknowns.", client.user_texts)
        self.assertEqual((run.stop_reason, len(run.claims)), ("submitted", 1))

    def test_the_reading_gives_up_after_the_nudges_run_out(self):
        run, _ = self.research([], answers={})                       # the fake model never calls the tool
        self.assertEqual((run.stop_reason, run.claims), ("error", []))
        self.assertTrue(any("model stopped without submitting claims" in e for e in run.events))

    def test_summary_and_claims_report(self):
        run, _ = self.research(self.full_turns())
        text = summarise(run)
        self.assertIn("Searches: 16 of 19 (16 billed, 0 from the cache)", text)
        self.assertIn("Pages dropped or links refused: 3", text)
        self.assertIn("Model calls: 2", text)
        report = claims_report(run, self.root / "20261010_120000")
        self.assertIn("snippet", report)
        self.assertIn("rejected: quote_not_found", report)
        self.assertIn('search  ok             Jane Doe       "Jane Doe" "Example Robotics"', report)


class OpenAlexAndFundingTests(WorkflowTestCase):
    OA_URL = "https://api.openalex.org/works?filter=authorships.author.id:A1&sort=cited_by_count:desc"

    def fake_openalex(self, calls):
        def openalex(name, institutions, coauthors, aliases=(), fields=()):
            calls.append((name, list(institutions), list(coauthors), tuple(aliases), tuple(fields)))
            if name != "Jane Doe":
                return OpenAlexResult(requests=["https://api.openalex.org/authors?search=John+Roe"], candidates=3, name_mismatch=1, no_overlap=2,
                                decisions=[("John Roe", ["affiliation"], False), ("J. Roe", [], False)])
            work = Work("Closed-chain planning for robots", 2025, "2025-01-01", 12, "Journal X", "", ["Jane Doe"])
            return OpenAlexResult(requests=["https://api.openalex.org/authors?search=Jane+Doe", "https://api.openalex.org/works?filter=x"],
                                  candidates=2, name_mismatch=1, accepted_by="affiliation+topic", author_id="A1",
                                  decisions=[("Jane Doe", ["affiliation", "topic"], True)], author_name="Jane Doe",
                                  institutions=["The University of Texas at Austin"], works=[work], source_url=self.OA_URL,
                                  source_text='{"works": [{"title": "Closed-chain planning for robots"}]}')
        return openalex

    def test_openalex_works_become_an_api_source_the_model_reads_and_claims_from(self):
        oa_calls = []
        orion = claim(subject="Jane Doe", claim_type="prior_role", value="Engineer, UC Austin", source_url=TEAM,
                      quote="Jane Doe, CEO.", source_origin="company", value_normalized={"organization": "UC Austin"})
        paper = claim(subject="Jane Doe", claim_type="technical", value="Closed-chain planning for robots (2025, Journal X)",
                      source_url=self.OA_URL, quote="Closed-chain planning for robots", source_origin="founder",
                      signal="phd_or_publications")
        sector = claim(claim_type="sector", attribute="primary", value="ai_software", source_url=COMPANY,
                       quote="Example Robotics builds welding robots.")
        run, client = self.research([submit(claims=[orion, sector], entities=[entity("Jane Doe", "Acme Corp")]),
                                     submit(claims=[paper], call_id="s2")],
                                    openalex=self.fake_openalex(oa_calls))
        self.assertEqual(oa_calls[0][:4], ("Jane Doe", ["UC Austin"], ["John Roe"], ()))   # sourced institutions, co-founders
        self.assertEqual(oa_calls[0][4], ("Computer Science",))                          # from the sourced sector tag, below
        self.assertEqual(oa_calls[1][1], [])
        api = [a for a in run.actions if a.kind.value == "api"]
        self.assertEqual([a.for_subject for a in api], ["Jane Doe", "Jane Doe", "John Roe"])    # every request is logged
        source = next(s for s in run.sources if s.url == self.OA_URL)
        self.assertEqual(source.found_via, FoundVia.API)
        self.assertIn("OpenAlex works for Jane Doe (author matched on affiliation and topic)", client.tool_results[0][0].content)
        self.assertIn("Closed-chain planning for robots", client.tool_results[0][0].content)
        c = run.claims[-1]
        self.assertEqual((c.status, c.source_origin.value, c.origin_corrected_from.value, c.signal.value),
                         (ClaimStatus.SOURCED, "database", "founder", "phd_or_publications"))
        events = " | ".join(run.events)
        self.assertIn("openalex: Jane Doe: 2 requests (cost $0.00), 2 candidate authors (1 name mismatch, 0 with fewer than two signals; affiliation, topic = accepted), accepted by affiliation+topic, 1 works", events)
        self.assertIn("openalex: John Roe: 1 requests (cost $0.00), 3 candidate authors (1 name mismatch, 2 with fewer than two signals; affiliation; no signal), accepted by none, 0 works", events)

    def test_openalex_institution_names_do_not_join_the_anchor_organisations(self):
        oa = self.fake_openalex([])
        ut_page = "https://ut.example/jane"
        pages = dict(PAGES)
        pages[ut_page] = page(ut_page, "Jane Doe led robotics research at the University of Texas at Austin.")
        calls = []

        def fetch(url):
            calls.append(url)
            return pages.get(url, FetchResult(requested_url=url, outcome=ActionOutcome.FETCH_FAILED, error="HTTP 404"))

        answers = {'"Jane Doe" "Acme Corp"': [hit(1, ut_page, "Jane Doe at UT", "Jane Doe, University of Texas at Austin.")]}
        run, _ = self.research([submit(entities=[entity("Jane Doe", "Acme Corp")]), submit()], fetch=fetch, openalex=oa, answers=answers)
        self.assertIn(ut_page, calls)                                         # fetched ...
        self.assertIn(f"dropped (names neither the company nor a co-founder): {ut_page}", " | ".join(run.events))   # ... and dropped

    def test_aliases_come_from_the_brackets_in_the_input(self):
        calls = []
        row = BatchRow(company_name="Example Robotics", company_url=COMPANY, accelerator_profile_url=YC,
                       founder_names=["Mira Lindqvist", "Jonas Weber", "Sam Okafor", "Jane Doe"])
        raw = ["Mira (Maya) Lindqvist", "Jonas (Joni) Weber", "Sam Okafor (registered as Samuel Okafor)", "Jane Doe"]
        self.research([submit(), submit()], row=row, raw=raw, openalex=self.fake_openalex(calls))
        self.assertEqual([c[3] for c in calls], [("Maya Lindqvist",), ("Joni Weber",), ("Samuel Okafor",), ()])

    def test_a_missing_openalex_key_is_noted_once_and_the_run_goes_on(self):
        run, _ = self.research([submit(claims=[claim()]), submit()],
                               openalex=lambda *a, **k: OpenAlexResult(error="OPENALEX_API_KEY not found in .env"))
        self.assertEqual(sum(e.startswith("openalex skipped") for e in run.events), 1)
        self.assertEqual(run.stop_reason, "submitted")

    def test_each_database_is_searched_by_domain_and_by_name_only_when_the_domain_finds_nothing(self):
        sites = []
        databases = ["pitchbook.com", "crunchbase.com", "dealroom.co", "tracxn.com"]
        tracxn = "https://tracxn.com/d/companies/example-robotics"
        answers = {'"example-robotics.com"': [hit(1, tracxn, "Example Robotics", "Example Robotics, example-robotics.com")]}
        run, _ = self.research([submit(claims=[claim()]), submit()], answers=answers, search=fake_search_for(self.searched, answers, sites))
        database = [a for a in run.actions if "[sites:" in a.query_or_url and "linkedin.com" not in a.query_or_url]
        # the fake answers the domain query for every database, so no name query is needed
        self.assertEqual([a.query_or_url for a in database],
                         [f'"example-robotics.com" [sites: {d}]' for d in databases])
        self.assertEqual([s for s in sites if s and s != ["linkedin.com"]], [[d] for d in databases])
        self.assertTrue(any(f"type=database {d} (domain)" in e for e in run.events for d in databases))
        self.assertFalse(any("(name)" in e for e in run.events))

    def test_the_name_is_the_fallback_per_database(self):
        run, _ = self.research([submit(claims=[claim()]), submit()], answers={})            # nothing is found anywhere
        queries = [a.query_or_url for a in run.actions if "[sites:" in a.query_or_url]
        self.assertEqual(queries[:4], ['"example-robotics.com" [sites: pitchbook.com]', '"Example Robotics" [sites: pitchbook.com]',
                                       '"example-robotics.com" [sites: crunchbase.com]', '"Example Robotics" [sites: crunchbase.com]'])
        self.assertTrue(any("type=database pitchbook.com (name)" in e for e in run.events))

    def test_linkedin_gets_one_search_per_founder_none_for_the_company_and_is_never_fetched(self):
        sites = []
        run, _ = self.research([submit(claims=[claim()]), submit()], search=fake_search_for(self.searched, None, sites))
        linkedin = [a.query_or_url for a in run.actions if "[sites: linkedin.com]" in a.query_or_url]
        self.assertEqual(linkedin, ['"Jane Doe" "Example Robotics" [sites: linkedin.com]', '"John Roe" "Example Robotics" [sites: linkedin.com]'])
        self.assertEqual({a.for_subject for a in run.actions if "[sites: linkedin.com]" in a.query_or_url}, {"Jane Doe", "John Roe"})
        self.assertTrue(any("type=linkedin founder" in e for e in run.events))
        self.assertFalse(any("type=linkedin company" in e for e in run.events))
        self.assertNotIn(LINKEDIN, self.fetched)                                          # its snippet is used, the page is not requested

    def test_the_snippet_budget_has_its_own_count_and_is_not_eaten_by_the_other_caps(self):
        many = [entity(who, f"Org {i}") for who in ("Jane Doe", "John Roe") for i in range(6)]
        run, _ = self.research([submit(entities=many), submit()], answers={})
        searches = [a for a in run.actions if a.kind.value == "search"]
        self.assertEqual(len([a for a in searches if "[sites:" not in a.query_or_url]), 9)        # the cap outside the budget is reached
        self.assertEqual(len([a for a in searches if "[sites:" in a.query_or_url]), 10)          # and the budget still runs in full

    def test_a_company_result_that_names_the_domain_passes_the_prefilter_a_namesake_does_not(self):
        h = hit(1, FUNDING, "Aiden - Tracxn", "Aiden Inc. Website: getaiden.com. Founded 2024.")
        self.assertTrue(pass_prefilter(h, None, "Aiden", "getaiden.com"))
        self.assertTrue(pass_prefilter(hit(1, FUNDING, "getaiden.com profile", "A startup."), None, "Aiden", "getaiden.com"))
        self.assertFalse(pass_prefilter(hit(1, FUNDING, "Robots", "Another firm"), None, "Aiden", "getaiden.com"))

    def test_a_result_page_is_fetched_only_where_robots_allows_otherwise_the_snippet_is_used(self):
        tracxn = "https://tracxn.com/d/companies/example-robotics"
        answers = {'"example-robotics.com"': [hit(1, tracxn, "Example Robotics funding", "Example Robotics raised $750K from Inuka Capital.")]}
        checked = []

        def robots(url):
            checked.append(url)
            return "tracxn.com" not in url

        paper = claim(claim_type="funding", attribute="total_raised", value="$750K", source_url=tracxn,
                      quote="raised $750K from Inuka Capital", source_origin="database")
        run, client = self.research([submit(claims=[claim()]), submit(claims=[paper], call_id="s2")], answers=answers, robots=robots)
        self.assertNotIn(tracxn, self.fetched)                                          # not requested at all
        self.assertFalse(any(a.query_or_url == tracxn for a in run.actions))             # and no failed fetch logged
        self.assertTrue(any(e.startswith("robots.txt disallows tracxn.com") for e in run.events))
        self.assertIn("raised $750K from Inuka Capital", client.tool_results[0][0].content)   # the snippet reaches the model
        self.assertEqual((run.claims[-1].status, run.claims[-1].rejection_reason), (ClaimStatus.SNIPPET, None))
        self.assertEqual(sum("tracxn.com" in u for u in checked), 1)                     # one robots check per host

    def test_signals_are_part_of_the_tool_and_the_prompt(self):
        claim_schema = SUBMIT_TOOL.input_schema["properties"]["claims"]["items"]["properties"]
        self.assertIn("signal", claim_schema)
        for kind in ("prior_exit", "prior_venture_raised", "quantified_traction", "named_customer_or_partner",
                     "outside_investor", "phd_or_publications", "prior_role_same_field"):
            self.assertIn(kind, SYSTEM_PROMPT)
            self.assertIn(kind, json.dumps(claim_schema["signal"]))
        self.assertIn('"OpenAlex works for', SYSTEM_PROMPT)
        self.assertNotIn("already checked", SYSTEM_PROMPT)
        self.assertNotIn("same person", SYSTEM_PROMPT)


class DiscoveryAndInputTests(WorkflowTestCase):
    def test_find_row_reads_another_csv_case_insensitively(self):
        csv_text = ("company_name,company_url,accelerator_profile_url,founder_names (separate with ;),batch\n"
                    "Acme Robotics,https://acme.example,https://www.ycombinator.com/companies/acme,"
                    "Jane Doe; Mira (Maya) Lindqvist,W26\n")
        path = self.root / "batch.csv"
        path.write_text(csv_text, encoding="utf-8")
        row, raw = find_row("acme robotics", path)
        self.assertEqual(row.founder_names, ["Jane Doe", "Mira Lindqvist"])
        self.assertEqual(raw[1], "Mira (Maya) Lindqvist")
        with self.assertRaises(SystemExit):
            find_row("Nope", path)

    def test_header_for_named_and_discovered_founders(self):
        named = task_header(ROW, RAW_FOUNDERS)
        self.assertIn("Founders (use these exact names as subject):", named)
        self.assertIn("John (Johnny) Roe", named)
        row = without_founder_names(ROW)
        self.assertEqual((row.founder_names, ROW.founder_names), ([], ["Jane Doe", "John Roe"]))
        discovered = task_header(row, [])
        self.assertIn("Founders: not given in the input", discovered)
        self.assertIn("record an identity claim", discovered)
        self.assertNotIn("Jane Doe", discovered)

    FOUNDERS_QUERY = '"Example Robotics" founders'

    def discovery_answers(self):
        return {self.FOUNDERS_QUERY: [hit(1, PRESS, "Example Robotics founders", "Jane Doe and John Roe founded Example Robotics.")],
                '"Jane Doe" "Example Robotics"': [hit(1, "https://other.example.org/jane", "Jane Doe", "Jane Doe leads Example Robotics.")]}

    def test_discovery_searches_for_the_founders_before_the_first_reading_and_for_each_after_it(self):
        row = without_founder_names(ROW)
        first = submit(claims=[claim(), JANE_ID, dict(JANE_ID, subject="John Roe", quote="John Roe Founder/CTO")])
        run, client = self.research([first, submit()], row=row, raw=[], answers=self.discovery_answers())
        self.assertEqual(run.budget.max_searches, 10 + 10)                             # 4 per founder, 2 for the company, then the snippet budget
        self.assertEqual(self.searched[:3], [self.FOUNDERS_QUERY, '"Jane Doe" "Example Robotics"', '"John Roe" "Example Robotics"'])
        self.assertIn("Search for the company: " + self.FOUNDERS_QUERY, client.user_texts[0])      # reading 1 sees its snippets
        self.assertIn("Jane Doe and John Roe founded Example Robotics.", client.user_texts[0])
        self.assertIn("Search for Jane Doe", client.tool_results[0][0].content)        # the name snippets reach the second one
        self.assertTrue(any("type=company founders" in e for e in run.events))

    def test_discovered_founders_are_written_back_and_reach_the_dossier_the_page_and_the_export(self):
        from src.assemble import assemble, load_run
        from src.export import export_runs
        from src.view import company_view
        row = without_founder_names(ROW)
        first = submit(claims=[claim(), JANE_ID, dict(JANE_ID, subject="John Roe", quote="John Roe Founder/CTO")])
        run, _ = self.research([first, submit()], row=row, raw=[], answers=self.discovery_answers())
        self.assertEqual(run.input.founder_names, ["Jane Doe", "John Roe"])
        self.assertIn("founders discovered: 2 (each with a sourced identity or role claim)", run.events)
        saved = load_run(self.root / "20261010_120000")                                   # what was written to run.json
        self.assertEqual(saved.input.founder_names, ["Jane Doe", "John Roe"])
        assembly = assemble(saved)
        self.assertEqual([f.name for f in assembly.dossier.founders], ["Jane Doe", "John Roe"])
        self.assertEqual(assembly.unassigned, [])
        self.assertTrue(all(f.identity_confirmed for f in assembly.dossier.founders))        # YC page: an independent source
        self.assertEqual(len(company_view(saved)["founders"]), 2)
        out = self.root / "export"
        export_runs([saved], out)
        import csv
        with (out / "people.csv").open(encoding="utf-8") as f:
            self.assertEqual([r["Name"] for r in csv.DictReader(f)], ["Jane Doe", "John Roe"])

    def test_only_subjects_with_a_sourced_identity_or_role_claim_become_founders(self):
        row = without_founder_names(ROW)
        investor = claim(subject="Goose Capital", claim_type="funding", attribute="investor", value="Goose Capital",
                         source_url=COMPANY, quote="Example Robotics builds welding robots.", source_origin="media")
        prior_only = claim(subject="Sam Smith", claim_type="prior_role", value="Engineer at Initech", source_url=TEAM,
                           quote="Our team.", source_origin="company")
        unsourced = claim(subject="Pat Quinn", claim_type="identity", value="Founder", source_url=YC,
                          quote="Pat Quinn Founder/COO", source_origin="counterparty")           # the quote is not on the page
        role_only = claim(subject="Jane Doe", claim_type="role", attribute="title", value="Founder/CEO", source_url=YC,
                          quote="Jane Doe Founder/CEO", source_origin="counterparty")
        run, _ = self.research([submit(claims=[claim(), investor, prior_only, unsourced, role_only]), submit()], row=row, raw=[])
        self.assertEqual(run.input.founder_names, ["Jane Doe"])

    def test_at_most_four_founders_are_adopted(self):
        row = without_founder_names(ROW)
        names = [f"Founder Number{i}" for i in range(6)]
        text = " ".join(f"{n} Founder/CEO" for n in names)
        pages = dict(PAGES)
        pages[YC] = page(YC, text)
        calls = []

        def fetch(url):
            calls.append(url)
            return pages.get(url, FetchResult(requested_url=url, outcome=ActionOutcome.FETCH_FAILED, error="HTTP 404"))

        claims = [claim(subject=n, claim_type="identity", value="Founder", source_url=YC, quote=f"{n} Founder/CEO",
                        source_origin="counterparty") for n in names]
        run, _ = self.research([submit(claims=claims), submit()], row=row, raw=[], fetch=fetch)
        self.assertEqual(len(run.input.founder_names), 4)
        self.assertEqual(run.budget.max_searches, 16 + 12)                             # the cap of 16 plus the snippet budget of 8 + 4

    def test_the_search_cap_is_bounded(self):
        self.assertEqual(Researcher._search_cap(1), 5)
        self.assertEqual(Researcher._search_cap(2, discovery=True), 10)
        self.assertEqual(Researcher._search_cap(ASSUMED_FOUNDERS), 9)
        self.assertEqual(Researcher._search_cap(5), MAX_SEARCHES)
        self.assertEqual(Researcher._snippet_cap(2), 10)                                # not part of the cap above


class CacheTests(WorkflowTestCase):
    def test_a_second_run_reuses_pages_and_searches_and_is_not_billed_again(self):
        page_cache, search_cache = PageCache(self.root / "cache"), SearchCache(self.root / "cache" / "search")
        search = cached_search(search_cache, fake_search_for(self.searched))
        turns = lambda: [submit(claims=[claim()], entities=[entity("Jane Doe", "Acme Corp")]), submit(call_id="s2")]
        first, _ = self.research(turns(), name="run1", cache=page_cache, search=search)
        fetched_once, searched_once = len(self.fetched), len(self.searched)
        second, _ = self.research(turns(), name="run2", cache=page_cache, search=search)
        self.assertEqual((len(self.fetched), len(self.searched)), (fetched_once, searched_once))   # nothing live
        self.assertEqual([a.query_or_url for a in second.actions], [a.query_or_url for a in first.actions])
        self.assertTrue(any(e.startswith("cache: reused page") for e in second.events))
        self.assertTrue(any(e.startswith("search reused from cache") for e in second.events))
        self.assertAlmostEqual(first.cost_usd - second.cost_usd, first.budget.searches_used * CREDIT_PRICE_USD)
        self.assertEqual(second.budget.searches_used, first.budget.searches_used)                  # still counted as searches
        self.assertIn("0 billed", summarise(second))

    def test_failed_searches_are_not_cached(self):
        cache = SearchCache(self.root / "s")
        cache.put(SearchResponse(query="q", error="HTTP 429"))
        self.assertIsNone(cache.get("q"))
        cache.put(SearchResponse(query="Q  one", credits=1, hits=[hit(1, PRESS, "t", "s")]))
        again = cache.get("q one")
        self.assertTrue(again.cached and again.cost_usd == 0 and again.credits == 1)
        self.assertEqual(again.hits[0].url, PRESS)
        self.assertIsNone(SearchCache(self.root / "s", read=False).get("q one"))


if __name__ == "__main__":
    unittest.main()


class SitePagesTests(WorkflowTestCase):
    """Anchor pages beyond the homepage: the about link, then the sitemap's pages, then common paths."""
    LISTED = ["https://www.example-robotics.com/en-GB/about-us/", "https://www.example-robotics.com/en-GB/team",
              "https://www.example-robotics.com/en-GB/press/", "https://www.example-robotics.com/en-GB/press/2026/launch",
              "https://www.example-robotics.com/impressum", "https://www.example-robotics.com/blog/post-1",
              "https://other.example.net/team"]

    def pages(self, *paths, base="https://www.example-robotics.com"):
        out = dict(PAGES)
        for path in paths:
            url = base + path
            out[url] = page(url, f"Example Robotics page {path}")
        return out

    def research_with(self, listed, pages, **kw):
        calls = []

        def fetch(url):
            calls.append(url)
            return pages.get(url, FetchResult(requested_url=url, outcome=ActionOutcome.FETCH_FAILED, error="HTTP 404"))

        run, client = self.research([submit(), submit()], fetch=fetch, answers={}, sitemap=lambda site: listed, **kw)
        return run, client, calls

    def test_page_kinds_allow_language_prefixes(self):
        self.assertEqual(page_kind("https://x.example/en-GB/about-us/"), "about")
        self.assertEqual(page_kind("https://x.example/de/impressum"), "imprint")
        self.assertEqual(page_kind("https://x.example/team"), "team")
        self.assertEqual(page_kind("https://x.example/en/press/2026/launch"), "press")
        self.assertEqual(page_kind("https://x.example/blog/team-lunch"), None)

    def test_pick_site_pages_takes_one_per_kind_in_priority_order_on_the_own_domain(self):
        picked = pick_site_pages(self.LISTED, "https://www.example-robotics.com/", set(), set(), 4)
        self.assertEqual(picked, ["https://www.example-robotics.com/en-GB/team", "https://www.example-robotics.com/en-GB/about-us/",
                                  "https://www.example-robotics.com/en-GB/press/", "https://www.example-robotics.com/impressum"])
        self.assertEqual(len(pick_site_pages(self.LISTED, "https://www.example-robotics.com/", set(), set(), 2)), 2)
        self.assertEqual(pick_site_pages(self.LISTED, "https://www.example-robotics.com/", {"team", "about"}, set(), 4),
                         ["https://www.example-robotics.com/en-GB/press/", "https://www.example-robotics.com/impressum"])

    def test_sitemap_pages_are_fetched_shown_to_the_model_and_counted(self):
        pages = self.pages("/en-GB/about-us/", "/en-GB/press/", "/impressum")
        run, client, calls = self.research_with(self.LISTED, pages)
        for path in ("/en-GB/about-us/", "/en-GB/press/", "/impressum"):
            self.assertIn("https://www.example-robotics.com" + path, calls)
        self.assertNotIn("https://www.example-robotics.com/en-GB/team", calls)             # the homepage's team link was already read
        self.assertIn("Example Robotics page /en-GB/about-us/", client.user_texts[0])
        self.assertTrue(any(e.startswith("sitemap: 7 urls, 3 pages picked") and e.endswith("pages kept: 3") for e in run.events))   # TEAM was the homepage's about link
        self.assertLessEqual(sum(a.for_subject is None and a.kind.value == "fetch" for a in run.actions), 2 + 4)

    def test_without_a_sitemap_common_paths_are_tried_and_failures_are_fine(self):
        pages = self.pages("/about", "/impressum", base="https://example-robotics.com")
        run, _, calls = self.research_with([], pages)
        self.assertIn("https://example-robotics.com/about", calls)
        self.assertIn("https://example-robotics.com/impressum", calls)
        self.assertTrue(any("common paths tried" in e for e in run.events))
        self.assertEqual(sum(1 for s in run.sources if s.url.endswith(("/about", "/impressum"))), 2)

    def test_a_homepage_that_redirects_to_another_domain_is_flagged(self):
        pages = dict(PAGES)
        pages[COMPANY] = FetchResult(requested_url=COMPANY, outcome=ActionOutcome.OK, final_url="https://tryargus.dev/", status=200,
                                     title="Argus", text="Argus builds things.", links=[])
        run, _, _ = self.research_with([], pages)
        self.assertIn("redirect: example-robotics.com now redirects to tryargus.dev", run.events)

    def test_a_redirect_within_the_same_domain_is_not_flagged(self):
        pages = dict(PAGES)
        pages[COMPANY] = FetchResult(requested_url=COMPANY, outcome=ActionOutcome.OK, final_url="https://www.example-robotics.com/en-GB",
                                     status=200, title="x", text="Example Robotics builds welding robots.", links=[])
        run, _, _ = self.research_with([], pages)
        self.assertFalse(any(e.startswith("redirect:") for e in run.events))
