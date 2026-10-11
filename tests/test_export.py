"""
Tests for src/export.py: the three tables, the links between them, and what may appear in primary fields.
Runs are built by hand with the Builder from test_assemble. Run from the project root:
    python -m unittest discover tests
"""

import csv
import tempfile
import unittest
from pathlib import Path

from schema import Attribute as A, ClaimStatus, ClaimType as CT, RejectionReason, SourceOrigin as O
from schema import Action, ActionKind, SearchResult
from src.export import CLAIM_COLUMNS, COMPANY_COLUMNS, LINKEDIN_COLUMN, PEOPLE_COLUMNS, export_runs, linkedin_from_search
from tests.test_assemble import Builder, PRESS, SITE, T, YC


def sample_run():
    b = (Builder()
         .add("company", CT.COMPANY_DESCRIPTION, "Builds welding robots", O.COMPANY, SITE)
         .add("company", CT.SECTOR, "ai_software", attribute=A.SECONDARY)
         .add("company", CT.SECTOR, "industrial_hardware", attribute=A.PRIMARY)
         .add("company", CT.LOCATION, "San Francisco", attribute=A.HQ)
         .add("company", CT.STAGE_SIGNAL, "2", attribute=A.TEAM_SIZE)
         .add("company", CT.STAGE_SIGNAL, "6", O.COMPANY, SITE, attribute=A.TEAM_SIZE)
         .add("company", CT.STAGE_SIGNAL, "Demo bookable", O.COMPANY, SITE, attribute=A.PRODUCT_STATUS)
         .add("company", CT.FUNDING, "Y Combinator", attribute=A.INVESTOR)
         .add("company", CT.FUNDING, "$750K", O.MEDIA, PRESS, attribute=A.TOTAL_RAISED)
         .add("Jane Doe", CT.IDENTITY, "Founder/CEO of Acme")
         .add("Jane Doe", CT.ROLE, "Co-Founder & CEO", O.COMPANY, SITE, attribute=A.TITLE)
         .add("Jane Doe", CT.PRIOR_ROLE, "Engineer, Acme Corp", O.FOUNDER, "https://www.linkedin.com/in/jane-doe")
         .add("Jane Doe", CT.EDUCATION, "PhD, UT Austin", O.FOUNDER, attribute=A.PHD)
         .add("Jane Doe", CT.TECHNICAL, "Paper X", O.MEDIA, PRESS, status=ClaimStatus.REJECTED,
              reason=RejectionReason.QUOTE_NOT_FOUND)                                   # lead
         .add("Jane Doe", CT.PUBLIC_PRESENCE, "Talk", O.MEDIA, PRESS, status=ClaimStatus.REJECTED,
              reason=RejectionReason.NO_SOURCE)                                          # log only
         .add("John Roe", CT.ROLE, "Co-Founder & CTO", O.COMPANY, SITE, attribute=A.TITLE))
    b.run.run_id = "run1"
    b.run.input.accelerator_profile_url = YC
    return b.run


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.counts = export_runs([sample_run()], self.out)
        self.companies = read(self.out / "companies.csv")
        self.people = read(self.out / "people.csv")
        self.claims = read(self.out / "claims.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def test_headers_and_counts(self):
        self.assertEqual(list(self.companies[0]), COMPANY_COLUMNS)
        self.assertEqual(list(self.people[0]), PEOPLE_COLUMNS)
        self.assertEqual(list(self.claims[0]), CLAIM_COLUMNS)
        self.assertEqual(self.counts, {"companies": 1, "people": 2, "claims": 16})

    def test_company_row(self):
        c = self.companies[0]
        self.assertEqual((c["Name"], c["Domains"]), ("Acme", "acme.com"))
        self.assertEqual(c["Description"], "Builds welding robots")
        self.assertEqual(c["Primary location"], "San Francisco")
        self.assertEqual(c["Sector"], "industrial_hardware, ai_software")                  # primary first
        self.assertEqual(c["Investors"], "Y Combinator")
        self.assertEqual(c["Funding"], "total_raised: $750K")
        self.assertIn("team_size: 2 / 6 (sources disagree)", c["Stage signals"])
        self.assertIn("product_status: Demo bookable", c["Stage signals"])
        self.assertEqual(c["Accelerator profile URL"], YC)

    def test_people_link_to_companies_by_domain(self):
        domains = {c["Domains"] for c in self.companies}
        self.assertTrue(all(p["Company"] in domains for p in self.people))
        self.assertTrue(all(c["Company"] in domains for c in self.claims))

    def test_person_row(self):
        jane = next(p for p in self.people if p["Name"] == "Jane Doe")
        self.assertEqual(jane["Job title"], "Co-Founder & CEO")
        self.assertEqual(jane[LINKEDIN_COLUMN], "")                              # no search result names the company: empty
        self.assertEqual((jane["Identity confirmed"], jane["Background self-reported only"]), ("true", "true"))
        self.assertEqual(jane["Identity level"], "confirmed by an independent source")
        self.assertEqual(jane["Education"], "PhD, UT Austin")
        john = next(p for p in self.people if p["Name"] == "John Roe")
        self.assertEqual((john[LINKEDIN_COLUMN], john["Identity confirmed"]), ("", "false"))
        self.assertEqual(john["Identity level"], "not confirmed")

    def test_leads_and_rejected_claims_never_reach_primary_fields(self):
        jane = next(p for p in self.people if p["Name"] == "Jane Doe")
        self.assertEqual(jane["Technical"], "")                         # the lead
        self.assertEqual(jane["Public presence"], "")                   # the log-only claim
        self.assertEqual(jane["Leads"], "1")
        everything = " ".join(" ".join(r.values()) for r in self.companies + self.people)
        self.assertNotIn("Paper X", everything)
        self.assertNotIn("Talk", everything)

    def test_claims_table_has_every_claim_with_visibility(self):
        by_value = {c["Value"]: c for c in self.claims}
        self.assertEqual(by_value["Paper X"]["Visibility"], "lead")
        self.assertEqual(by_value["Paper X"]["Rejection reason"], "quote not found on page")
        self.assertEqual(by_value["Paper X"]["Source URL"], PRESS)                 # leads keep their URL
        self.assertEqual(by_value["Talk"]["Visibility"], "log only")
        self.assertEqual(by_value["Talk"]["Rejection reason"], "no_source")
        self.assertEqual(by_value["Y Combinator"]["Visibility"], "main")
        self.assertEqual(by_value["Y Combinator"]["Confirmation"], "confirmed")
        self.assertEqual(by_value["PhD, UT Austin"]["Person"], "Jane Doe")
        self.assertEqual(by_value["Builds welding robots"]["Person"], "")
        self.assertTrue(all(c["Claim ID"].startswith("run1-c") for c in self.claims))
        self.assertEqual(len({c["Claim ID"] for c in self.claims}), len(self.claims))

    def test_two_runs_for_one_company_keep_the_later_one(self):
        first, second = sample_run(), sample_run()
        first.run_id, second.run_id = "old", "new"
        with tempfile.TemporaryDirectory() as d:
            counts = export_runs([first, second], Path(d))
            rows = read(Path(d) / "companies.csv")
        self.assertEqual(counts["companies"], 1)
        self.assertEqual(rows[0]["Run ID"], "new")


class FlagsColumnTests(unittest.TestCase):
    def test_a_redirect_flag_is_a_column_of_the_companies_table(self):
        from src.export import company_row
        from src.assemble import assemble
        run = sample_run()
        self.assertEqual(company_row(assemble(run), run)["Flags"], "")
        run.events.append("redirect: hydraoss.io now redirects to tryargus.dev")
        self.assertEqual(company_row(assemble(run), run)["Flags"], "hydraoss.io now redirects to tryargus.dev")
        self.assertIn("Flags", COMPANY_COLUMNS)


class LinkedInColumnTests(unittest.TestCase):
    """The column is filled only from a stored search result of the founder's own search whose snippet names the company."""

    def run_with(self, results):
        run = Builder().run
        run.actions += [Action(action_id="a1", kind=ActionKind.SEARCH, query_or_url='"Jane Doe" "Acme"', timestamp=T, for_subject="Jane Doe"),
                        Action(action_id="a2", kind=ActionKind.SEARCH, query_or_url='"John Roe" "Acme"', timestamp=T, for_subject="John Roe")]
        run.search_results = [SearchResult(action_id=a, rank=i, url=u, title=t, snippet=sn) for i, (a, u, t, sn) in enumerate(results, 1)]
        return run

    def test_a_profile_whose_snippet_names_the_company_is_used_without_scheme_or_country_host(self):
        run = self.run_with([("a1", "https://de.linkedin.com/in/jane-doe-123?trk=x", "Jane Doe - Founder", "Founder at Acme. Previously at Initech.")])
        self.assertEqual(linkedin_from_search(run, "Jane Doe"), "linkedin.com/in/jane-doe-123")

    def test_otherwise_it_is_empty(self):
        cases = {
            "snippet does not name the company": [("a1", "https://www.linkedin.com/in/jane-doe", "Jane Doe", "Engineer at Initech.")],
            "a company page, not a profile": [("a1", "https://www.linkedin.com/company/acme", "Acme", "Acme builds robots. Jane Doe")],
            "not a LinkedIn result": [("a1", "https://news.example.org/in/jane", "Jane Doe", "Founder at Acme.")],
            "found by another founder's search": [("a2", "https://www.linkedin.com/in/jane-doe", "Jane Doe", "Founder at Acme.")],
            "a different person": [("a1", "https://www.linkedin.com/in/someone", "Someone Else - Acme", "Works at Acme.")],
            "company named only as part of a word": [("a1", "https://www.linkedin.com/in/jane-doe", "Jane Doe", "Works at Acmeville.")],
        }
        for label, results in cases.items():
            self.assertEqual(linkedin_from_search(self.run_with(results), "Jane Doe"), "", label)

    def test_the_header_says_it_is_unchecked(self):
        self.assertIn("LinkedIn (from search, not checked)", PEOPLE_COLUMNS)
        self.assertNotIn("LinkedIn", PEOPLE_COLUMNS)
