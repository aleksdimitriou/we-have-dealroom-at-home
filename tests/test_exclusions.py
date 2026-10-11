"""
Tests for the exclusions: a person's age (never recorded), contact-data brokers, and the founder work email (admitted only
from the company's own site, shown only in people.csv). No network.
Run from the project root:  python -m unittest discover tests
"""

import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from schema import (
    LEAD_REASONS, Action, ActionKind, BatchRow, ClaimStatus, ClaimType, ProposedClaim, RejectionReason, Run,
    SearchResult, Source, SourceOrigin,
)
from src.agent import skip_reason
from src.assemble import assemble, founder_email
from src.export import export_runs
from src.pii_check import check_exports, check_run
from src.verify import is_contact_broker, verify_claims
from src.view import company_view
from tests.test_agent import WorkflowTestCase, hit, submit

T = datetime(2026, 10, 9, 12, 0)
SITE = "https://www.example-robotics.com/"
TEAM = "https://www.example-robotics.com/team"
IMPRINT = "https://example-robotics.com/impressum"
DIRECTORY = "https://directory.example.org/example-robotics"
BROKER = "https://rocketreach.co/jane-doe-email_123"
LINKEDIN = "https://www.linkedin.com/in/jane-doe"

TEXTS = {
    "s1": "Example Robotics builds welding robots. Jane Doe, Founder/CEO, aged 34. Jane Doe is 34 years old. "
          "Jane Doe was born in 1990. Our 5-year-old company ships robots. The 24-year-old founder Jane Doe.",
    "s2": "Team: Jane Doe, Founder/CEO, jane@example-robotics.com. John Roe, CTO, john.roe@gmail.com",
    "s3": "Contact Jane Doe at jane@example-robotics.com",
    "s4": "Impressum: Geschäftsführerin Jane Doe, E-Mail: jane.doe@example-robotics.com",
}


def make_run() -> Run:
    sources = [Source(source_id=s, url=u, fetched_at=T, fetched_by_action="a1", content_hash="h")
               for s, u in (("s1", SITE), ("s2", TEAM), ("s3", DIRECTORY), ("s4", IMPRINT))]
    return Run(run_id="r", model="m", started_at=T,
               input=BatchRow(company_name="Example Robotics", company_url=SITE, founder_names=["Jane Doe", "John Roe"]),
               actions=[Action(action_id="a9", kind=ActionKind.SEARCH, query_or_url="q", timestamp=T)],
               sources=sources,
               search_results=[SearchResult(action_id="a9", rank=1, url=LINKEDIN, title="Jane Doe",
                                            snippet="Jane Doe - Founder at Example Robotics - jane@example-robotics.com")])


def proposed(**kw) -> ProposedClaim:
    fields = dict(subject="Jane Doe", claim_type=ClaimType.IDENTITY, value="Jane Doe, Founder/CEO", source_url=SITE,
                  quote="Jane Doe, Founder/CEO", source_origin=SourceOrigin.COMPANY)
    fields.update(kw)
    return ProposedClaim(**fields)


def check(*claims):
    out, _ = verify_claims(list(claims), make_run(), TEXTS)
    return [(c.status, c.rejection_reason) for c in out], out


EXCLUDED = (ClaimStatus.REJECTED, RejectionReason.EXCLUDED_PERSONAL_DATA)


class AgeTests(unittest.TestCase):
    def test_a_founders_age_is_rejected_and_its_text_is_not_kept(self):
        for quote in ("Jane Doe, Founder/CEO, aged 34", "Jane Doe is 34 years old", "Jane Doe was born in 1990"):
            results, out = check(proposed(quote=quote))
            self.assertEqual(results, [EXCLUDED], quote)
            self.assertEqual((out[0].value, out[0].quote), ("[removed]", "[removed]"))
        self.assertEqual(check(proposed(value="Founder, 34 years old"))[0], [EXCLUDED])        # in the value too

    def test_a_company_age_is_not_a_persons_age(self):
        ok = proposed(subject="company", claim_type=ClaimType.STAGE_SIGNAL, attribute="founded", value="5-year-old company",
                      quote="Our 5-year-old company ships robots.")
        self.assertEqual(check(ok)[0], [(ClaimStatus.SOURCED, None)])
        tied = proposed(subject="company", claim_type=ClaimType.COMPANY_DESCRIPTION, value="Led by a young founder",
                        quote="The 24-year-old founder Jane Doe.")
        self.assertEqual(check(tied)[0], [EXCLUDED])

    def test_the_exclusions_are_log_only(self):
        self.assertNotIn(RejectionReason.EXCLUDED_PERSONAL_DATA, LEAD_REASONS)
        self.assertNotIn(RejectionReason.EXCLUDED_SOURCE, LEAD_REASONS)


class BrokerTests(unittest.TestCase):
    def test_a_broker_is_never_a_source(self):
        results, out = check(proposed(source_url=BROKER, quote="Jane Doe, Founder/CEO", source_origin=SourceOrigin.DATABASE))
        self.assertEqual(results, [(ClaimStatus.REJECTED, RejectionReason.EXCLUDED_SOURCE)])
        self.assertEqual(out[0].quote, "[removed]")

    def test_the_list_and_the_fetch_rule(self):
        for url in (BROKER, "https://www.zoominfo.com/p/x", "https://app.apollo.io/people/1", "https://lusha.com/x",
                    "https://www.contactout.com/x", "https://www.signalhire.com/x"):
            self.assertTrue(is_contact_broker(url), url)
            self.assertEqual(skip_reason(url), "contact-data broker")
        self.assertFalse(is_contact_broker("https://www.crunchbase.com/organization/x"))


class BrokerSearchTests(WorkflowTestCase):
    def test_broker_results_are_removed_before_they_are_stored_or_shown(self):
        answers = {'"Jane Doe" "Example Robotics"': [hit(1, BROKER, "Jane Doe email", "Jane Doe email and phone"),
                                                     hit(2, "https://news.example.org/jane", "Jane Doe", "Jane Doe leads Example Robotics.")]}
        run, client = self.research([submit(), submit()], answers=answers)
        self.assertNotIn(BROKER, [r.url for r in run.search_results])
        self.assertNotIn("rocketreach", client.user_texts[0])
        self.assertTrue(any("contact-data broker result(s) removed" in e for e in run.events))


class EmailTests(unittest.TestCase):
    def email(self, **kw):
        fields = dict(claim_type=ClaimType.EMAIL, value="jane@example-robotics.com", source_url=TEAM,
                      quote="Jane Doe, Founder/CEO, jane@example-robotics.com", source_origin=SourceOrigin.COMPANY)
        fields.update(kw)
        return proposed(**fields)

    def test_an_address_on_the_companys_own_page_and_domain_is_admitted(self):
        self.assertEqual(check(self.email())[0], [(ClaimStatus.SOURCED, None)])
        imprint = self.email(value="jane.doe@example-robotics.com", source_url=IMPRINT,
                             quote="E-Mail: jane.doe@example-robotics.com")
        self.assertEqual(check(imprint)[0], [(ClaimStatus.SOURCED, None)])

    def test_every_other_email_is_excluded(self):
        cases = {
            "another site": self.email(source_url=DIRECTORY, quote="Contact Jane Doe at jane@example-robotics.com"),
            "another domain": self.email(subject="John Roe", value="john.roe@gmail.com", quote="John Roe, CTO, john.roe@gmail.com"),
            "a snippet": self.email(source_url=LINKEDIN, quote="jane@example-robotics.com", source_origin=SourceOrigin.FOUNDER),
            "a guess not in the quote": self.email(value="j.doe@example-robotics.com"),
            "the company as subject": self.email(subject="company"),
        }
        for name, c in cases.items():
            results, out = check(c)
            self.assertEqual(results, [EXCLUDED], name)
            self.assertEqual(out[0].value, "[removed]", name)


def email_run() -> Run:
    run = make_run()
    run.run_id = "run1"
    claims, evidence = verify_claims([
        proposed(source_url=TEAM, quote="Jane Doe, Founder/CEO"),
        proposed(claim_type=ClaimType.EMAIL, value="jane@example-robotics.com", source_url=TEAM,
                 quote="Jane Doe, Founder/CEO, jane@example-robotics.com"),
    ], run, TEXTS)
    run.claims, run.evidence = claims, evidence
    return run


class EmailOutputTests(unittest.TestCase):
    def test_the_email_is_in_people_csv_only(self):
        run = email_run()
        self.assertEqual(founder_email(run, "Jane Doe"), "jane@example-robotics.com")
        dossier = assemble(run).dossier
        self.assertFalse(any(t == ClaimType.EMAIL for f in dossier.founders for t in f.facts))         # not in the dossier
        page = company_view(run)
        self.assertNotIn("jane@example-robotics.com", str(page))                                       # not on the page
        self.assertEqual(len(page["claims"]), 1)                                                        # the identity claim only
        with tempfile.TemporaryDirectory() as tmp:
            export_runs([run], Path(tmp))
            with (Path(tmp) / "people.csv").open(encoding="utf-8") as f:
                people = {r["Name"]: r for r in csv.DictReader(f)}
            self.assertEqual((people["Jane Doe"]["Email"], people["John Roe"]["Email"]), ("jane@example-robotics.com", ""))
            with (Path(tmp) / "claims.csv").open(encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r["Claim type"] == "email"]
            self.assertEqual((rows[0]["Value"], rows[0]["Quote"]), ("[see people.csv Email]", "[removed]"))
            self.assertEqual(check_exports(Path(tmp)), [])                                              # allowed in Email only

    def test_pii_check_allows_an_admitted_email_and_flags_everything_else(self):
        run = email_run()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "run.json").write_text(run.model_dump_json(), encoding="utf-8")
            self.assertEqual(check_run(d), [])
            run.claims[0].quote = "Jane Doe, Founder/CEO, jane@example-robotics.com, +49 30 12345678"
            (d / "run.json").write_text(run.model_dump_json(), encoding="utf-8")
            self.assertEqual(sorted(k for _, _, k in check_run(d)), ["email", "phone"])
            with (d / "companies.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Name", "Description"])
                w.writerow(["Acme", "write to info@acme.example or call +1 (415) 555-0132"])
            self.assertEqual(sorted(k for _, _, k in check_exports(d)), ["email", "phone"])


if __name__ == "__main__":
    unittest.main()
