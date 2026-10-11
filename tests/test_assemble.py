"""
Tests for src/assemble.py and src/render.py. Runs are built by hand: claims already carry the status,
source URL and quote that verify.py gave them.
Run from the project root:  python -m unittest discover tests
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from schema import (
    Attribute, BatchRow, Claim, ClaimStatus, ClaimType, ConfirmationStatus, Evidence,
    RejectionReason, Run, Source, SourceOrigin, Unknown,
)
from src.assemble import (IDENTITY_INDEPENDENT, IDENTITY_SITE_AND_SNIPPET, IDENTITY_UNCONFIRMED, assemble, confirmation,
                          identity_confirmed, identity_level, load_run, values_disagree)
from src.render import render_text

T = datetime(2026, 10, 5, 12, 0)
YC = "https://www.ycombinator.com/companies/acme"
SITE = "https://acme.com/about"
PRESS = "https://news.example.org/acme"
CT, A, O = ClaimType, Attribute, SourceOrigin


class Builder:
    """Builds a Run, one add() per claim."""

    def __init__(self, founders=("Jane Doe", "John Roe")):
        self.run = Run(run_id="r", model="m", started_at=T, finished_at=T, stop_reason="submitted",
                       input=BatchRow(company_name="Acme", company_url="https://acme.com", founder_names=list(founders)))

    def add(self, subject, claim_type, value, origin=O.COUNTERPARTY, url=YC, attribute=None,
            status=ClaimStatus.SOURCED, reason=None, date=None, normalized=None):
        n = len(self.run.claims) + 1
        claim = Claim(claim_id=f"c{n:03d}", subject=subject, claim_type=claim_type, attribute=attribute, value=value,
                      source_origin=origin, status=status, rejection_reason=reason, source_date=date,
                      value_normalized=normalized, source_url=url, quote=f"quote {n}")
        if status == ClaimStatus.SOURCED:
            sid = f"s{n:03d}"
            self.run.sources.append(Source(source_id=sid, url=url, fetched_at=T, fetched_by_action="a1", content_hash="h"))
            self.run.evidence.append(Evidence(evidence_id=f"e{n:03d}", source_id=sid, quote="q"))
            claim.evidence_id = f"e{n:03d}"
        self.run.claims.append(claim)
        return self

    def build(self):
        return assemble(self.run)


def facts_of(assembly, claim_type, founder=None):
    holder = assembly.dossier if founder is None else next(f for f in assembly.dossier.founders if f.name == founder)
    return holder.facts.get(claim_type, [])


class ConfirmationTests(unittest.TestCase):
    def claim(self, origin, value="x", date=None, attribute=A.TEAM_SIZE):
        return Claim(claim_id="c", subject="company", claim_type=CT.STAGE_SIGNAL, attribute=attribute, value=value,
                     source_origin=origin, source_date=date)

    def test_confirmed_needs_one_counterparty(self):
        self.assertEqual(confirmation([self.claim(O.FOUNDER), self.claim(O.COUNTERPARTY)], True), ConfirmationStatus.CONFIRMED)

    def test_corroborated_needs_two_origins_other_than_media_none_counterparty(self):
        self.assertEqual(confirmation([self.claim(O.FOUNDER), self.claim(O.COMPANY)], True), ConfirmationStatus.CORROBORATED)
        self.assertEqual(confirmation([self.claim(O.COMPANY), self.claim(O.DATABASE)], True), ConfirmationStatus.CORROBORATED)

    def test_media_never_corroborates_but_still_counts_with_a_counterparty(self):
        for other in (O.FOUNDER, O.COMPANY, O.DATABASE, O.MEDIA):
            self.assertEqual(confirmation([self.claim(other), self.claim(O.MEDIA)], True),
                             ConfirmationStatus.SINGLE_SOURCE, other)
        self.assertEqual(confirmation([self.claim(O.MEDIA), self.claim(O.MEDIA)], True), ConfirmationStatus.SINGLE_SOURCE)
        self.assertEqual(confirmation([self.claim(O.MEDIA), self.claim(O.COUNTERPARTY)], True), ConfirmationStatus.CONFIRMED)

    def test_same_origin_twice_is_single_source(self):
        self.assertEqual(confirmation([self.claim(O.FOUNDER), self.claim(O.FOUNDER)], True), ConfirmationStatus.SINGLE_SOURCE)

    def test_disagreement_overrides_confirmed(self):
        claims = [self.claim(O.COUNTERPARTY, "2"), self.claim(O.COMPANY, "6")]
        self.assertEqual(confirmation(claims, True), ConfirmationStatus.SOURCES_DISAGREE)

    def test_containment_is_not_a_disagreement(self):
        claims = [self.claim(O.COUNTERPARTY, "San Francisco"), self.claim(O.MEDIA, "San Francisco, CA")]
        self.assertFalse(values_disagree(claims))

    def test_different_dates_on_every_claim_read_as_an_update(self):
        dated = [self.claim(O.COMPANY, "5", date="Jan 2026"), self.claim(O.COMPANY, "6", date="Aug 2026")]
        self.assertFalse(values_disagree(dated))
        undated = [self.claim(O.COMPANY, "5", date="Jan 2026"), self.claim(O.COMPANY, "6")]
        self.assertTrue(values_disagree(undated))

    def test_value_normalized_is_compared_when_present(self):
        a = self.claim(O.MEDIA, "$750K")
        a.value_normalized = {"amount": "750000"}
        b = self.claim(O.DATABASE, "$750,000")
        b.value_normalized = {"amount": "750000"}
        self.assertFalse(values_disagree([a, b]))


class GroupingTests(unittest.TestCase):
    def test_scalar_attribute_groups_and_set_valued_items_stay_apart(self):
        b = (Builder()
             .add("company", CT.LOCATION, "San Francisco", attribute=A.HQ)
             .add("company", CT.LOCATION, "San Francisco, CA", O.MEDIA, PRESS, attribute=A.HQ)
             .add("company", CT.FUNDING, "Y Combinator", attribute=A.INVESTOR)
             .add("company", CT.FUNDING, "Goose Capital", O.COMPANY, SITE, attribute=A.INVESTOR)
             .add("company", CT.FUNDING, "y combinator", O.MEDIA, PRESS, attribute=A.INVESTOR))
        a = b.build()
        location = facts_of(a, CT.LOCATION)
        self.assertEqual(len(location), 1)
        self.assertEqual(len(location[0].claims), 2)
        self.assertEqual(location[0].confirmation, ConfirmationStatus.CONFIRMED)
        funding = {f.claims[0].value.lower(): f for f in facts_of(a, CT.FUNDING)}
        self.assertEqual(set(funding), {"y combinator", "goose capital"})
        self.assertEqual(funding["y combinator"].confirmation, ConfirmationStatus.CONFIRMED)    # YC + media
        self.assertEqual(funding["goose capital"].confirmation, ConfirmationStatus.SINGLE_SOURCE)

    def test_product_status_observations_are_separate_facts_not_a_conflict(self):
        a = (Builder().add("company", CT.STAGE_SIGNAL, "Demo bookable", O.COMPANY, SITE, attribute=A.PRODUCT_STATUS)
             .add("company", CT.STAGE_SIGNAL, "SOC 2 in progress", O.COMPANY, SITE, attribute=A.PRODUCT_STATUS)).build()
        facts = facts_of(a, CT.STAGE_SIGNAL)
        self.assertEqual(len(facts), 2)
        self.assertNotIn(ConfirmationStatus.SOURCES_DISAGREE, [f.confirmation for f in facts])

    def test_scalar_conflict_is_flagged(self):
        a = (Builder().add("company", CT.STAGE_SIGNAL, "2", attribute=A.TEAM_SIZE)
             .add("company", CT.STAGE_SIGNAL, "6", O.COMPANY, SITE, attribute=A.TEAM_SIZE)).build()
        self.assertEqual(facts_of(a, CT.STAGE_SIGNAL)[0].confirmation, ConfirmationStatus.SOURCES_DISAGREE)

    def test_identity_is_one_fact_per_person_whatever_the_wording(self):
        a = (Builder().add("Jane Doe", CT.IDENTITY, "Founder/CEO of Acme")
             .add("Jane Doe", CT.IDENTITY, "Named as representative", O.COMPANY, SITE)).build()
        facts = facts_of(a, CT.IDENTITY, "Jane Doe")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].confirmation, ConfirmationStatus.CONFIRMED)

    def test_claims_without_attributes_are_one_fact_per_wording(self):
        a = (Builder().add("Jane Doe", CT.PRIOR_ROLE, "Engineer at Acme Corp", O.FOUNDER)
             .add("Jane Doe", CT.PRIOR_ROLE, "engineer at acme corp.", O.COMPANY, SITE)
             .add("Jane Doe", CT.PRIOR_ROLE, "US Army", O.FOUNDER)).build()
        facts = facts_of(a, CT.PRIOR_ROLE, "Jane Doe")
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0].confirmation, ConfirmationStatus.CORROBORATED)

    def test_founder_names_match_case_insensitively(self):
        a = Builder().add("jane doe", CT.ROLE, "CEO", O.COMPANY, SITE, attribute=A.TITLE).build()
        self.assertEqual(len(facts_of(a, CT.ROLE, "Jane Doe")), 1)


class EducationLevelTests(unittest.TestCase):
    def test_degrees_without_a_stated_level_are_separate_facts_and_never_disagree(self):
        a = (Builder()
             .add("Jane Doe", CT.EDUCATION, "Stanford University", O.FOUNDER, SITE, attribute=A.DEGREE_LEVEL_NOT_STATED)
             .add("Jane Doe", CT.EDUCATION, "Emerson College", O.MEDIA, PRESS, attribute=A.DEGREE_LEVEL_NOT_STATED)
             .build())
        facts = facts_of(a, CT.EDUCATION, "Jane Doe")
        self.assertEqual(len(facts), 2)
        self.assertNotIn(ConfirmationStatus.SOURCES_DISAGREE, [f.confirmation for f in facts])


class FounderTests(unittest.TestCase):
    def test_identity_confirmed_rule(self):
        counterparty = Builder().add("Jane Doe", CT.IDENTITY, "Founder", O.COUNTERPARTY, YC)
        company_origin = Builder().add("Jane Doe", CT.IDENTITY, "Founder", O.COMPANY, SITE)
        media_on_company_domain = Builder().add("Jane Doe", CT.IDENTITY, "Founder", O.MEDIA, "https://blog.acme.com/x")
        media_elsewhere = Builder().add("Jane Doe", CT.IDENTITY, "Founder", O.MEDIA, PRESS)
        self.assertTrue(identity_confirmed(counterparty.run, "Jane Doe"))
        self.assertFalse(identity_confirmed(company_origin.run, "Jane Doe"))
        self.assertFalse(identity_confirmed(media_on_company_domain.run, "Jane Doe"))
        self.assertTrue(identity_confirmed(media_elsewhere.run, "Jane Doe"))
        self.assertFalse(identity_confirmed(counterparty.run, "John Roe"))

    def test_a_third_party_role_claim_also_confirms_identity(self):
        role_on_accelerator = Builder().add("Jane Doe", CT.ROLE, "Founder/CEO", O.COUNTERPARTY, YC, attribute=A.TITLE)
        role_in_media = Builder().add("Jane Doe", CT.ROLE, "CEO", O.MEDIA, PRESS, attribute=A.TITLE)
        role_on_company_site = Builder().add("Jane Doe", CT.ROLE, "CEO", O.COMPANY, SITE, attribute=A.TITLE)
        role_from_founder = Builder().add("Jane Doe", CT.ROLE, "CEO", O.FOUNDER, "https://janedoe.dev", attribute=A.TITLE)
        role_third_party_on_company_domain = Builder().add("Jane Doe", CT.ROLE, "CEO", O.MEDIA, "https://blog.acme.com/x", attribute=A.TITLE)
        prior_role = Builder().add("Jane Doe", CT.PRIOR_ROLE, "Engineer, Initech", O.COUNTERPARTY, PRESS)
        self.assertTrue(identity_confirmed(role_on_accelerator.run, "Jane Doe"))
        self.assertTrue(identity_confirmed(role_in_media.run, "Jane Doe"))
        self.assertFalse(identity_confirmed(role_on_company_site.run, "Jane Doe"))
        self.assertFalse(identity_confirmed(role_from_founder.run, "Jane Doe"))
        self.assertFalse(identity_confirmed(role_third_party_on_company_domain.run, "Jane Doe"))
        self.assertFalse(identity_confirmed(prior_role.run, "Jane Doe"))            # a prior job does not link them to the company

    def test_identity_badge_has_three_levels(self):
        LINKEDIN = "https://www.linkedin.com/in/jane-doe"
        independent = Builder().add("Jane Doe", CT.IDENTITY, "Founder", O.COUNTERPARTY, YC)
        both = (Builder().add("Jane Doe", CT.IDENTITY, "Founder", O.COMPANY, SITE)
                .add("Jane Doe", CT.ROLE, "Founder", O.FOUNDER, LINKEDIN, attribute=A.TITLE, status=ClaimStatus.SNIPPET))
        site_only = Builder().add("Jane Doe", CT.IDENTITY, "Founder", O.COMPANY, SITE)
        snippet_only = Builder().add("Jane Doe", CT.ROLE, "Founder", O.FOUNDER, LINKEDIN, attribute=A.TITLE, status=ClaimStatus.SNIPPET)
        snippet_on_company_domain = (Builder().add("Jane Doe", CT.IDENTITY, "Founder", O.COMPANY, SITE)
                                     .add("Jane Doe", CT.ROLE, "Founder", O.COMPANY, "https://acme.com/team", attribute=A.TITLE, status=ClaimStatus.SNIPPET))
        prior_role_snippet = (Builder().add("Jane Doe", CT.IDENTITY, "Founder", O.COMPANY, SITE)
                              .add("Jane Doe", CT.PRIOR_ROLE, "Engineer, Initech", O.FOUNDER, LINKEDIN, status=ClaimStatus.SNIPPET))
        self.assertEqual(identity_level(independent.run, "Jane Doe"), IDENTITY_INDEPENDENT)
        self.assertEqual(identity_level(both.run, "Jane Doe"), IDENTITY_SITE_AND_SNIPPET)
        self.assertEqual(identity_level(both.run, "John Roe"), IDENTITY_UNCONFIRMED)        # someone else's claims
        for run in (site_only.run, snippet_only.run, snippet_on_company_domain.run, prior_role_snippet.run):
            self.assertEqual(identity_level(run, "Jane Doe"), IDENTITY_UNCONFIRMED)
        # the independent-identity bool does not move for the middle level
        self.assertFalse(identity_confirmed(both.run, "Jane Doe"))

    def test_self_reported_only_badge(self):
        a = (Builder()
             .add("Jane Doe", CT.PRIOR_ROLE, "Engineer", O.FOUNDER)
             .add("Jane Doe", CT.EDUCATION, "PhD", O.FOUNDER, attribute=A.PHD)
             .add("John Roe", CT.PRIOR_ROLE, "Analyst", O.FOUNDER)
             .add("John Roe", CT.PRIOR_VENTURE, "Founded Orion", O.MEDIA, PRESS)).build()
        jane, john = a.dossier.founders
        self.assertTrue(jane.background_self_reported_only)
        self.assertFalse(john.background_self_reported_only)

    def test_badge_is_off_without_any_background(self):
        a = Builder().add("Jane Doe", CT.IDENTITY, "Founder").build()
        self.assertFalse(a.dossier.founders[0].background_self_reported_only)

    def test_unknowns_go_to_their_subject(self):
        b = Builder()
        b.run.unknowns = [Unknown(subject="Jane Doe", claim_type=CT.TECHNICAL, searched_for="no GitHub"),
                          Unknown(subject="company", claim_type=CT.FUNDING, searched_for="no amount"),
                          Unknown(subject="Stranger", claim_type=CT.ROLE, searched_for="?")]
        a = b.build()
        self.assertEqual([u.claim_type for u in a.dossier.founders[0].unknowns], [CT.TECHNICAL])
        self.assertEqual([u.claim_type for u in a.dossier.unknowns], [CT.FUNDING])
        self.assertTrue(any("Stranger" in n for n in a.notes))

    def test_claims_about_people_not_in_the_input_are_set_aside_and_noted(self):
        a = Builder().add("Sigve Borgmo", CT.ROLE, "Engineer", O.FOUNDER, attribute=A.TITLE).build()
        self.assertEqual(len(a.unassigned), 1)
        self.assertTrue(any("Sigve Borgmo" in n for n in a.notes))
        self.assertTrue(all(not f.facts for f in a.dossier.founders))


class SectorAndLeadTests(unittest.TestCase):
    def test_sector_limit_keeps_primary_and_secondary(self):
        a = (Builder()
             .add("company", CT.SECTOR, "ai_software", attribute=A.SECONDARY)
             .add("company", CT.SECTOR, "industrial_hardware", attribute=A.PRIMARY)
             .add("company", CT.SECTOR, "climate_energy", attribute=A.SECONDARY)).build()
        kept = {c.value for f in facts_of(a, CT.SECTOR) for c in f.claims}
        self.assertEqual(kept, {"industrial_hardware", "ai_software"})
        self.assertTrue(any("climate_energy" in n for n in a.notes))

    def test_leads_are_separate_labelled_and_never_in_facts(self):
        b = (Builder()
             .add("company", CT.FUNDING, "Goose Capital", O.COMPANY, SITE, attribute=A.INVESTOR,
                  status=ClaimStatus.REJECTED, reason=RejectionReason.QUOTE_NOT_FOUND)
             .add("Jane Doe", CT.PRIOR_ROLE, "Engineer at Acme Corp", O.MEDIA, PRESS,
                  status=ClaimStatus.REJECTED, reason=RejectionReason.FETCH_FAILED)
             .add("Jane Doe", CT.EDUCATION, "PhD", O.FOUNDER, attribute=A.PHD,
                  status=ClaimStatus.REJECTED, reason=RejectionReason.NO_SOURCE)
             .add("Jane Doe", CT.TECHNICAL, "Paper", O.MEDIA, PRESS, status=ClaimStatus.REJECTED,
                  reason=RejectionReason.SOURCE_NOT_FETCHED))
        a = b.build()
        self.assertEqual([f.claims[0].value for f in a.dossier.leads], ["Goose Capital"])
        self.assertEqual(a.dossier.leads[0].confirmation, ConfirmationStatus.LEAD)
        jane = a.dossier.founders[0]
        self.assertEqual([f.claims[0].value for f in jane.leads], ["Engineer at Acme Corp"])       # no_source / not_fetched: log only
        self.assertEqual(a.dossier.facts, {})
        self.assertEqual(jane.facts, {})

    def test_a_lead_is_dropped_when_a_sourced_fact_already_states_it(self):
        b = (Builder()
             .add("company", CT.FUNDING, "Y Combinator", attribute=A.INVESTOR)
             .add("company", CT.FUNDING, "Y Combinator", O.MEDIA, PRESS, attribute=A.INVESTOR,
                  status=ClaimStatus.REJECTED, reason=RejectionReason.QUOTE_NOT_FOUND)
             .add("company", CT.LOCATION, "Austin", O.MEDIA, PRESS, attribute=A.HQ,
                  status=ClaimStatus.REJECTED, reason=RejectionReason.QUOTE_NOT_FOUND)
             .add("company", CT.LOCATION, "San Francisco", attribute=A.HQ))
        a = b.build()
        self.assertEqual([f.claims[0].value for f in a.dossier.leads], ["Austin"])     # a different value stays visible
        self.assertTrue(any("dropped" in n for n in a.notes))


class RenderAndLoadTests(unittest.TestCase):
    def test_render_layout_order_badge_and_labels(self):
        b = (Builder()
             .add("company", CT.COMPANY_DESCRIPTION, "Builds robots", O.COMPANY, SITE)
             .add("company", CT.FUNDING, "Y Combinator", attribute=A.INVESTOR)
             .add("Jane Doe", CT.PRIOR_ROLE, "Engineer", O.FOUNDER)
             .add("company", CT.FUNDING, "Goose Capital", O.COMPANY, SITE, attribute=A.INVESTOR,
                  status=ClaimStatus.REJECTED, reason=RejectionReason.QUOTE_NOT_FOUND))
        b.run.unknowns = [Unknown(subject="company", claim_type=CT.SECTOR, searched_for="nothing fits")]
        text = render_text(b.build())
        self.assertLess(text.index("COMPANY"), text.index("FOUNDERS"))
        self.assertLess(text.index("FOUNDERS"), text.index("UNKNOWNS (company)"))
        self.assertLess(text.index("UNKNOWNS (company)"), text.index("LEADS (company"))
        self.assertLess(text.index("What they build"), text.index("Funding"))
        self.assertIn("Background: self-reported only, not independently checked", text)
        self.assertIn("[quote not found on page] funding: Goose Capital", text)
        self.assertIn("identity NOT confirmed", text)

    def test_load_run_backfills_old_runs_from_the_submission(self):
        b = Builder().add("company", CT.LOCATION, "San Francisco", attribute=A.HQ)
        b.run.claims[0].source_url = b.run.claims[0].quote = None            # a run saved before the fields existed
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "run.json").write_text(b.run.model_dump_json())
            (Path(d) / "submission.json").write_text(json.dumps(
                {"claims": [{"source_url": YC, "quote": "Location:San Francisco"}], "unknowns": []}))
            run = load_run(Path(d))
        self.assertEqual((run.claims[0].source_url, run.claims[0].quote), (YC, "Location:San Francisco"))

    def test_leads_show_their_url(self):
        b = Builder().add("company", CT.FUNDING, "Goose Capital", O.COMPANY, SITE, attribute=A.INVESTOR,
                          status=ClaimStatus.REJECTED, reason=RejectionReason.FETCH_FAILED)
        self.assertIn(f"<{SITE}>", render_text(b.build()))
