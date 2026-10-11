"""
Tests for src/summary.py (with a fake model call; no API) and src/view.py (ordering, labels, filler, page safety).
Run from the project root:  python -m unittest discover tests
"""

import json
import re
import unittest

from schema import Attribute as A, Claim, ClaimStatus, ClaimType as CT, RejectionReason, SourceOrigin as O
from src.assemble import assemble
from src.llm import Usage
from src.summary import LABEL, check_summary, drop_failing, summarise_dossier, summary_input
from schema import SECTOR_TAGS, SignalKind
from src.view import (
    SECTOR_LABELS, bold_terms, build_page, company_view, is_absence, is_filler, label_text, origin_pill, segments,
)
from tests.test_assemble import Builder, PRESS, SITE, YC


def sample():
    b = (Builder()
         .add("company", CT.COMPANY_DESCRIPTION, "Builds welding robots", O.COMPANY, SITE)
         .add("company", CT.SECTOR, "aerospace_defense", attribute=A.PRIMARY)
         .add("company", CT.SECTOR, "ai_software", attribute=A.SECONDARY)
         .add("company", CT.LOCATION, "Austin", O.COMPANY, SITE, attribute=A.HQ)
         .add("company", CT.STAGE_SIGNAL, "2", attribute=A.TEAM_SIZE)
         .add("company", CT.STAGE_SIGNAL, "2026", attribute=A.FOUNDED)
         .add("company", CT.STAGE_SIGNAL, "Pilot with Acme Steel", O.COMPANY, SITE, attribute=A.CUSTOMER_OR_PILOT)
         .add("company", CT.STAGE_SIGNAL, "SOC 2 Type 1 in progress", O.COMPANY, SITE, attribute=A.PRODUCT_STATUS)
         .add("company", CT.BUSINESS_MODEL, "Book a demo", O.COMPANY, SITE)
         .add("company", CT.BUSINESS_MODEL, "Sells to steel plants through paid pilots", O.COMPANY, SITE)
         .add("company", CT.LOCATION, "Operates in Texas and Ohio", O.COMPANY, SITE, attribute=A.OPERATING_SITES)
         .add("company", CT.FUNDING, "Y Combinator", attribute=A.INVESTOR)
         .add("Jane Doe", CT.IDENTITY, "Founder/CEO of Acme")
         .add("Jane Doe", CT.ROLE, "Co-Founder & CEO", O.COMPANY, SITE, attribute=A.TITLE)
         .add("Jane Doe", CT.ROLE, "December 2025", O.FOUNDER, SITE, attribute=A.START_DATE)
         .add("Jane Doe", CT.PRIOR_VENTURE, "Founded Orion, acquired 2024", O.FOUNDER, YC,
              normalized={"organization": "Orion"})
         .add("Jane Doe", CT.EDUCATION, "PhD, UT Austin", O.FOUNDER, YC, attribute=A.PHD,
              normalized={"organization": "UT Austin"})
         .add("Jane Doe", CT.TECHNICAL, "Paper X", O.MEDIA, PRESS, status=ClaimStatus.REJECTED,
              reason=RejectionReason.QUOTE_NOT_FOUND))
    b.run.claims[-1].quote = "</script><script>alert(1)</script>"          # hostile page text must stay text
    b.run.input.accelerator_profile_url = YC
    next(c for c in b.run.claims if c.value == "Y Combinator").quote = "Y Combinator Summer 2026"
    for value, signal in (("Founded Orion, acquired 2024", SignalKind.PRIOR_EXIT), ("PhD, UT Austin", SignalKind.PHD_OR_PUBLICATIONS),
                          ("Pilot with Acme Steel", SignalKind.NAMED_CUSTOMER_OR_PARTNER)):
        next(c for c in b.run.claims if c.value == value).signal = signal
    return b.run


def ids(run):
    """Claim ids by value, so the tests do not depend on the order the sample is built in."""
    return {c.value: c.claim_id for c in run.claims}


def claim(value, claim_type=CT.STAGE_SIGNAL, attribute=None, quote=None, **kw):
    return Claim(claim_id="c", subject="company", claim_type=claim_type, attribute=attribute, value=value,
                 source_origin=kw.pop("origin", O.COMPANY), quote=quote, **kw)


class LabelAndFillerTests(unittest.TestCase):
    def test_bare_values_get_a_label(self):
        self.assertEqual(label_text(claim("2", attribute=A.TEAM_SIZE)), "Team size: 2")
        self.assertEqual(label_text(claim("2026", attribute=A.FOUNDED)), "Founded: 2026")
        self.assertEqual(label_text(claim("December 2025", CT.ROLE, A.START_DATE)), "Started: December 2025")
        self.assertEqual(label_text(claim("Co-Founder & CEO", CT.ROLE, A.TITLE)), "Co-Founder & CEO")
        self.assertEqual(label_text(claim("coming soon", attribute=A.PRODUCT_STATUS)), "Product status: coming soon")
        self.assertEqual(label_text(claim("Product status: beta", attribute=A.PRODUCT_STATUS)), "Product status: beta")

    def test_sectors_read_as_words_never_snake_case(self):
        self.assertEqual(set(SECTOR_LABELS), set(SECTOR_TAGS))
        self.assertTrue(all("_" not in label for label in SECTOR_LABELS.values()))
        self.assertEqual(label_text(claim("aerospace_defense", CT.SECTOR, A.PRIMARY)), "Primary sector: Aerospace & defence")

    def test_filler(self):
        self.assertTrue(is_filler(claim("Book a demo", CT.BUSINESS_MODEL)))
        self.assertTrue(is_filler(claim("Request a pilot", CT.BUSINESS_MODEL)))
        self.assertTrue(is_filler(claim("SOC 2 Type 1 & 2 in progress", attribute=A.PRODUCT_STATUS)))
        self.assertTrue(is_filler(claim("Hiring: founding engineer", attribute=A.PUBLIC_LISTING)))
        self.assertFalse(is_filler(claim("Job postings open: avionics engineer, hydrodynamics engineer, admin",
                                         attribute=A.PUBLIC_LISTING)))
        self.assertFalse(is_filler(claim("SOC 2 Type 2 certified", attribute=A.PRODUCT_STATUS)))
        self.assertFalse(is_filler(claim("Pilot with Acme Steel", attribute=A.CUSTOMER_OR_PILOT)))
        self.assertFalse(is_filler(claim("Book a demo", CT.PRIOR_ROLE)))           # only company signals can be filler

    def test_absences(self):
        for text in ("No funding disclosed", "No funding terms disclosed beyond Y Combinator backing",
                     "The company has not disclosed any funding", "Undisclosed", "None listed"):
            self.assertTrue(is_absence(claim(text, CT.FUNDING, A.TOTAL_RAISED)), text)
        self.assertFalse(is_absence(claim("$2M from Goose Capital", CT.FUNDING, A.TOTAL_RAISED)))
        self.assertFalse(is_absence(claim("No-code workflow builder", CT.COMPANY_DESCRIPTION)))   # only funding and stage claims


class PillAndBoldTests(unittest.TestCase):
    def pill(self, **kw):
        return origin_pill(claim("x", source_url=kw.pop("url", SITE), **kw))["label"]

    def test_origin_pills_in_plain_words(self):
        self.assertEqual(self.pill(origin=O.COUNTERPARTY, url=YC), "confirmed by YC")
        self.assertEqual(self.pill(origin=O.COUNTERPARTY, url="https://antler.co/portfolio"), "confirmed by Antler")
        self.assertEqual(self.pill(origin=O.COUNTERPARTY, url="https://www.ox.ac.uk/people"), "confirmed by ox.ac.uk")
        self.assertEqual(self.pill(origin=O.FOUNDER), "own claim")
        self.assertEqual(self.pill(origin=O.COMPANY), "company site")
        self.assertEqual(self.pill(origin=O.MEDIA), "media")
        self.assertEqual(self.pill(origin=O.DATABASE), "database")

    def test_snippet_pills_name_the_kind_of_page(self):
        c = lambda url: claim("x", source_url=url, status=ClaimStatus.SNIPPET, origin=O.FOUNDER)
        self.assertEqual(origin_pill(c("https://in.linkedin.com/in/jane"))["label"], "LinkedIn snippet")
        self.assertEqual(origin_pill(c("https://x.com/jane"))["label"], "X snippet")
        self.assertEqual(origin_pill(c("https://news.example.org/a"))["label"], "search snippet")

    def test_fact_pills_follow_the_confirmation_status(self):
        view = company_view(sample())
        pills = {f["lines"][0][0]["t"]: f["pill"] for f in view["funding"]}
        self.assertEqual(pills["Investor: "], {"label": "confirmed by YC", "kind": "confirmed"})

    def test_bold_terms_come_from_structured_fields(self):
        c = claim("Founded Orion, raised $4M from Dynamo Ventures (lead)", CT.PRIOR_VENTURE,
                  value_normalized={"organization": "Orion"})
        self.assertEqual(bold_terms(c), ["Orion", "$4M"])
        investor = claim("Dynamo Ventures (lead)", CT.FUNDING, A.INVESTOR)
        self.assertEqual(bold_terms(investor), ["Dynamo Ventures"])
        self.assertEqual(bold_terms(claim("750K USD seed", CT.FUNDING, A.TOTAL_RAISED)), ["750K USD seed"])
        self.assertEqual(bold_terms(claim("Builds welding robots")), [])             # no model judgement: nothing to bold

    def test_segments_never_overlap_and_keep_every_character(self):
        text = "Founded Orion, then Orion Labs raised $4M."
        parts = segments(text, ["Orion", "Orion Labs", "$4M"])
        self.assertEqual("".join(p["t"] for p in parts), text)
        self.assertEqual([p["t"] for p in parts if p["b"]], ["Orion", "Orion Labs", "$4M"])
        self.assertEqual(segments("plain", []), [{"t": "plain", "b": False}])


class CompanyViewTests(unittest.TestCase):
    def setUp(self):
        run = sample()
        i = ids(run)
        self.cited = [i["Founded Orion, acquired 2024"], i["PhD, UT Austin"]]
        self.view = company_view(run, {"label": LABEL, "problems": [],
                                       "sentences": [{"text": "Jane Doe founded Orion at UT Austin.", "claim_ids": self.cited}],
                                       "founder_headlines": [{"founder": "jane doe", "text": "Orion founder; PhD",
                                                              "claim_ids": self.cited}]})

    def test_review_mode_counts_every_claim_on_the_page_whether_or_not_its_row_was_opened(self):
        listed = {c["id"] for c in self.view["claims"]}
        self.assertEqual(len(listed), len(self.view["claims"]))                       # each once
        # every claim of the sample is on the page (facts, chips, header line, the lead), though few rows are drawn at load
        self.assertEqual(listed, {c.claim_id for c in sample().claims})

    def test_header_strip(self):
        v = self.view
        self.assertEqual(v["build"]["lines"][0][0]["t"], "Builds welding robots")
        self.assertEqual([(c["kind"], c["text"]) for c in v["chips"]], [
            ("Sector", "Aerospace & defence"), ("Sector", "AI software"), ("Location", "Austin"),
            ("founded", "Founded 2026"), ("team_size", "Team size 2")])
        self.assertEqual((v["accelerator"], v["batch"]), ("Y Combinator", "Summer 2026"))
        self.assertEqual(v["acceleratorUrl"], "https://www.ycombinator.com/companies/acme")

    def test_summary_and_headline_bold_the_structured_terms_of_the_claims_they_cite(self):
        bold = [p["t"] for p in self.view["summary"]["sentences"][0]["segs"] if p["b"]]
        self.assertEqual(bold, ["Orion", "UT Austin"])
        self.assertEqual([p["t"] for p in self.view["founders"][0]["headline"]["segs"] if p["b"]], ["Orion"])

    def test_founder_card(self):
        jane = self.view["founders"][0]
        self.assertEqual((jane["name"], jane["role"]), ("Jane Doe", "Co-Founder & CEO"))
        self.assertEqual((jane["identityLevel"], jane["identityLabel"]), ("independent", "confirmed by an independent source"))
        self.assertEqual([x["pill"]["kind"] for x in jane["identity"]], ["confirmed"])         # the claim behind the badge
        self.assertEqual([c["title"] for c in jane["cells"]], ["Ventures", "Roles", "Education"])
        roles = [line[0]["t"] for f in jane["cells"][1]["facts"] for line in f["lines"]]
        self.assertIn("Started: December 2025", roles)
        self.assertEqual(len(jane["leads"]), 1)
        self.assertEqual(jane["headline"]["refs"], self.cited)

    def test_key_facts_strip_is_in_signal_priority_order(self):
        keys = [(k["signal"], k["kind"], k["subject"]) for k in self.view["keyFacts"]]
        self.assertEqual(keys, [("prior_exit", "Prior exit", "Jane Doe"), ("named_customer_or_partner", "Customer or partner", "company"),
                                ("phd_or_publications", "PhD or publications", "Jane Doe")])
        self.assertEqual(self.view["keyFacts"][0]["fact"]["pill"]["kind"], "own")

    def test_how_they_sell_is_one_header_line_and_secondary_locations_go_to_everything_else(self):
        sell = "".join(p["t"] for p in self.view["sell"]["lines"][0])
        self.assertEqual(sell, "How they sell: Sells to steel plants through paid pilots")
        other = ["".join(p["t"] for p in f["lines"][0]) for f in self.view["other"]]
        self.assertIn("Operates in Texas and Ohio", other)
        for f in self.view["signals"]:
            self.assertNotIn("How they sell", "".join(p["t"] for p in f["lines"][0]))

    def test_signals_funding_and_everything_else(self):
        signals = ["".join(p["t"] for p in f["lines"][0]) for f in self.view["signals"]]
        self.assertEqual(signals, ["Pilot with Acme Steel"])                           # founded and team size are chips
        self.assertEqual(["".join(p["t"] for p in f["lines"][0]) for f in self.view["funding"]], ["Investor: Y Combinator"])
        filler = sorted("".join(p["t"] for p in f["lines"][0]) for f in self.view["filler"])
        self.assertEqual(filler, ["Book a demo", "Product status: SOC 2 Type 1 in progress"])

    def test_a_claim_that_states_an_absence_becomes_an_unknown(self):
        run = sample()
        run.claims.append(Claim(claim_id="c099", subject="company", claim_type=CT.FUNDING, attribute=A.TOTAL_RAISED,
                                value="No funding disclosed", source_origin=O.MEDIA, status=ClaimStatus.SOURCED,
                                source_url=PRESS, quote="has not disclosed funding", evidence_id="e001"))
        view = company_view(run)
        self.assertEqual(["".join(p["t"] for p in f["lines"][0]) for f in view["funding"]], ["Investor: Y Combinator"])
        self.assertTrue(any(u["text"].startswith("No funding disclosed") and u["url"] == PRESS for u in view["unknowns"]))

    def test_a_lead_or_rejected_claim_never_shows_a_confirmed_or_corroborated_pill(self):
        run = sample()
        run.claims.append(Claim(claim_id="c090", subject="company", claim_type=CT.FUNDING, attribute=A.INVESTOR, value="Goose Capital",
                                source_origin=O.COUNTERPARTY, status=ClaimStatus.REJECTED, rejection_reason=RejectionReason.ROBOTS_BLOCKED,
                                source_url="https://funds.example.org/p", quote="Goose Capital"))
        view = company_view(run)
        pills = [c["pill"] for f in view["leads"] for c in f["claims"]]
        self.assertEqual(pills, [{"label": "page could not be fetched (robots.txt)", "kind": "lead"}])
        self.assertEqual(view["leads"][0]["pill"]["kind"], "lead")
        kinds = json.dumps(pills)
        self.assertNotIn("confirmed", kinds)
        self.assertNotIn("corroborated", kinds)

    def test_database_snippets_have_their_own_pill(self):
        c = claim("x", source_url="https://www.crunchbase.com/organization/acme", status=ClaimStatus.SNIPPET, origin=O.DATABASE)
        self.assertEqual(origin_pill(c)["label"], "database snippet")

    def test_page_is_safe_self_contained_and_never_says_the_other_word(self):
        page = build_page([self.view])
        self.assertNotIn("</script><script>alert", page)                           # the hostile quote is escaped
        self.assertIn("\\u003c/script>", page)
        self.assertEqual(re.findall(r"verif", page, re.I), [])
        self.assertNotRegex(page, r"<script[^>]+src=|<link\b|localStorage|sessionStorage|innerHTML")
        data = json.loads(re.search(r'id="data">(.*?)</script>', page, re.S).group(1))
        self.assertEqual(data["companies"][0]["summary"]["label"], "Summary generated from sourced claims")

    def test_review_controls_are_hidden_by_default_and_the_how_to_read_panel_is_on_the_landing_view(self):
        page = build_page([self.view])
        self.assertIn("body:not(.review) .rev { display:none !important; }", page)
        self.assertNotIn('<body class="review"', page)
        self.assertIn('<input type="checkbox" id="review">', page)
        self.assertIn("How to read this", page)
        self.assertIn("Sourced does not mean true.", page)
        self.assertIn("Code picks 3-5 key facts per company in that priority order, preferring stronger status at equal priority, at most 2 per founder.", page)
        self.assertIn("prior_exit (prior venture acquired or public), prior_venture_raised (money raised by a prior venture), "
                      "quantified_traction (revenue, contracts, LOIs, pilots with a number), named_customer_or_partner, "
                      "outside_investor (investor other than the accelerator), phd_or_publications, "
                      "prior_role_same_field (prior role in the company's own field)", page)
        self.assertEqual(page.count("LEGEND"), 0)                                   # no legend on every company

    def test_dropped_summary_sentences_are_in_the_notes_and_the_page_never_warns(self):
        summary = {"label": LABEL, "problems": ["old format problem"], "sentences": [{"text": "Jane Doe founded Orion.", "claim_ids": self.cited}],
                   "founder_headlines": [], "dropped": [{"kind": "sentence", "text": "A great team.", "problems": ["sentence uses evaluative words: great"]}]}
        view = company_view(sample(), summary)
        self.assertIn("summary sentence dropped (sentence uses evaluative words: great): A great team.", view["notes"])
        self.assertNotIn("problems", view["summary"])
        page = build_page([view])
        self.assertNotIn("Checks found", page)

    def test_a_redirect_is_flagged_at_the_top_of_the_company_and_on_the_landing_card(self):
        run = sample()
        run.events.append("redirect: hydraoss.io now redirects to tryargus.dev")
        view = company_view(run)
        self.assertEqual(view["flags"], ["hydraoss.io now redirects to tryargus.dev"])
        self.assertEqual(company_view(sample())["flags"], [])
        page = build_page([view])
        self.assertIn("Check this company: ", page)
        self.assertIn("need review", page)
        self.assertIn('"Check: " + co.flags[0]', page)

    def test_the_page_can_carry_a_label(self):
        data = json.loads(re.search(r'id="data">(.*?)</script>', build_page([self.view], "demo version"), re.S).group(1))
        self.assertEqual(data["label"], "demo version")

    def test_run_mode(self):
        self.assertTrue(self.view["modeText"].startswith("public web only:"))


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.dossier = assemble(sample()).dossier
        self.ids = {c["id"] for c in summary_input(self.dossier)}

    def test_input_is_sourced_claims_only(self):
        rows = summary_input(self.dossier)
        self.assertEqual(len(rows), 17)                                             # the lead is not a fact
        self.assertTrue(all(set(r) == {"id", "subject", "type", "attribute", "value", "origin", "confirmation", "signal", "key", "profile"} for r in rows))
        self.assertEqual(sorted(r["value"] for r in rows if r["profile"]),
                         ["Book a demo", "Builds welding robots", "Sells to steel plants through paid pilots"])
        self.assertEqual(sorted(r["value"] for r in rows if r["key"]),
                         ["Founded Orion, acquired 2024", "PhD, UT Austin", "Pilot with Acme Steel"])
        self.assertEqual({r["value"]: r["signal"] for r in rows if r["signal"]}["PhD, UT Austin"], "phd_or_publications")

    def test_checks(self):
        good = {"sentences": [{"text": "The company says it builds welding robots.", "claim_ids": ["c001"]}],
                "founder_headlines": [{"founder": "Jane Doe", "text": "Founded Orion", "claim_ids": ["c010"]}]}
        self.assertEqual(check_summary(good, self.ids), [])
        bad = {"sentences": [{"text": "A promising team.", "claim_ids": []},
                             {"text": "It has a strong pilot.", "claim_ids": ["c999"]}], "founder_headlines": []}
        problems = check_summary(bad, self.ids)
        self.assertTrue(any("cites no claim" in p for p in problems))
        self.assertTrue(any("c999" in p for p in problems))
        self.assertTrue(any("promising" in p for p in problems) and any("strong" in p for p in problems))
        own = {"sentences": [{"text": "According to her own profile, Jane founded Orion.", "claim_ids": ["c001"]}], "founder_headlines": []}
        self.assertTrue(any("own profile" in p for p in check_summary(own, self.ids)))
        long = {"sentences": [{"text": " ".join(["word"] * 60), "claim_ids": ["c001"]}], "founder_headlines": []}
        self.assertTrue(any("60 words" in p for p in check_summary(long, self.ids)))

    def test_the_first_sentence_must_be_the_company_profile(self):
        from src.summary import profile_claim_ids
        profile = set(profile_claim_ids(self.dossier))
        self.assertEqual(len(profile), 3)
        descr = next(iter(profile))
        other = next(i for i in self.ids if i not in profile)
        good = {"sentences": [{"text": "The company says it builds welding robots.", "claim_ids": [descr]},
                              {"text": "Jane Doe founded Orion.", "claim_ids": [other]}], "founder_headlines": []}
        self.assertEqual(check_summary(good, self.ids, profile), [])
        bad = {"sentences": [{"text": "Jane Doe founded Orion.", "claim_ids": [other]},
                             {"text": "The company says it builds welding robots.", "claim_ids": [descr]}], "founder_headlines": []}
        self.assertTrue(any("first sentence must be the company profile" in p for p in check_summary(bad, self.ids, profile)))
        self.assertEqual(check_summary(bad, self.ids, set()), [])                          # no profile claims: no such rule

    def test_a_summary_without_a_profile_first_sentence_gets_the_retry_but_is_not_cut_down(self):
        from src.summary import profile_claim_ids
        descr = profile_claim_ids(self.dossier)[0]
        other = next(i for i in self.ids if i not in set(profile_claim_ids(self.dossier)))
        answers = [{"sentences": [{"text": "Jane Doe founded Orion.", "claim_ids": [other]}], "founder_headlines": []},
                   {"sentences": [{"text": "Jane Doe founded Orion.", "claim_ids": [other]}], "founder_headlines": []}]
        seen = []

        def fake(system, user):
            seen.append(user)
            return answers[len(seen) - 1], "claude-sonnet-5-5", Usage(input_tokens=10, output_tokens=5)

        out = summarise_dossier(self.dossier, fake)
        self.assertEqual(len(seen), 2)
        self.assertIn("profile_ids:", seen[0])
        self.assertIn("the first sentence must be the company profile", seen[1])
        self.assertEqual(len(out["sentences"]), 1)                       # the sentence is true and cited: it stays
        self.assertEqual(out["dropped"], [])
        del descr

    def test_retries_once_then_drops_the_sentences_that_still_fail(self):
        answers = [{"sentences": [{"text": "A promising team.", "claim_ids": ["c001"]}], "founder_headlines": []},
                   {"sentences": [{"text": "A great team.", "claim_ids": ["c001"]},
                                  {"text": "The company says it builds welding robots.", "claim_ids": ["c001"]}],
                    "founder_headlines": [{"founder": "Jane Doe", "text": "A seasoned founder", "claim_ids": ["c001"]}]}]
        seen = []

        def fake(system, user):
            seen.append(user)
            return answers[len(seen) - 1], "claude-sonnet-5-5", Usage(input_tokens=1000, output_tokens=100)

        out = summarise_dossier(self.dossier, fake)
        self.assertEqual(len(seen), 2)
        self.assertIn("promising", seen[1])                                         # the retry lists the problem
        self.assertEqual([x["text"] for x in out["sentences"]], ["The company says it builds welding robots."])
        self.assertEqual(out["founder_headlines"], [])
        self.assertEqual(out["problems"], [])                                       # nothing left to warn about
        self.assertEqual([(d["kind"], d["text"]) for d in out["dropped"]],
                         [("sentence", "A great team."), ("headline", "A seasoned founder")])
        self.assertTrue(any("great" in p for p in out["dropped"][0]["problems"]))
        self.assertEqual(out["label"], LABEL)
        self.assertAlmostEqual(out["cost_usd"], 2 * (1000 * 2 + 100 * 10) / 1e6)

    def test_surplus_sentences_are_cut(self):
        five = {"sentences": [{"text": f"Sentence {i} about the company.", "claim_ids": ["c001"]} for i in range(5)],
                "founder_headlines": []}
        clean, dropped = drop_failing(five, self.ids)
        self.assertEqual(len(clean["sentences"]), 4)
        self.assertEqual([d["problems"] for d in dropped], [["more than 4 sentences"]])

    def test_no_retry_when_clean(self):
        calls = []

        def fake(system, user):
            calls.append(1)
            return ({"sentences": [{"text": "The company says it builds welding robots.", "claim_ids": ["c001"]}],
                     "founder_headlines": []}, "claude-sonnet-5-5", Usage())

        self.assertEqual(summarise_dossier(self.dossier, fake)["problems"], [])
        self.assertEqual(len(calls), 1)
