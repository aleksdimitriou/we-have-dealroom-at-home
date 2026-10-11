"""
Tests for src/evaluate.py with a fake judge (substring matching). No API calls.
Run from the project root:  python -m unittest discover tests
"""

import unittest
from datetime import datetime

from schema import (
    Action, ActionKind, Attribute, BatchRow, Claim, ClaimStatus, ClaimType, RejectionReason, Run, Source,
)
from src.evaluate import GoldenFact, Judgment, evaluate, format_report, parse_judgments

T = datetime(2026, 10, 4, 12, 0)
YC = "https://www.ycombinator.com/companies/acme"
SITE = "https://acme.com/"
PAPER = "https://arxiv.org/abs/1234.5678"


def gold(subject, claim_type, value, url=YC, origin="counterparty", linkedin_only=False):
    return GoldenFact("Acme", subject, claim_type, value, url, origin, linkedin_only)


GOLDEN = [
    gold("company", "location", "San Francisco"),
    gold("company", "funding", "Y Combinator"),
    gold("company", "funding", "Goose Capital", url=SITE, origin="company"),
    gold("Jane Doe", "education", "PhD, UT Austin", url="https://www.linkedin.com/in/jane", origin="founder", linkedin_only=True),
    gold("Jane Doe", "technical", "Paper Radiation surveys", url=PAPER, origin="founder"),
    gold("Jane Doe", "role", "Co-Founder & CEO", url=SITE, origin="company"),
]


def claim(i, subject, claim_type, value, origin="counterparty", status=ClaimStatus.SOURCED, attribute=None):
    return Claim(claim_id=f"c{i}", subject=subject, claim_type=claim_type, value=value, source_origin=origin,
                 status=status, attribute=attribute,
                 rejection_reason=RejectionReason.QUOTE_NOT_FOUND if status == ClaimStatus.REJECTED else None)


def make_run(claims):
    return Run(
        run_id="r", model="m", started_at=T, finished_at=T,
        input=BatchRow(company_name="Acme", company_url=SITE, founder_names=["Jane Doe"]),
        actions=[Action(action_id="a1", kind=ActionKind.FETCH, query_or_url=SITE, timestamp=T),
                 Action(action_id="a2", kind=ActionKind.FETCH, query_or_url=YC, timestamp=T, for_subject="Jane Doe")],
        sources=[Source(source_id="s1", url="https://acme.com", fetched_at=T, fetched_by_action="a1", content_hash="h"),
                 Source(source_id="s2", url=YC, fetched_at=T, fetched_by_action="a2", content_hash="h")],
        claims=claims)


def fake_judge(golden, claims):
    """Matches a claim to the first golden fact whose value shares its first word (case-insensitive);
    'wrong' in the claim text means contradicts; partial if the claim is shorter than the golden value."""
    out = []
    for c in claims:
        match = next((i for i, g in enumerate(golden) if g.value.lower().split()[0] == c.value.lower().split()[0]), None)
        if match is None:
            out.append(Judgment("not_in_key", None))
        elif "wrong" in c.value:
            out.append(Judgment("contradicts", match))
        elif len(c.value) < len(golden[match].value) - 5:
            out.append(Judgment("partial", match))
        else:
            out.append(Judgment("same", match))
    return out


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        claims = [
            claim(1, "company", ClaimType.LOCATION, "San Francisco", attribute=Attribute.HQ),            # same
            claim(2, "company", ClaimType.FUNDING, "Y Combinator (Summer 2026)", attribute=Attribute.INVESTOR),  # same
            claim(3, "company", ClaimType.SECTOR, "industrial_hardware"),                                 # no golden type
            claim(4, "Jane Doe", ClaimType.ROLE, "Co-Founder", origin="founder"),                         # partial, origin wrong
            claim(5, "Jane Doe", ClaimType.TECHNICAL, "Paper wrong title"),                               # contradicts
            claim(6, "Jane Doe", ClaimType.PUBLIC_PRESENCE, "NRC Fellow", origin="founder"),               # no golden type
            claim(7, "Jane Doe", ClaimType.EDUCATION, "PhD, UT Austin", status=ClaimStatus.REJECTED),     # rejected: ignored
        ]
        self.ev = evaluate(GOLDEN, {"Acme": make_run(claims)}, fake_judge)

    def verdicts(self):
        return {(g.subject, c.claim_type, c.value): j.verdict
                for g in self.ev.groups for c, j in zip(g.claims, g.judgments)}

    def test_verdicts_and_not_in_key_without_judge(self):
        v = self.verdicts()
        self.assertEqual(v[("company", "location", "San Francisco")], "same")
        self.assertEqual(v[("company", "sector", "industrial_hardware")], "not_in_key")
        self.assertEqual(v[("Jane Doe", "role", "Co-Founder")], "partial")
        self.assertEqual(v[("Jane Doe", "technical", "Paper wrong title")], "contradicts")
        self.assertEqual(len(v), 6)            # the rejected claim is not evaluated

    def test_misses_and_reachability(self):
        misses = {(m["subject"], m["claim_type"], m["value"]): m for m in self.ev.misses}
        self.assertEqual(set(misses), {("company", "funding", "Goose Capital"),
                                       ("Jane Doe", "education", "PhD, UT Austin"),
                                       ("Jane Doe", "technical", "Paper Radiation surveys")})
        self.assertTrue(misses[("company", "funding", "Goose Capital")]["reachable"])      # company site was fetched
        self.assertFalse(misses[("Jane Doe", "education", "PhD, UT Austin")]["reachable"])  # LinkedIn
        self.assertTrue(misses[("Jane Doe", "education", "PhD, UT Austin")]["linkedin_only"])
        self.assertFalse(misses[("Jane Doe", "technical", "Paper Radiation surveys")]["reachable"])  # never fetched
        self.assertTrue(misses[("Jane Doe", "technical", "Paper Radiation surveys")]["contradicted"])

    def test_report_numbers(self):
        report = format_report(self.ev)
        # 6 golden facts; covered same/partial: location, YC funding, role(partial) -> strict 2, lenient 3
        self.assertIn("MISSES: 3 golden facts not covered (1 reachable", report)
        self.assertIn("CONTRADICTIONS: 1", report)
        self.assertIn("ORIGIN ACCURACY on matched claims: 2 of 3", report)    # role claim has the wrong origin
        self.assertIn("Acme: company 1, Jane Doe 1", report)

    def test_cost_cap_leaves_claims_unjudged_and_counts_them(self):
        ev = evaluate(GOLDEN, {"Acme": make_run([claim(1, "company", ClaimType.LOCATION, "San Francisco")])},
                      lambda g, c: None)
        self.assertIn("1 claims unjudged", format_report(ev))
        self.assertEqual(len([m for m in ev.misses if m["unjudged"]]), 1)


class ParseTests(unittest.TestCase):
    def test_parse_judgments_handles_bad_output(self):
        data = {"judgments": [{"claim_id": "C0", "verdict": "same", "golden_id": "G1", "reason": "same degree"},
                              {"claim_id": "C1", "verdict": "same", "golden_id": "G9"},      # out of range
                              {"claim_id": "C2", "verdict": "bogus", "golden_id": "G0"}]}   # unknown verdict
        out = parse_judgments(data, n_golden=2, n_claims=4)                                   # C3 missing
        self.assertEqual([(j.verdict, j.golden_index) for j in out],
                         [("same", 1), ("not_in_key", None), ("not_in_key", 0), ("not_in_key", None)])
        self.assertEqual(out[0].reason, "same degree")
