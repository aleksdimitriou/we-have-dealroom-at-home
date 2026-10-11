"""
Tests for src/keyfacts.py: the key facts come from the model's signal tags, picked by code.
Run from the project root:  python -m unittest discover tests
"""

import unittest

from schema import Attribute as A, ClaimStatus, ClaimType as CT, SignalKind as S, SourceOrigin as O
from src.assemble import assemble
from src.keyfacts import PRIORITY, pick_key_facts
from tests.test_assemble import Builder, PRESS, SITE, YC


def tagged(builder, subject, claim_type, value, signal, origin=O.FOUNDER, url=SITE, **kw):
    builder.add(subject, claim_type, value, origin, url, **kw)
    builder.run.claims[-1].signal = signal
    return builder


def picks(builder):
    return [(k.kind, k.subject, k.fact.claims[0].value) for k in pick_key_facts(assemble(builder.run).dossier)]


class KeyFactTests(unittest.TestCase):
    def test_kinds_come_out_in_priority_order(self):
        b = Builder()
        tagged(b, "Jane Doe", CT.PRIOR_ROLE, "Engineer at Initech", S.PRIOR_ROLE_SAME_FIELD)
        tagged(b, "company", CT.FUNDING, "Goose Capital", S.OUTSIDE_INVESTOR, O.MEDIA, PRESS, attribute=A.INVESTOR)
        tagged(b, "John Roe", CT.PRIOR_VENTURE, "Orion acquired by Acme", S.PRIOR_EXIT)
        tagged(b, "company", CT.STAGE_SIGNAL, "Pilot worth $200K", S.QUANTIFIED_TRACTION, O.COMPANY, SITE, attribute=A.CUSTOMER_OR_PILOT)
        self.assertEqual([k for k, _, _ in picks(b)], [S.PRIOR_EXIT, S.QUANTIFIED_TRACTION, S.OUTSIDE_INVESTOR, S.PRIOR_ROLE_SAME_FIELD])
        self.assertEqual([k for k in PRIORITY][:2], [S.PRIOR_EXIT, S.PRIOR_VENTURE_RAISED])

    def test_a_stronger_status_wins_at_equal_priority(self):
        b = Builder()
        tagged(b, "company", CT.FUNDING, "Own-claimed investor", S.OUTSIDE_INVESTOR, O.COMPANY, SITE, attribute=A.INVESTOR)
        tagged(b, "company", CT.FUNDING, "Confirmed investor", S.OUTSIDE_INVESTOR, O.COUNTERPARTY, YC, attribute=A.INVESTOR)
        tagged(b, "company", CT.FUNDING, "Snippet investor", S.OUTSIDE_INVESTOR, O.MEDIA, PRESS, attribute=A.INVESTOR,
               status=ClaimStatus.SNIPPET)
        self.assertEqual([v for _, _, v in picks(b)], ["Confirmed investor", "Own-claimed investor", "Snippet investor"])

    def test_at_most_two_per_founder_and_five_in_all(self):
        b = Builder()
        for i in range(4):
            tagged(b, "Jane Doe", CT.TECHNICAL, f"Paper {i}", S.PHD_OR_PUBLICATIONS)
        tagged(b, "John Roe", CT.TECHNICAL, "Paper J", S.PHD_OR_PUBLICATIONS)
        for i in range(4):
            tagged(b, "company", CT.FUNDING, f"Investor {i}", S.OUTSIDE_INVESTOR, O.MEDIA, PRESS, attribute=A.INVESTOR)
        result = picks(b)
        self.assertEqual(len(result), 5)
        self.assertEqual([k for k, _, _ in result], [S.OUTSIDE_INVESTOR] * 4 + [S.PHD_OR_PUBLICATIONS])   # kinds in priority order
        self.assertEqual(result[-1][1:], ("Jane Doe", "Paper 0"))
        b2 = Builder()
        for i in range(4):
            tagged(b2, "Jane Doe", CT.TECHNICAL, f"Paper {i}", S.PHD_OR_PUBLICATIONS)
        tagged(b2, "John Roe", CT.TECHNICAL, "Paper J", S.PHD_OR_PUBLICATIONS)
        self.assertEqual([(s, v) for _, s, v in picks(b2)], [("Jane Doe", "Paper 0"), ("John Roe", "Paper J")])   # one publications fact each

    def test_the_one_per_founder_limit_is_only_for_publications(self):
        b = Builder()
        tagged(b, "Jane Doe", CT.TECHNICAL, "Paper 0", S.PHD_OR_PUBLICATIONS)
        tagged(b, "Jane Doe", CT.TECHNICAL, "Paper 1", S.PHD_OR_PUBLICATIONS)
        tagged(b, "Jane Doe", CT.PRIOR_ROLE, "Engineer at Initech", S.PRIOR_ROLE_SAME_FIELD)
        tagged(b, "Jane Doe", CT.PRIOR_ROLE, "Engineer at Acme", S.PRIOR_ROLE_SAME_FIELD)
        self.assertEqual([v for _, _, v in picks(b)], ["Paper 0", "Engineer at Initech"])      # still at most 2 per founder

    def test_untagged_facts_and_leads_are_never_key_facts(self):
        b = Builder()
        b.add("Jane Doe", CT.PRIOR_VENTURE, "Founded Orion", O.FOUNDER, SITE)               # no signal
        tagged(b, "Jane Doe", CT.PRIOR_VENTURE, "Orion acquired", S.PRIOR_EXIT, status=ClaimStatus.REJECTED,
               reason=__import__("schema").RejectionReason.QUOTE_NOT_FOUND)
        self.assertEqual(picks(b), [])

    def test_fewer_than_three_candidates_gives_fewer_key_facts(self):
        b = Builder()
        tagged(b, "John Roe", CT.PRIOR_VENTURE, "Orion raised $4M", S.PRIOR_VENTURE_RAISED)
        self.assertEqual(len(picks(b)), 1)


if __name__ == "__main__":
    unittest.main()
