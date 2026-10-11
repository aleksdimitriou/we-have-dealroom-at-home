"""
keyfacts.py: picks a company's key facts, in code, from the signal tags the model put on claims.

Each claim may carry one signal kind (schema.SignalKind). A fact's signal is the first signal on its claims. Code takes
3-5 key facts per company: kinds in priority order (prior exit first), the stronger confirmation status first within a
kind, at most 2 facts per founder and at most one phd_or_publications fact per founder (a founder's papers do not crowd
out the rest). Company-level facts (a named customer, an outside investor, traction) have no founder limit. Leads and
facts without a signal are never key facts. Nothing here judges: the model tags, code picks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from schema import CompanyDossier, ConfirmationStatus, Fact, SignalKind

PRIORITY = [SignalKind.PRIOR_EXIT, SignalKind.PRIOR_VENTURE_RAISED, SignalKind.QUANTIFIED_TRACTION,
            SignalKind.NAMED_CUSTOMER_OR_PARTNER, SignalKind.OUTSIDE_INVESTOR, SignalKind.PHD_OR_PUBLICATIONS,
            SignalKind.PRIOR_ROLE_SAME_FIELD]
MAX_KEY_FACTS = 5
MAX_PER_FOUNDER = 2

# Stronger first.
STATUS_RANK = {ConfirmationStatus.CONFIRMED: 0, ConfirmationStatus.CORROBORATED: 1, ConfirmationStatus.SINGLE_SOURCE: 2,
               ConfirmationStatus.SOURCES_DISAGREE: 3, ConfirmationStatus.SNIPPET: 4}


@dataclass
class KeyFact:
    kind: SignalKind
    subject: str                 # a founder's name, or "company"
    fact: Fact


def fact_signal(fact: Fact):
    """The first signal tag on the fact's claims, or None."""
    return next((c.signal for c in fact.claims if c.signal), None)


def pick_key_facts(dossier: CompanyDossier, limit: int = MAX_KEY_FACTS, per_founder: int = MAX_PER_FOUNDER) -> list[KeyFact]:
    candidates: dict[SignalKind, list[tuple[int, int, KeyFact]]] = {k: [] for k in PRIORITY}
    order = 0
    holders = [("company", dossier.facts)] + [(f.name, f.facts) for f in dossier.founders]
    for subject, by_type in holders:
        for facts in by_type.values():
            for fact in facts:
                kind = fact_signal(fact)
                if kind is not None and fact.confirmation in STATUS_RANK:
                    candidates[kind].append((STATUS_RANK[fact.confirmation], order, KeyFact(kind, subject, fact)))
                    order += 1
    chosen: list[KeyFact] = []
    per: Counter = Counter()
    taken: set = set()
    for kind in PRIORITY:
        for _, _, key in sorted(candidates[kind], key=lambda t: (t[0], t[1])):
            if len(chosen) >= limit:
                return chosen
            if key.subject != "company" and per[key.subject] >= per_founder:
                continue
            if kind == SignalKind.PHD_OR_PUBLICATIONS and (key.subject, kind) in taken:
                continue                                                  # one per founder
            taken.add((key.subject, kind))
            chosen.append(key)
            per[key.subject] += 1
    return chosen
