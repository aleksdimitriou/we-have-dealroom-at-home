"""
render.py: prints an assembled dossier as plain text in the fixed layout:
company block -> founder sections -> unknowns -> leads.

A view only: it reads what assemble.py built and decides nothing.
"""

from __future__ import annotations

from schema import ClaimType, ConfirmationStatus, Fact
from src.assemble import LEAD_LABELS, Assembly

COMPANY_BLOCK = [
    (ClaimType.COMPANY_DESCRIPTION, "What they build"),
    (ClaimType.SECTOR, "Sector"),
    (ClaimType.LOCATION, "Location"),
    (ClaimType.STAGE_SIGNAL, "Stage"),
    (ClaimType.FUNDING, "Funding"),
    (ClaimType.BUSINESS_MODEL, "How they sell"),
]
FOUNDER_BLOCK = [
    (ClaimType.IDENTITY, "Identity"),
    (ClaimType.ROLE, "Role"),
    (ClaimType.PRIOR_VENTURE, "Prior ventures"),
    (ClaimType.PRIOR_ROLE, "Prior roles"),
    (ClaimType.EDUCATION, "Education"),
    (ClaimType.TECHNICAL, "Technical"),
    (ClaimType.PUBLIC_PRESENCE, "Public presence"),
]
STATUS_LABELS = {
    ConfirmationStatus.CONFIRMED: "confirmed",
    ConfirmationStatus.CORROBORATED: "corroborated",
    ConfirmationStatus.SINGLE_SOURCE: "single source",
    ConfirmationStatus.SOURCES_DISAGREE: "SOURCES DISAGREE",
    ConfirmationStatus.LEAD: "lead",
}


def _claim_line(fact: Fact) -> list[str]:
    """One line per fact; a fact whose sources disagree gets one line per claim."""
    disagree = fact.confirmation == ConfirmationStatus.SOURCES_DISAGREE
    groups = [[c] for c in fact.claims] if disagree else [fact.claims]
    label = STATUS_LABELS[fact.confirmation]
    attribute = f"{fact.attribute.value}: " if fact.attribute else ""
    lines = []
    for group in groups:
        first = group[0]
        origins = sorted({c.source_origin.value for c in group})
        urls = sorted({c.source_url for c in group if c.source_url})
        date = f" ({first.source_date})" if first.source_date else ""
        lines.append(f"  - {attribute}{first.value}{date}  [{label}; {', '.join(origins)}]  <{', '.join(urls)}>")
    return lines


def _section(types: list[tuple[ClaimType, str]], facts: dict[ClaimType, list[Fact]]) -> list[str]:
    lines = []
    for claim_type, heading in types:
        if claim_type in facts:
            lines.append(f" {heading}")
            for fact in facts[claim_type]:
                lines += _claim_line(fact)
    return lines


def _unknowns(unknowns) -> list[str]:
    return [f"  - {u.claim_type.value}: {u.searched_for}" for u in unknowns]


def _leads(leads: list[Fact]) -> list[str]:
    lines = []
    for fact in leads:
        for c in fact.claims:
            lines.append(f"  - [{LEAD_LABELS[c.rejection_reason]}] {c.claim_type.value}: {c.value}  <{c.source_url or ''}>")
    return lines


def render_text(assembly: Assembly) -> str:
    d = assembly.dossier
    out = [f"{d.company_name.upper()}  {d.company_url}",
           "Every line traces to a quote on a page; 'sourced' means a page says it, not that it is true.",
           "", "COMPANY"]
    out += _section(COMPANY_BLOCK, d.facts) or ["  (no sourced company facts)"]

    out += ["", "FOUNDERS"]
    for f in d.founders:
        out += [f"", f" {f.name}  [identity {'confirmed' if f.identity_confirmed else 'NOT confirmed'}]"]
        if f.background_self_reported_only:
            out.append("  Background: self-reported only, not independently checked")
        out += _section(FOUNDER_BLOCK, f.facts) or ["  (no sourced founder facts)"]
        if f.unknowns:
            out += ["  Unknown:"] + _unknowns(f.unknowns)
        if f.leads:
            out += ["  Leads (not facts):"] + _leads(f.leads)

    out += ["", "UNKNOWNS (company)"] + (_unknowns(d.unknowns) or ["  none recorded"])
    out += ["", "LEADS (company; not facts, never in the main fields)"] + (_leads(d.leads) or ["  none"])
    if assembly.notes:
        out += ["", "ASSEMBLY NOTES"] + [f"  - {n}" for n in assembly.notes]
    return "\n".join(out) + "\n"
