"""
assemble.py: builds the dossier from a run's claims, in code. The agent never writes the dossier.

Input: a finished Run (claims already checked by verify.py; each claim carries its source_url and quote).
Output: a CompanyDossier plus a list of notes, so nothing the assembly decides is silent.

What it does:
    - groups SOURCED claims into facts by fact_key = subject / claim_type / attribute;
    - derives each fact's confirmation status (confirmed, corroborated, single_source, sources_disagree);
    - sets identity_confirmed (an independent source names the person; identity_level gives the three-level badge) and the self-reported-only
      background badge;
    - enforces the sector limit (at most 2 tags);
    - puts rejected claims in LEAD_REASONS into Leads, never into the main fields;
    - copies the agent's unknowns to the company or founder they are about.

Known limit: claims without attributes (prior_role, prior_venture, ...) are separate facts when their wording
differs, so the same job described in two ways is two facts. Merging them would need judgement, not code.

Usage (from the project root, venv active):
    python -m src.assemble runs/<id> [runs/<id> ...]
writes dossier.json and dossier.txt into each run folder.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from schema import (
    LEAD_REASONS, SET_VALUED_ATTRIBUTES,
    Claim, ClaimStatus, ClaimType, CompanyDossier, ConfirmationStatus, Fact, FounderDossier,
    RejectionReason, Run, SourceOrigin, Unknown,
)
from src.verify import domain_of, is_on_domain

BACKGROUND_TYPES = {ClaimType.PRIOR_ROLE, ClaimType.PRIOR_VENTURE, ClaimType.EDUCATION, ClaimType.TECHNICAL,
                    ClaimType.PUBLIC_PRESENCE}
IDENTITY_TYPES = (ClaimType.IDENTITY, ClaimType.ROLE)
MAX_SECTOR_TAGS = 2
THIRD_PARTY_ORIGINS = (SourceOrigin.COUNTERPARTY, SourceOrigin.DATABASE, SourceOrigin.MEDIA)

# The identity badge has three levels (derived by code from the claims; schema.py keeps the bool for an independent source).
IDENTITY_INDEPENDENT, IDENTITY_SITE_AND_SNIPPET, IDENTITY_UNCONFIRMED = "independent", "site_and_snippet", "unconfirmed"
IDENTITY_LABELS = {IDENTITY_INDEPENDENT: "confirmed by an independent source",
                   IDENTITY_SITE_AND_SNIPPET: "named as founder by the company site and a snippet",
                   IDENTITY_UNCONFIRMED: "not confirmed"}

LEAD_LABELS = {
    RejectionReason.FETCH_FAILED: "page could not be fetched",
    RejectionReason.ROBOTS_BLOCKED: "page could not be fetched (robots.txt)",
    RejectionReason.SNIPPET_ONLY: "seen only in a search snippet",
    RejectionReason.QUOTE_NOT_FOUND: "quote not found on page",
    RejectionReason.IDENTITY_UNANCHORED: "page found by search does not name the company or a known employer",
}


@dataclass
class Assembly:
    dossier: CompanyDossier
    notes: list[str] = field(default_factory=list)  # every decision the assembly made that dropped or moved something
    unassigned: list[Claim] = field(default_factory=list)   # sourced claims about someone who is not in the input row


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def normalise_value(text: str) -> str:
    """Lower-case, accents folded by NFKC, punctuation removed, spaces collapsed. For grouping and comparing only."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def comparable(claim: Claim) -> str:
    """What a claim says, in a form that can be compared: value_normalized if the agent gave one, else the value."""
    if claim.value_normalized:
        return "; ".join(f"{k}={normalise_value(v)}" for k, v in sorted(claim.value_normalized.items()))
    return normalise_value(claim.value)


def same_value(a: str, b: str) -> bool:
    """Equal, or one contains the other ('San Francisco' vs 'San Francisco, CA')."""
    return a == b or (bool(a) and bool(b) and (a in b or b in a))


def same_subject(a: str, b: str) -> bool:
    return " ".join(a.lower().split()) == " ".join(b.lower().split())


def fact_key(claim: Claim, subject: str) -> str:
    """subject / claim_type / attribute. Facts are one per attribute for scalar attributes. Set-valued
    attributes (investors, customers, ...) and claim types without attributes get one fact per distinct value.
    Identity is always one fact per person: it is one conceptual statement, however it is worded."""
    base = f"{subject}/{claim.claim_type.value}/{claim.attribute.value if claim.attribute else ''}"
    if claim.claim_type == ClaimType.IDENTITY:
        return base
    if claim.attribute is None or claim.attribute in SET_VALUED_ATTRIBUTES:
        return f"{base}={normalise_value(claim.value)}"
    return base


# --------------------------------------------------------------------------
# Confirmation
# --------------------------------------------------------------------------

def values_disagree(claims: list[Claim]) -> bool:
    """True if the claims state incompatible values for one scalar fact. Two values that carry different,
    non-empty source dates on every claim are read as an update, not a contradiction."""
    groups: list[tuple[str, set[Optional[str]]]] = []
    for c in claims:
        value = comparable(c)
        for i, (existing, dates) in enumerate(groups):
            if same_value(existing, value):
                dates.add(c.source_date)
                break
        else:
            groups.append((value, {c.source_date}))
    if len(groups) < 2:
        return False
    all_dates = [d for _, dates in groups for d in dates]
    every_claim_dated = all(c.source_date for c in claims)
    dates_never_shared = len(all_dates) == len(set(all_dates))
    return not (every_claim_dated and dates_never_shared)


def confirmation(claims: list[Claim], scalar: bool) -> ConfirmationStatus:
    """Confirmation: confirmed = a counterparty source; corroborated = two different origins other than media, none of
    them a counterparty (media mostly rewrites the company's own announcement, so it never corroborates); otherwise
    single source. Media still counts as a source for identity."""
    origins = {c.source_origin for c in claims}
    if scalar and values_disagree(claims):
        return ConfirmationStatus.SOURCES_DISAGREE
    if SourceOrigin.COUNTERPARTY in origins:
        return ConfirmationStatus.CONFIRMED
    if len(origins - {SourceOrigin.MEDIA}) >= 2:
        return ConfirmationStatus.CORROBORATED
    return ConfirmationStatus.SINGLE_SOURCE


def identity_confirmed(run: Run, founder: str) -> bool:
    """Independent identity (widened in the Oct 2026 tuning pass): a sourced identity or role claim whose origin is
    counterparty, database or media and whose source domain differs from the company's. A role claim from a
    third party ("Founder/CTO" on an accelerator page) links the person to the company as well as an identity claim."""
    sources = {s.source_id: s for s in run.sources}
    evidence = {e.evidence_id: e for e in run.evidence}
    company_domain = domain_of(run.input.company_url)
    for c in run.claims:
        if (c.status == ClaimStatus.SOURCED and c.claim_type in IDENTITY_TYPES and same_subject(c.subject, founder)
                and c.source_origin in THIRD_PARTY_ORIGINS and c.evidence_id in evidence):
            source = sources[evidence[c.evidence_id].source_id]
            if not is_on_domain(source.url, company_domain):
                return True
    return False


def identity_level(run: Run, founder: str) -> str:
    """The identity badge: 'independent' is identity_confirmed (an independent source); 'site_and_snippet'
    is a sourced identity or role claim on the company's own domain plus a snippet-tier identity or role claim from a
    page on another domain, which is two readings but neither is an independent page we opened; otherwise 'unconfirmed'."""
    if identity_confirmed(run, founder):
        return IDENTITY_INDEPENDENT
    company_domain = domain_of(run.input.company_url)
    about = [c for c in run.claims if c.claim_type in IDENTITY_TYPES and same_subject(c.subject, founder) and c.source_url]
    on_site = any(c.status == ClaimStatus.SOURCED and is_on_domain(c.source_url, company_domain) for c in about)
    snippet = any(c.status == ClaimStatus.SNIPPET and not is_on_domain(c.source_url, company_domain) for c in about)
    return IDENTITY_SITE_AND_SNIPPET if on_site and snippet else IDENTITY_UNCONFIRMED


def identity_claims(run: Run, founder: str) -> tuple[str, list[Claim]]:
    """The identity level and the claims behind it, for the badge's detail: the independent claims, the company-site
    and snippet claims, or (not confirmed) the sourced identity and role claims there are."""
    level = identity_level(run, founder)
    company_domain = domain_of(run.input.company_url)
    about = [c for c in run.claims if c.claim_type in IDENTITY_TYPES and same_subject(c.subject, founder) and c.source_url]
    sourced = [c for c in about if c.status == ClaimStatus.SOURCED]
    if level == IDENTITY_INDEPENDENT:
        return level, [c for c in sourced if c.source_origin in THIRD_PARTY_ORIGINS and not is_on_domain(c.source_url, company_domain)]
    if level == IDENTITY_SITE_AND_SNIPPET:
        return level, [c for c in sourced if is_on_domain(c.source_url, company_domain)] + \
                      [c for c in about if c.status == ClaimStatus.SNIPPET and not is_on_domain(c.source_url, company_domain)]
    return level, sourced


def founder_email(run: Run, founder: str) -> str:
    """A founder's work email: the first sourced email claim about them. verify.py admits one only from a page on
    the company's own domain, with the address on that domain. Used for people.csv; never shown on the page."""
    return next((c.value.strip() for c in run.claims if c.claim_type == ClaimType.EMAIL and c.status == ClaimStatus.SOURCED
                 and same_subject(c.subject, founder)), "")


# --------------------------------------------------------------------------
# Building facts
# --------------------------------------------------------------------------

def build_facts(claims: list[Claim], subject: str, status: ConfirmationStatus | None = None) -> list[Fact]:
    """Group claims into facts, in order of first appearance. status=None derives it (sourced claims);
    otherwise every fact gets that status (leads)."""
    grouped: dict[str, list[Claim]] = {}
    for c in claims:
        grouped.setdefault(fact_key(c, subject), []).append(c)
    facts = []
    for key, members in grouped.items():
        first = members[0]
        scalar = (first.attribute is not None and first.attribute not in SET_VALUED_ATTRIBUTES
                  and first.claim_type != ClaimType.IDENTITY)
        facts.append(Fact(fact_key=key, subject=subject, claim_type=first.claim_type, attribute=first.attribute,
                          claims=members, confirmation=status or confirmation(members, scalar)))
    return facts


def by_type(facts: list[Fact]) -> dict[ClaimType, list[Fact]]:
    out: dict[ClaimType, list[Fact]] = {}
    for f in facts:
        out.setdefault(f.claim_type, []).append(f)
    return out


def limit_sector_tags(claims: list[Claim], notes: list[str]) -> list[Claim]:
    """At most MAX_SECTOR_TAGS distinct sector values per company: primary first, then secondary, then by
    order of appearance. Claims for any other value are left out of the dossier and listed in the notes."""
    sector = [c for c in claims if c.claim_type == ClaimType.SECTOR]
    rank = {"primary": 0, "secondary": 1}
    ordered = sorted(sector, key=lambda c: rank.get(c.attribute.value if c.attribute else "", 2))
    allowed: list[str] = []
    for c in ordered:
        if c.value not in allowed and len(allowed) < MAX_SECTOR_TAGS:
            allowed.append(c.value)
    kept = []
    for c in claims:
        if c.claim_type == ClaimType.SECTOR and c.value not in allowed:
            notes.append(f"sector limit: left out {c.claim_id} sector '{c.value}' (kept {', '.join(allowed)})")
        else:
            kept.append(c)
    return kept


def assemble(run: Run) -> Assembly:
    """Build the company dossier from a run."""
    notes: list[str] = []

    founders = run.input.founder_names

    def subject_of(claim: Claim) -> Optional[str]:
        if same_subject(claim.subject, "company"):
            return "company"
        return next((f for f in founders if same_subject(claim.subject, f)), None)

    sourced: dict[str, list[Claim]] = {"company": [], **{f: [] for f in founders}}
    leads: dict[str, list[Claim]] = {"company": [], **{f: [] for f in founders}}
    snippets: dict[str, list[Claim]] = {"company": [], **{f: [] for f in founders}}
    unassigned: list[Claim] = []
    for c in run.claims:
        subject = subject_of(c)
        if c.claim_type == ClaimType.EMAIL:              # people.csv only (founder_email), never in the dossier
            continue
        if c.status == ClaimStatus.SNIPPET:
            if subject is None:
                notes.append(f"{c.claim_id} (snippet) is about '{c.subject}', who is not in the input row; left out")
            else:
                snippets[subject].append(c)
        elif c.status == ClaimStatus.SOURCED:
            if subject is None:
                unassigned.append(c)
                notes.append(f"{c.claim_id} is about '{c.subject}', who is not in the input row; not in any section")
            else:
                sourced[subject].append(c)
        elif c.status == ClaimStatus.REJECTED and c.rejection_reason in LEAD_REASONS and subject is not None:
            leads[subject].append(c)

    sourced["company"] = limit_sector_tags(sourced["company"], notes)

    unknowns: dict[str, list[Unknown]] = {"company": [], **{f: [] for f in founders}}
    for u in run.unknowns:
        subject = "company" if same_subject(u.subject, "company") else next(
            (f for f in founders if same_subject(u.subject, f)), None)
        if subject is None:
            notes.append(f"unknown about '{u.subject}' ({u.claim_type.value}) is not about the company or a founder")
        else:
            unknowns[subject].append(u)

    def lead_facts(subject: str, source_facts: list[Fact]) -> list[Fact]:
        """Leads, minus any whose statement a sourced fact already makes."""
        have = {}
        for f in source_facts:
            for c in f.claims:
                have.setdefault(f.fact_key.split("=")[0], []).append(comparable(c))
        kept = []
        for c in leads[subject]:
            key = fact_key(c, subject)
            if any(same_value(comparable(c), v) for v in have.get(key.split("=")[0], [])):
                notes.append(f"lead {c.claim_id} dropped: a sourced fact already states it")
            else:
                kept.append(c)
        return build_facts(kept, subject, ConfirmationStatus.LEAD)

    def snippet_facts(subject: str, page_facts: list[Fact]) -> list[Fact]:
        """Snippet-status facts: their own facts (key suffixed "#snippet"), never merged with page-sourced facts
        and never counted as confirmed or corroborated. A snippet claim a sourced fact already states is dropped."""
        have: dict[str, list[str]] = {}
        for f in page_facts:
            for c in f.claims:
                have.setdefault(f.fact_key.split("=")[0], []).append(comparable(c))
        kept = []
        for c in snippets[subject]:
            if any(same_value(comparable(c), v) for v in have.get(fact_key(c, subject).split("=")[0], [])):
                notes.append(f"snippet claim {c.claim_id} dropped: a page-sourced fact already states it")
            else:
                kept.append(c)
        facts = build_facts(kept, subject, ConfirmationStatus.SNIPPET)
        for f in facts:
            f.fact_key += "#snippet"
        return facts

    page_company_facts = build_facts(sourced["company"], "company")
    company_facts = page_company_facts + snippet_facts("company", page_company_facts)
    founder_dossiers = []
    for name in founders:
        page_facts = build_facts(sourced[name], name)
        facts = page_facts + snippet_facts(name, page_facts)
        background = [c for c in sourced[name] if c.claim_type in BACKGROUND_TYPES]
        founder_dossiers.append(FounderDossier(
            name=name,
            identity_confirmed=identity_confirmed(run, name),
            background_self_reported_only=bool(background) and all(c.source_origin == SourceOrigin.FOUNDER for c in background),
            facts=by_type(facts),
            unknowns=unknowns[name],
            leads=lead_facts(name, page_facts),
        ))

    dossier = CompanyDossier(
        company_name=run.input.company_name, company_url=run.input.company_url,
        facts=by_type(company_facts), founders=founder_dossiers,
        unknowns=unknowns["company"], leads=lead_facts("company", page_company_facts))
    return Assembly(dossier=dossier, notes=notes, unassigned=unassigned)


# --------------------------------------------------------------------------
# Loading a run folder, and the CLI
# --------------------------------------------------------------------------

REDIRECT = re.compile(r"^redirect: (\S+) now redirects to (\S+)$")


def run_flags(run: Run) -> list[str]:
    """What the analyst must see at the top of the dossier: a company site that redirects to another domain can mean a
    rename, a pivot or a different company, so everything read from it needs review."""
    return [f"{m.group(1)} now redirects to {m.group(2)}" for e in run.events if (m := REDIRECT.match(e))]


def load_run(run_dir: Path) -> Run:
    """Read run.json. Runs saved before claims carried source_url and quote are backfilled from
    submission.json (same order as the claims)."""
    run = Run.model_validate_json((run_dir / "run.json").read_text())
    submission_path = run_dir / "submission.json"
    if submission_path.exists() and any(c.source_url is None for c in run.claims):
        submitted = json.loads(submission_path.read_text()).get("claims", [])
        for claim, p in zip(run.claims, submitted):
            claim.source_url = claim.source_url or p.get("source_url")
            claim.quote = claim.quote or p.get("quote")
    return run


def latest_runs(runs_dir: Path) -> dict[str, Path]:
    """company name -> latest finished run directory."""
    found: dict[str, Path] = {}
    for d in sorted(runs_dir.iterdir()):
        run_json = d / "run.json"
        if run_json.exists():
            run = Run.model_validate_json(run_json.read_text())
            if run.finished_at:
                found[run.input.company_name] = d
    return found


def main() -> None:
    from src.render import render_text

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m src.assemble runs/<id> [runs/<id> ...]")
    for arg in sys.argv[1:]:
        run_dir = Path(arg)
        run = load_run(run_dir)
        assembly = assemble(run)
        (run_dir / "dossier.json").write_text(json.dumps({
            "dossier": json.loads(assembly.dossier.model_dump_json()),
            "notes": assembly.notes,
            "unassigned": [c.claim_id for c in assembly.unassigned],
        }, indent=2, ensure_ascii=False))
        text = render_text(assembly)
        (run_dir / "dossier.txt").write_text(text)
        print(text)


if __name__ == "__main__":
    main()
