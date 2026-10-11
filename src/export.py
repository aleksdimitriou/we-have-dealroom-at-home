"""
export.py: three linked CSV tables for a CRM import (built for Attio's web-app CSV import).

    companies.csv  one row per company          -> import into the Companies object
    people.csv     one row per founder          -> import into the People object
    claims.csv     one row per claim, all of them -> import into a custom object "Claims"

Links: people.Company and claims.Company hold the company's domain (Attio matches Companies on Domains, so
the link is made at import time). claims.Person holds the founder's full name.

The primary fields of companies and people hold sourced facts only. Leads (rejected claims worth a look) and
rejected claims never appear there: they are rows in claims.csv, with Visibility "lead" or "log only".

Usage (from the project root, venv active):
    python -m src.export [runs/<id> ...] [--out exports/<name>]
With no run folders, the latest finished run per company is used. Output goes to exports/ (gitignored).
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from schema import Attribute, ClaimStatus, ClaimType, ConfirmationStatus, Fact, FounderDossier, Run
from src.assemble import Assembly, IDENTITY_LABELS, LEAD_LABELS, assemble, founder_email, identity_level, latest_runs, load_run, run_flags
from src.verify import domain_of, is_on_domain, normalise_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPANY_COLUMNS = ["Name", "Domains", "Description", "Primary location", "Sector", "Business model",
                   "Stage signals", "Investors", "Funding", "Unknowns", "Leads", "Accelerator profile URL", "Flags", "Run ID"]
LINKEDIN_COLUMN = "LinkedIn (from search, not checked)"
PEOPLE_COLUMNS = ["Name", "Company", "Job title", "Email", LINKEDIN_COLUMN, "Identity confirmed", "Identity level", "Background self-reported only",
                  "Prior ventures", "Prior roles", "Education", "Technical", "Public presence", "Unknowns", "Leads",
                  "Run ID"]
CLAIM_COLUMNS = ["Claim ID", "Company", "Person", "Subject", "Claim type", "Attribute", "Value", "Normalised value",
                 "Source date", "Source URL", "Quote", "Source origin", "Origin corrected from", "Status",
                 "Confirmation", "Rejection reason", "Signal", "Visibility", "Fact key", "Run ID"]

JOIN = " | "


# --------------------------------------------------------------------------
# Text for one fact / one claim type
# --------------------------------------------------------------------------

def fact_text(fact: Fact, with_attribute: bool = False) -> str:
    """The fact's value. If sources disagree, all values are shown and flagged."""
    prefix = f"{fact.attribute.value}: " if with_attribute and fact.attribute else ""
    if fact.confirmation == ConfirmationStatus.SOURCES_DISAGREE:
        values = list(dict.fromkeys(c.value for c in fact.claims))
        return f"{prefix}{' / '.join(values)} (sources disagree)"
    return f"{prefix}{fact.claims[0].value}"


def primary(facts: dict[ClaimType, list[Fact]]) -> dict[ClaimType, list[Fact]]:
    """The facts allowed in the CRM's primary fields: page-sourced only. Snippet-status facts stay in claims.csv."""
    return {t: [f for f in fs if f.confirmation != ConfirmationStatus.SNIPPET] for t, fs in facts.items()}


def joined(facts: dict[ClaimType, list[Fact]], claim_type: ClaimType, sep: str = JOIN,
           with_attribute: bool = False, only: Optional[set[Attribute]] = None,
           skip: Optional[set[Attribute]] = None) -> str:
    items = []
    for f in facts.get(claim_type, []):
        if only is not None and f.attribute not in only:
            continue
        if skip is not None and f.attribute in skip:
            continue
        items.append(fact_text(f, with_attribute))
    return sep.join(items)


def unknown_text(unknowns) -> str:
    return JOIN.join(f"{u.claim_type.value}: {u.searched_for}" for u in unknowns)


def attio_linkedin(url: str) -> str:
    """Attio's LinkedIn attribute wants the URL to start with linkedin.com/in, so the scheme and www are dropped."""
    return re.sub(r"^https?://(www\.)?", "", url)


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------

def company_row(assembly: Assembly, run: Run) -> dict[str, str]:
    d = assembly.dossier
    facts = primary(d.facts)
    sector = sorted((f for f in facts.get(ClaimType.SECTOR, [])),
                    key=lambda f: 0 if f.attribute == Attribute.PRIMARY else 1)
    return {
        "Name": d.company_name,
        "Domains": domain_of(d.company_url),
        "Description": joined(facts, ClaimType.COMPANY_DESCRIPTION),
        "Primary location": joined(facts, ClaimType.LOCATION, only={Attribute.HQ}),
        "Sector": ", ".join(fact_text(f) for f in sector),
        "Business model": joined(facts, ClaimType.BUSINESS_MODEL),
        "Stage signals": joined(facts, ClaimType.STAGE_SIGNAL, sep="; ", with_attribute=True),
        "Investors": joined(facts, ClaimType.FUNDING, sep=", ", only={Attribute.INVESTOR}),
        "Funding": joined(facts, ClaimType.FUNDING, sep="; ", with_attribute=True, skip={Attribute.INVESTOR}),
        "Unknowns": unknown_text(d.unknowns),
        "Leads": str(sum(len(f.claims) for f in d.leads)),
        "Accelerator profile URL": run.input.accelerator_profile_url or "",
        "Flags": "; ".join(run_flags(run)),
        "Run ID": run.run_id,
    }


def linkedin_from_search(run: Run, founder: str) -> str:
    """A LinkedIn profile URL for the founder, only from a stored search result of one of the founder's own searches
    whose snippet names the company and whose title or snippet names the founder (last name). It was never fetched and
    nothing confirms it is the right person beyond that: the column header says so. Empty when there is no such result."""
    searches = {a.action_id for a in run.actions if a.kind.value == "search" and a.for_subject == founder}
    company = normalise_text(run.input.company_name).lower()
    surname = (normalise_text(founder).lower().split() or [""])[-1]
    for r in run.search_results:
        if r.action_id not in searches or not is_on_domain(r.url, "linkedin.com") or "/in/" not in r.url:
            continue
        snippet = normalise_text(r.snippet).lower()
        about_founder = re.search(rf"(?<!\w){re.escape(surname)}(?!\w)", normalise_text(r.title + " " + r.snippet).lower())
        if re.search(rf"(?<!\w){re.escape(company)}(?!\w)", snippet) and about_founder:
            slug = r.url.split("/in/", 1)[1].split("?")[0].strip("/")
            return f"linkedin.com/in/{slug}"
    return ""


def person_row(founder: FounderDossier, assembly: Assembly, run: Run) -> dict[str, str]:
    facts = primary(founder.facts)
    return {
        "Name": founder.name,
        "Company": domain_of(assembly.dossier.company_url),
        "Job title": joined(facts, ClaimType.ROLE, only={Attribute.TITLE}),
        "Email": founder_email(run, founder.name),           # from the company's own site only; the one place an email appears
        LINKEDIN_COLUMN: linkedin_from_search(run, founder.name),
        "Identity confirmed": str(founder.identity_confirmed).lower(),
        "Identity level": IDENTITY_LABELS[identity_level(run, founder.name)],
        "Background self-reported only": str(founder.background_self_reported_only).lower(),
        "Prior ventures": joined(facts, ClaimType.PRIOR_VENTURE),
        "Prior roles": joined(facts, ClaimType.PRIOR_ROLE),
        "Education": joined(facts, ClaimType.EDUCATION),
        "Technical": joined(facts, ClaimType.TECHNICAL),
        "Public presence": joined(facts, ClaimType.PUBLIC_PRESENCE),
        "Unknowns": unknown_text(founder.unknowns),
        "Leads": str(sum(len(f.claims) for f in founder.leads)),
        "Run ID": run.run_id,
    }


def claim_rows(assembly: Assembly, run: Run) -> list[dict[str, str]]:
    """One row per claim in the run, whatever its status. Visibility says where the analyst sees it:
    main (sourced, in the dossier), lead (rejected but shown in Leads), log only (rejected, audit trail only),
    left out (sourced but dropped by the sector limit, or about someone not in the input row)."""
    d = assembly.dossier
    main_facts = [f for fs in d.facts.values() for f in fs]
    main_facts += [f for fd in d.founders for fs in fd.facts.values() for f in fs]
    lead_facts = list(d.leads) + [f for fd in d.founders for f in fd.leads]
    fact_of = {c.claim_id: f for f in [*main_facts, *lead_facts] for c in f.claims}
    lead_ids = {c.claim_id for f in lead_facts for c in f.claims}
    founders = {f.name.lower() for f in d.founders}
    domain = domain_of(d.company_url)
    rows = []
    for c in run.claims:
        fact = fact_of.get(c.claim_id)
        if c.claim_id in lead_ids:
            visibility = "lead"
        elif c.status == ClaimStatus.SNIPPET and fact is not None:
            visibility = "snippet"        # shown on the page with a snippet marker; never in primary CRM fields
        elif c.status == ClaimStatus.SOURCED and fact is not None:
            visibility = "main"
        elif c.status == ClaimStatus.SOURCED:
            visibility = "left out"
        else:
            visibility = "log only"
        is_founder = c.subject.strip().lower() in founders
        email = c.claim_type == ClaimType.EMAIL                 # the address lives in people.csv only
        rows.append({
            "Claim ID": f"{run.run_id}-{c.claim_id}",
            "Company": domain,
            "Person": c.subject if is_founder else "",
            "Subject": c.subject,
            "Claim type": c.claim_type.value,
            "Attribute": c.attribute.value if c.attribute else "",
            "Value": "[see people.csv Email]" if email and c.status == ClaimStatus.SOURCED else "[removed]" if email else c.value,
            "Normalised value": "; ".join(f"{k}={v}" for k, v in (c.value_normalized or {}).items()),
            "Source date": c.source_date or "",
            "Source URL": c.source_url or "",
            "Quote": "[removed]" if email else c.quote or "",
            "Source origin": c.source_origin.value,
            "Origin corrected from": c.origin_corrected_from.value if c.origin_corrected_from else "",
            "Status": c.status.value,
            "Confirmation": fact.confirmation.value if fact and visibility in ("main", "snippet") else (
                "lead" if visibility == "lead" else ""),
            "Rejection reason": (LEAD_LABELS.get(c.rejection_reason, c.rejection_reason.value)
                                 if c.rejection_reason else ""),
            "Signal": c.signal.value if c.signal else "",
            "Visibility": visibility,
            "Fact key": fact.fact_key if fact else "",
            "Run ID": run.run_id,
        })
    return rows


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------

def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export_runs(runs: list[Run], out_dir: Path) -> dict[str, int]:
    """Assemble each run and write the three tables. One company per row: if two runs are for the same
    domain, the later run replaces the earlier one."""
    companies: dict[str, dict] = {}
    people: dict[tuple[str, str], dict] = {}
    claims: dict[str, list[dict]] = {}
    for run in runs:
        assembly = assemble(run)
        domain = domain_of(run.input.company_url)
        companies[domain] = company_row(assembly, run)
        for key in [k for k in people if k[0] == domain]:
            del people[key]
        for founder in assembly.dossier.founders:
            people[(domain, founder.name)] = person_row(founder, assembly, run)
        claims[domain] = claim_rows(assembly, run)

    out_dir.mkdir(parents=True, exist_ok=True)
    claim_list = [row for rows in claims.values() for row in rows]
    write_csv(out_dir / "companies.csv", COMPANY_COLUMNS, list(companies.values()))
    write_csv(out_dir / "people.csv", PEOPLE_COLUMNS, list(people.values()))
    write_csv(out_dir / "claims.csv", CLAIM_COLUMNS, claim_list)
    return {"companies": len(companies), "people": len(people), "claims": len(claim_list)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export run folders as three linked CSV tables.")
    parser.add_argument("runs", nargs="*", help="run folders; default: the latest finished run per company")
    parser.add_argument("--out", default=None, help="output folder (default: exports/<timestamp>)")
    args = parser.parse_args()

    dirs = [Path(d) for d in args.runs] or list(latest_runs(PROJECT_ROOT / "runs").values())
    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "exports" / datetime.now().strftime("%Y%m%d_%H%M%S")
    counts = export_runs([load_run(d) for d in dirs], out_dir)
    print(f"Wrote {counts['companies']} companies, {counts['people']} people, {counts['claims']} claims to "
          f"{out_dir.relative_to(PROJECT_ROOT) if out_dir.is_relative_to(PROJECT_ROOT) else out_dir}/")


if __name__ == "__main__":
    main()
