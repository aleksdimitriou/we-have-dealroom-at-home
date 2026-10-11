"""
view.py: one self-contained HTML page of dossiers, for reading (and, in review mode, rating) claims.

A view only: assemble.py decides facts and confirmation; this module orders and labels them, in Python, so the
choices are testable. The page's script only draws what Python prepared. Layout of a company, top to bottom:
    header strip (name, website, accelerator and batch, what they build, chips: sector, location, founded, team size)
    -> summary box (if summary.json exists; key nouns in bold)
    -> founders (compact cards: role and identity, headline, a grid of ventures / roles / education / technical /
       recognition)
    -> Signals | Funding
    -> "Everything else" (collapsed): filler, absences, other facts, unknowns, leads, assembly notes
A landing view explains how to read the page. Every fact shows one plain-word pill (confirmed by <source>, corroborated,
own claim, company site, media, database, LinkedIn snippet, sources differ); the quote and link open on click.
Review mode (a toggle, off by default) adds Right / Wrong buttons, counts, claim ids and run ids.

Bold words come from structured data only: value_normalized.organization, investor names, amounts. Never from the
model's judgement. The page uses the word "sourced", never the other word.

Usage (from the project root, venv active):
    python -m src.view runs/<id> [runs/<id> ...] [--out exports/<name>.html]
Output goes to exports/ (gitignored): the page holds real people's data.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from schema import (
    ActionKind, Attribute, Claim, ClaimStatus, ClaimType, ConfirmationStatus, Fact, Run, SignalKind, SourceOrigin,
)
from src.assemble import IDENTITY_LABELS, LEAD_LABELS, assemble, build_facts, identity_claims, load_run, run_flags, same_subject
from src.keyfacts import pick_key_facts
from src.summary import load_summary
from src.verify import domain_of, is_on_domain, registrable_domain

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Values that mean nothing without a label ("2", "December 2025", "coming soon") get one.
VALUE_LABELS = {
    Attribute.FOUNDED: "Founded", Attribute.TEAM_SIZE: "Team size", Attribute.START_DATE: "Started",
    Attribute.TOTAL_RAISED: "Total raised", Attribute.ROUND_LABEL: "Round", Attribute.ROUND_DATE: "Round date",
    Attribute.HQ: "HQ", Attribute.OPERATING_SITES: "Operates in", Attribute.PRIMARY: "Primary sector",
    Attribute.SECONDARY: "Secondary sector", Attribute.INVESTOR: "Investor", Attribute.PRODUCT_STATUS: "Product status",
}

# Readable sector labels for the chips; the tags themselves stay snake_case in the data.
SECTOR_LABELS = {
    "ai_software": "AI software", "dev_infra": "Developer infrastructure", "enterprise_software": "Enterprise software",
    "fintech": "Fintech", "health_bio": "Health & biotech", "climate_energy": "Climate & energy",
    "mobility_logistics": "Mobility & logistics", "aerospace_defense": "Aerospace & defence",
    "industrial_hardware": "Industrial hardware", "consumer": "Consumer",
    "commerce_marketplaces": "Commerce & marketplaces", "education_work": "Education & work",
    "real_estate_construction": "Real estate & construction", "food_agri": "Food & agriculture",
}

# Source names for "confirmed by ..." and for the accelerator line (by registrable domain).
SOURCE_NAMES = {"ycombinator.com": "YC", "antler.co": "Antler", "techstars.com": "Techstars", "500.co": "500 Global",
                "entrepreneurfirst.com": "Entrepreneur First", "crunchbase.com": "Crunchbase"}
ACCELERATOR_NAMES = {"ycombinator.com": "Y Combinator", "antler.co": "Antler", "techstars.com": "Techstars",
                     "500.co": "500 Global", "entrepreneurfirst.com": "Entrepreneur First"}
DATABASE_DOMAINS = ("tracxn.com", "crunchbase.com", "pitchbook.com", "dealroom.co", "cbinsights.com", "owler.com")
KEY_LABELS = {
    SignalKind.PRIOR_EXIT: "Prior exit", SignalKind.PRIOR_VENTURE_RAISED: "Prior venture raised",
    SignalKind.QUANTIFIED_TRACTION: "Traction", SignalKind.NAMED_CUSTOMER_OR_PARTNER: "Customer or partner",
    SignalKind.OUTSIDE_INVESTOR: "Outside investor", SignalKind.PHD_OR_PUBLICATIONS: "PhD or publications",
    SignalKind.PRIOR_ROLE_SAME_FIELD: "Prior role in the field",
}
BATCH = re.compile(r"\b(?:Summer|Winter|Spring|Fall)\s+20\d\d\b")

# Founder grid cells, in order: (claim types, title).
FOUNDER_CELLS = [
    ((ClaimType.PRIOR_VENTURE,), "Ventures"), ((ClaimType.ROLE, ClaimType.PRIOR_ROLE), "Roles"),
    ((ClaimType.EDUCATION,), "Education"), ((ClaimType.TECHNICAL,), "Technical"),
    ((ClaimType.PUBLIC_PRESENCE,), "Recognition"),
]
TRACTION_ORDER = [Attribute.CUSTOMER_OR_PILOT, Attribute.PRODUCT_STATUS, Attribute.PUBLIC_LISTING,
                  Attribute.OPERATING_SITES]
FUNDING_ORDER = [Attribute.INVESTOR, Attribute.TOTAL_RAISED, Attribute.ROUND_LABEL, Attribute.ROUND_DATE]
CHIPPED = (Attribute.FOUNDED, Attribute.TEAM_SIZE)        # shown as chips, not in the Signals column

# Filler: true but low-information claims, moved to "Everything else".
_DEMO_BUTTON = re.compile(r"\b(book|request|schedule|get|try)\s+(a\s+|your\s+)?(demo|pilot|call)\b|\bcontact sales\b", re.I)
_COMPLIANCE = re.compile(r"\b(soc\s?2|iso\s?27001|gdpr|hipaa)\b", re.I)
_IN_PROGRESS = re.compile(r"in progress|pending|underway|working towards", re.I)
_JOB = re.compile(r"\b(job|jobs|hiring|careers?|open roles?|positions?|we're hiring)\b", re.I)
# A claim that states an absence ("no funding disclosed") is an unknown, not a fact.
_ABSENCE = re.compile(
    r"\b(undisclosed|not (publicly )?disclosed|none (disclosed|listed|announced)|n/a)\b"
    r"|\b(no|not|none|without)\b[^.;]{0,50}\b(disclos\w*|funding|investors?|raised|known|public(ly)?|available|announced)\b", re.I)
_AMOUNT = re.compile(r"[$€£]\s?\d[\d,.]*\s?(?:[KkMmBb]\b|million|billion|thousand)?\+?|\b\d[\d,.]*\s?(?:million|billion)\b")


def label_text(claim: Claim) -> str:
    """The claim's value, with a label when the value alone would be bare."""
    label = VALUE_LABELS.get(claim.attribute)
    value = SECTOR_LABELS.get(claim.value.strip(), claim.value) if claim.claim_type == ClaimType.SECTOR else claim.value
    if label and not value.lower().startswith(label.lower()):
        return f"{label}: {value}"
    return value


def is_filler(claim: Claim) -> bool:
    """Demo or pilot buttons, compliance work in progress, and single job postings."""
    if claim.claim_type not in (ClaimType.STAGE_SIGNAL, ClaimType.BUSINESS_MODEL):
        return False
    text = f"{claim.value} {claim.quote or ''}"
    if _DEMO_BUTTON.search(claim.value):
        return True
    if _COMPLIANCE.search(text) and _IN_PROGRESS.search(text):
        return True
    if claim.attribute == Attribute.PUBLIC_LISTING and _JOB.search(claim.value):
        listed = [part for part in re.split(r",|;|\band\b", claim.value.split(":")[-1]) if part.strip()]
        return len(listed) <= 1                          # one posting is filler; a list of roles is a signal
    return False


def is_absence(claim: Claim) -> bool:
    """A funding or stage claim that says something is missing ("no funding disclosed")."""
    return claim.claim_type in (ClaimType.FUNDING, ClaimType.STAGE_SIGNAL) and bool(_ABSENCE.search(claim.value))


# --------------------------------------------------------------------------
# Pills: one plain word or phrase per fact, coloured by strength
# --------------------------------------------------------------------------

registrable = registrable_domain


def source_name(url: str) -> str:
    return SOURCE_NAMES.get(registrable(url), domain_of(url))


def snippet_label(url: str) -> str:
    if is_on_domain(url, "linkedin.com"):
        return "LinkedIn snippet"
    if any(is_on_domain(url, d) for d in DATABASE_DOMAINS):
        return "database snippet"
    if is_on_domain(url, "x.com") or is_on_domain(url, "twitter.com"):
        return "X snippet"
    return "search snippet"


def origin_pill(claim: Claim) -> dict:
    """The pill for one claim, from who wrote it (and where it was checked)."""
    if claim.status == ClaimStatus.SNIPPET:
        return {"label": snippet_label(claim.source_url or ""), "kind": "snippet"}
    origin = claim.source_origin
    if origin == SourceOrigin.COUNTERPARTY:
        return {"label": f"confirmed by {source_name(claim.source_url or '')}", "kind": "confirmed"}
    return {SourceOrigin.FOUNDER: {"label": "own claim", "kind": "own"},
            SourceOrigin.COMPANY: {"label": "company site", "kind": "site"},
            SourceOrigin.MEDIA: {"label": "media", "kind": "media"},
            SourceOrigin.DATABASE: {"label": "database", "kind": "database"}}[origin]


def fact_pill(fact: Fact) -> dict:
    status = fact.confirmation
    if status == ConfirmationStatus.SNIPPET:
        return origin_pill(fact.claims[0])
    if status == ConfirmationStatus.SOURCES_DISAGREE:
        return {"label": "sources differ", "kind": "differ"}
    if status == ConfirmationStatus.CONFIRMED:
        counterparty = next(c for c in fact.claims if c.source_origin == SourceOrigin.COUNTERPARTY)
        return origin_pill(counterparty)
    if status == ConfirmationStatus.CORROBORATED:
        return {"label": "corroborated", "kind": "corroborated"}
    return origin_pill(fact.claims[0])


# --------------------------------------------------------------------------
# Bold words, from structured data
# --------------------------------------------------------------------------

def bold_terms(claim: Claim) -> list[str]:
    """Words to bold in this claim's text: the organisation, the investor, the amount. Structured fields only."""
    terms = []
    organisation = (claim.value_normalized or {}).get("organization")
    if organisation:
        terms.append(organisation)
    if claim.attribute == Attribute.INVESTOR:
        terms.append(re.sub(r"\s*\(.*?\)\s*$", "", claim.value).strip())
    if claim.attribute == Attribute.TOTAL_RAISED:
        terms.append(claim.value.strip())
    terms += [m.group(0).strip() for m in _AMOUNT.finditer(claim.value)]
    amount = (claim.value_normalized or {}).get("amount")
    if amount and amount in claim.value:
        terms.append(amount)
    return [t for t in dict.fromkeys(terms) if t]


def segments(text: str, terms: list[str]) -> list[dict]:
    """Split text into [{t, b}] pieces; the pieces matching a term (case-insensitive, longest term first, no
    overlaps) have b=True. The page draws them as text and <b>, never as HTML."""
    spans: list[tuple[int, int]] = []
    for term in sorted({t for t in terms if t}, key=len, reverse=True):
        for m in re.finditer(re.escape(term), text, re.I):
            if not any(m.start() < e and s < m.end() for s, e in spans):
                spans.append((m.start(), m.end()))
    out, pos = [], 0
    for s, e in sorted(spans):
        if s > pos:
            out.append({"t": text[pos:s], "b": False})
        out.append({"t": text[s:e], "b": True})
        pos = e
    if pos < len(text):
        out.append({"t": text[pos:], "b": False})
    return out or [{"t": text, "b": False}]


# --------------------------------------------------------------------------
# Facts and claims for the page
# --------------------------------------------------------------------------

def claim_view(claim: Claim, lead: bool = False) -> dict:
    """A claim for the page. A lead or rejected claim gets a lead pill, never a confirmed or corroborated one."""
    if lead or claim.status == ClaimStatus.REJECTED:
        pill = {"label": LEAD_LABELS.get(claim.rejection_reason, "not sourced"), "kind": "lead"}
    else:
        pill = origin_pill(claim)
    return {
        "id": claim.claim_id, "type": claim.claim_type.value,
        "text": segments(label_text(claim), bold_terms(claim)), "pill": pill,
        "correctedFrom": claim.origin_corrected_from.value if claim.origin_corrected_from else None,
        "quote": claim.quote or "", "url": claim.source_url or "", "date": claim.source_date or "",
        "leadLabel": LEAD_LABELS.get(claim.rejection_reason, "") if lead and claim.rejection_reason else "",
    }


def fact_view(fact: Fact, lead: bool = False, subject: str = "") -> dict:
    """One line on the page: the fact's text (every distinct text when sources differ), its pill, its claims."""
    texts, seen = [], set()
    for c in fact.claims:
        text = label_text(c)
        if text.lower() not in seen:
            seen.add(text.lower())
            texts.append(segments(text, bold_terms(c)))
    differ = fact.confirmation == ConfirmationStatus.SOURCES_DISAGREE
    pill = {"label": LEAD_LABELS.get(fact.claims[0].rejection_reason, "lead"), "kind": "lead"} if lead else fact_pill(fact)
    return {"subject": subject, "lines": texts if differ else texts[:1], "pill": pill, "differ": differ,
            "snippet": fact.confirmation == ConfirmationStatus.SNIPPET,
            "claims": [claim_view(c, lead) for c in fact.claims]}


def run_mode(run: Run) -> str:
    """What the run used, from the run's own log: public web only (searches and fetched pages)."""
    searches = sum(a.kind == ActionKind.SEARCH for a in run.actions)
    return f"public web only: {searches} searches, {run.budget.pages_fetched} pages fetched"


def _by_attribute(facts: list[Fact], order: list[Attribute]) -> list[Fact]:
    rank = {a: i for i, a in enumerate(order)}
    return sorted(facts, key=lambda f: rank.get(f.attribute, len(order)))


def accelerator_of(run: Run) -> tuple[str, str, str]:
    """(name, url, batch) of the accelerator behind the profile URL; empty strings when there is none. The batch is
    a season and year found in a sourced claim from the accelerator's own pages."""
    url = run.input.accelerator_profile_url or ""
    if not url.startswith("http"):
        return "", "", ""
    name = ACCELERATOR_NAMES.get(registrable(url), domain_of(url))
    batch = ""
    for c in run.claims:
        if c.status == ClaimStatus.SOURCED and c.source_url and registrable(c.source_url) == registrable(url):
            m = BATCH.search(f"{c.value} {c.quote or ''}")
            if m:
                batch = m.group(0)
                break
    return name, url, batch


def page_claims(dossier, identity_claims_by_founder: list[Claim]) -> list[dict]:
    """Every claim the page carries (in facts, leads, absences and the identity badges), once each, so review mode
    counts them whether or not their row has been opened."""
    seen: dict[str, str] = {}
    groups = [dossier.facts, *(f.facts for f in dossier.founders)]
    for facts in groups:
        for fs in facts.values():
            for fact in fs:
                for c in fact.claims:
                    seen.setdefault(c.claim_id, c.claim_type.value)
    for fact in [*dossier.leads, *(x for f in dossier.founders for x in f.leads)]:
        for c in fact.claims:
            seen.setdefault(c.claim_id, c.claim_type.value)
    for c in identity_claims_by_founder:
        seen.setdefault(c.claim_id, c.claim_type.value)
    return [{"id": i, "type": t} for i, t in seen.items()]


def company_view(run: Run, summary: Optional[dict] = None) -> dict:
    """Everything the page shows for one company, already ordered and labelled."""
    assembly = assemble(run)
    d = assembly.dossier
    by_id = {c.claim_id: c for c in run.claims}
    filler: list[dict] = []
    absences: list[dict] = []
    other: list[dict] = []

    def keep(facts: list[Fact], subject: str = "") -> list[dict]:
        """Facts for a section. Filler goes to "Everything else"; a fact that states an absence becomes an unknown."""
        kept = []
        for f in facts:
            if all(is_absence(c) for c in f.claims):
                c = f.claims[0]
                absences.append({"type": c.claim_type.value, "subject": subject, "text": c.value,
                                 "quote": c.quote or "", "url": c.source_url or "", "claimId": c.claim_id})
                continue
            view = fact_view(f, subject=subject)
            (filler if all(is_filler(c) for c in f.claims) else kept).append(view)
        return kept

    def plain(segs: list[dict]) -> str:
        return "".join(s["t"] for s in segs)

    def linked_segments(text: str, ids: list[str]) -> list[dict]:
        terms = [t for i in ids if i in by_id for t in bold_terms(by_id[i])]
        return segments(text, terms)

    headlines = (summary or {}).get("founder_headlines", [])
    id_claims: list[Claim] = []
    founders = []
    for f in d.founders:
        cells = []
        for types, title in FOUNDER_CELLS:
            facts = [x for t in types for x in f.facts.get(t, [])]
            views = keep(facts, f.name)
            if views:
                cells.append({"title": title, "facts": views})
        title_fact = next((x for x in f.facts.get(ClaimType.ROLE, []) if x.attribute == Attribute.TITLE), None)
        level, behind = identity_claims(run, f.name)
        id_claims += behind
        identity = (build_facts([c for c in behind if c.status == ClaimStatus.SOURCED], f.name)
                    + build_facts([c for c in behind if c.status == ClaimStatus.SNIPPET], f.name, ConfirmationStatus.SNIPPET))
        headline = next((h for h in headlines if same_subject(h.get("founder", ""), f.name)), None)
        founders.append({
            "name": f.name, "role": title_fact.claims[0].value if title_fact else "",
            "identityLevel": level, "identityLabel": IDENTITY_LABELS[level], "selfReportedOnly": f.background_self_reported_only,
            "identity": [fact_view(x, subject=f.name) for x in identity],
            "headline": {"segs": linked_segments(headline["text"], headline.get("claim_ids", [])),
                         "refs": headline.get("claim_ids", [])} if headline else None,
            "cells": cells,
            "unknowns": [{"type": u.claim_type.value, "subject": f.name, "text": u.searched_for} for u in f.unknowns],
            "leads": [fact_view(x, lead=True, subject=f.name) for x in f.leads],
        })

    facts = d.facts
    description = keep(facts.get(ClaimType.COMPANY_DESCRIPTION, []), "company")
    build, other_descriptions = (description[0], description[1:]) if description else (None, [])
    other += other_descriptions

    def chip(kind: str, text: str, fact: dict) -> dict:
        return {"kind": kind, "text": text, "differ": fact["differ"], "fact": fact}

    chips = []
    sector = sorted(facts.get(ClaimType.SECTOR, []), key=lambda f: 0 if f.attribute == Attribute.PRIMARY else 1)
    for f, v in zip(sector, keep(sector, "company")):
        chips.append(chip("Sector", SECTOR_LABELS.get(f.claims[0].value.strip(), f.claims[0].value), v))
    locations = facts.get(ClaimType.LOCATION, [])
    hq = [f for f in locations if f.attribute == Attribute.HQ] or locations[:1]
    for f in hq[:1]:
        views = keep([f], "company")
        if views:
            chips.append(chip("Location", f.claims[0].value, views[0]))
    stage = facts.get(ClaimType.STAGE_SIGNAL, [])
    for attribute, prefix in ((Attribute.FOUNDED, "Founded "), (Attribute.TEAM_SIZE, "Team size ")):
        for f in [x for x in stage if x.attribute == attribute][:1]:
            views = keep([f], "company")
            if views:
                value = f.claims[0].value
                chips.append(chip(attribute.value, value if prefix.strip().lower() in value.lower() else prefix + value, views[0]))

    signals = keep(_by_attribute([f for f in stage if f.attribute not in CHIPPED], TRACTION_ORDER), "company")
    model = keep(facts.get(ClaimType.BUSINESS_MODEL, []), "company")
    for v in model:
        v["lines"] = [[{"t": "How they sell: ", "b": False}] + line for line in v["lines"]]
    sell, other_models = (model[0], model[1:]) if model else (None, [])     # one line in the header; the rest below
    other += other_models
    other += keep([f for f in locations if f not in hq[:1]], "company")     # "Operates in" and other locations
    funding = keep(_by_attribute(facts.get(ClaimType.FUNDING, []), FUNDING_ORDER), "company")
    shown = {ClaimType.STAGE_SIGNAL, ClaimType.FUNDING, ClaimType.COMPANY_DESCRIPTION, ClaimType.SECTOR,
             ClaimType.LOCATION, ClaimType.BUSINESS_MODEL}
    for t, fs in facts.items():                       # any other claim type about the company
        if t not in shown:
            other += keep(fs, "company")

    key_facts = []
    for k in pick_key_facts(d):
        if any(is_filler(c) or is_absence(c) for c in k.fact.claims):
            continue
        view = fact_view(k.fact, subject=k.subject)
        key_facts.append({"kind": KEY_LABELS[k.kind], "signal": k.kind.value, "subject": k.subject, "fact": view})
    accelerator, accelerator_url, batch = accelerator_of(run)
    unknowns = [{"type": u.claim_type.value, "subject": "company", "text": u.searched_for} for u in d.unknowns]
    unknowns += [dict(a, text=f"{a['text']} (stated by the source, so recorded as unknown)") for a in absences]
    return {
        "name": d.company_name, "url": d.company_url, "runId": run.run_id, "modeText": run_mode(run),
        "accelerator": accelerator, "acceleratorUrl": accelerator_url, "batch": batch,
        "build": build, "sell": sell, "chips": chips, "keyFacts": key_facts,
        "summary": None if not summary else {
            "label": summary.get("label", ""),
            "sentences": [{"segs": linked_segments(s["text"], s.get("claim_ids", [])), "refs": s.get("claim_ids", [])}
                          for s in summary.get("sentences", [])]},
        "founders": founders, "signals": signals, "funding": funding,
        "filler": filler, "other": other, "claims": page_claims(d, id_claims),
        "unknowns": unknowns, "leads": [fact_view(x, lead=True, subject="company") for x in d.leads],
        "notes": assembly.notes + [f"summary {d['kind']} dropped ({'; '.join(d['problems'])}): {d['text'][:90]}"
                                   for d in (summary or {}).get("dropped", [])],
        "buildText": plain(build["lines"][0]) if build else "",
        "flags": run_flags(run),
    }


def build_page(companies: list[dict], label: str = "") -> str:
    # ensure_ascii escapes every non-ASCII character; '<' is escaped so page text can never close the script tag.
    data = json.dumps({"companies": companies, "label": label}, ensure_ascii=True).replace("<", "\\u003c")
    return PAGE.replace("__DATA__", data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dossier page for some run folders.")
    parser.add_argument("runs", nargs="+", help="run folders")
    parser.add_argument("--out", default=None, help="output file (default: exports/dossiers_<timestamp>.html)")
    parser.add_argument("--label", default="", help='a label shown on the landing view, e.g. "demo version"')
    args = parser.parse_args()
    companies = []
    for d in args.runs:
        run = load_run(Path(d))
        companies.append(company_view(run, load_summary(Path(d))))
    companies.sort(key=lambda c: c["name"].lower())
    out = Path(args.out) if args.out else PROJECT_ROOT / "exports" / f"dossiers_{datetime.now():%Y%m%d_%H%M%S}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page(companies, args.label), encoding="utf-8")
    print(out)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Founder dossiers</title>
<style>
  :root { color-scheme: light; --navy:#14213D; --ink:#2b2f36; --muted:#5f6672; --rule:#d9dce1; --rule-soft:#e8eaed; --bg:#fff;
          --accent:#2a6f7f; --serif:Georgia, "Times New Roman", serif; --sans:system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.6 var(--sans); color:var(--ink); background:var(--bg); -webkit-print-color-adjust:exact; print-color-adjust:exact; }
  .layout { display:flex; min-height:100vh; }
  nav { width:236px; flex:none; border-right:1px solid var(--rule); background:#fff; padding:22px 14px; position:sticky; top:0; align-self:flex-start; height:100vh; overflow:auto; }
  nav h1 { font:700 17px/1.3 var(--serif); color:var(--navy); margin:0 0 14px; padding-bottom:10px; border-bottom:1px solid var(--rule); }
  nav button.tab { display:block; width:100%; text-align:left; background:transparent; border:0; border-left:3px solid transparent; border-radius:0; padding:6px 10px; margin-bottom:2px; cursor:pointer; font:inherit; color:var(--ink); }
  nav button.tab:hover { background:#f4f5f7; } nav button.tab[aria-selected="true"] { border-left-color:var(--accent); color:var(--navy); font-weight:600; background:#f4f5f7; }
  .small { display:block; font-weight:400; font-size:12px; color:var(--muted); }
  .toggle { display:flex; gap:8px; align-items:center; margin:16px 0 0; padding-top:12px; border-top:1px solid var(--rule); font-size:13px; cursor:pointer; }
  #copy { margin-top:10px; width:100%; font:inherit; font-size:13px; padding:6px; border-radius:2px; border:1px solid var(--accent); background:#fff; color:var(--accent); cursor:pointer; }
  #copytext { width:100%; height:120px; margin-top:6px; font:12px ui-monospace, Menlo, monospace; display:none; }
  main { flex:1; min-width:0; max-width:1000px; padding:0 36px 72px; }
  body:not(.review) .rev { display:none !important; }
  h2 { font:700 28px/1.2 var(--serif); margin:0; color:inherit; }
  h3 { font:700 12px/1.4 var(--sans); text-transform:uppercase; letter-spacing:.09em; color:var(--navy); margin:34px 0 10px; padding-bottom:6px; border-bottom:1px solid var(--rule); }
  a { color:var(--accent); }
  .cohead { background:var(--navy); color:#fff; margin:0 -36px; padding:26px 36px 20px; }
  .cohead h2 { color:#fff; } .cohead .sub { color:#c9d1de; margin-top:6px; } .cohead a { color:#bfe3ea; }
  .sub { color:var(--muted); font-size:13.5px; margin:2px 0 0; }
  .build { font:400 20px/1.45 var(--serif); color:var(--navy); margin:22px 0 8px; }
  .sell { font-size:15px; color:var(--muted); margin:0 0 14px; cursor:pointer; }
  .keyfacts { margin:26px 0 0; padding:2px 0 2px 18px; border-left:3px solid var(--accent); counter-reset:kf; }
  .keyfacts h3 { margin:0 0 6px; padding:0; border:0; color:var(--accent); }
  .keyfacts .fact { position:relative; padding-left:28px; counter-increment:kf; }
  .keyfacts .fact::before { content:counter(kf) "."; position:absolute; left:0; top:3px; font:700 14px/1.6 var(--serif); color:var(--accent); }
  .keyfacts .kind { font-size:11.5px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin-right:8px; white-space:nowrap; }
  .flag { border:1px solid #d9b45a; border-left:4px solid #b8860b; background:#fdf8ea; padding:10px 14px; margin:16px 0 0; color:#5a3d00; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin:0 0 8px; }
  .chip { font:inherit; font-size:13px; padding:1px 11px; border-radius:99px; border:1px solid var(--rule); background:#fff; cursor:pointer; color:var(--ink); }
  .chip:hover { border-color:var(--muted); } .chip.differ { border-color:#d9b45a; background:#fdf6e3; }
  .chip.sector { border-color:var(--accent); color:var(--accent); }
  .slot:empty { display:none; } .slot { margin:4px 0 0; }
  .box { border-top:1px solid var(--rule); border-bottom:1px solid var(--rule); padding:14px 0; margin:26px 0 0; }
  .summary p { margin:6px 0; } .summary .label { font-size:12px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); margin:0 0 6px; }
  .refs { font-size:11px; vertical-align:super; } .refs a { margin-left:3px; text-decoration:none; }
  .warn { font-size:12.5px; color:#8a3b12; }
  .founder { border-top:1px solid var(--rule); padding:16px 0 10px; margin:0; }
  .fhead { display:flex; flex-wrap:wrap; gap:4px 12px; align-items:baseline; }
  .fhead .name { font:700 19px/1.3 var(--serif); color:var(--navy); } .fhead .role { color:var(--muted); }
  .headline { font-weight:600; margin:6px 0 10px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px 32px; margin-top:8px; }
  .cell h4 { font:600 11.5px/1.4 var(--sans); text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:0 0 4px; }
  .cols { display:grid; grid-template-columns:1fr 1fr; gap:0 36px; }
  .fact { padding:4px 0; border-top:1px solid var(--rule-soft); } .fact:first-of-type { border-top:0; }
  .fact details > summary { cursor:pointer; list-style:none; display:flex; gap:10px; align-items:baseline; justify-content:space-between; }
  .fact details > summary::-webkit-details-marker { display:none; }
  .ftext { flex:1; min-width:0; overflow-wrap:anywhere; } .ftext div + div { margin-top:2px; }
  .fact.snippet .ftext { color:var(--muted); } .fact.differ { border-left:3px solid #d9b45a; padding-left:9px; }
  .subjecttag { color:var(--muted); font-size:12.5px; }
  /* Pills, strongest to weakest: navy fill, slate-blue, soft greys and tones, amber for disagreement, pale dashed for snippets.
     The words carry the meaning; colour only reinforces it. */
  .pill { display:inline-block; flex:none; font-size:11.5px; line-height:1.55; padding:0 9px; border-radius:99px; border:1px solid; white-space:nowrap; }
  .p-confirmed { background:var(--navy); border-color:var(--navy); color:#fff; }
  .p-corroborated { background:#dbe4ee; border-color:#8ea3bd; color:var(--navy); }
  .p-own { background:#ececec; border-color:#c4c4c4; color:#33363b; }
  .p-site { background:#e7ebef; border-color:#bcc5ce; color:#33363b; }
  .p-media { background:#f0e9de; border-color:#d3c6b0; color:#4a3f2c; }
  .p-database { background:#e1ecec; border-color:#aac4c4; color:#274646; }
  .p-differ { background:#f6e3b4; border-color:#d9b45a; color:#5a3d00; }
  .p-snippet { background:#fafafa; border:1px dashed #8d929a; color:#5f6672; }
  .p-lead { background:#f3e6df; border-color:#d3b3a3; color:#6e3519; }
  .evidence { margin:6px 0 8px 0; padding:4px 0 4px 14px; border-left:2px solid var(--rule); }
  .evidence .row { margin:6px 0; } blockquote { margin:6px 0; padding:0 0 0 12px; border-left:2px solid var(--rule); color:#434850; font-size:13.5px; white-space:pre-wrap; overflow-wrap:anywhere; }
  .meta { font-size:12.5px; color:var(--muted); display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  .badge { display:inline-block; font-size:12px; padding:0 10px; border-radius:99px; border:1px solid; cursor:pointer; }
  .badge.id-independent { background:var(--navy); border-color:var(--navy); color:#fff; }
  .badge.id-site_and_snippet { background:#dbe4ee; border-color:#8ea3bd; color:var(--navy); }
  .badge.id-unconfirmed { background:#fafafa; border:1px dashed #8d929a; color:#5f6672; }
  .badge.self { background:#f6e3b4; border-color:#d9b45a; color:#5a3d00; cursor:default; }
  .rate { display:inline-flex; gap:4px; } .rate button { font:inherit; font-size:12px; padding:0 9px; border-radius:2px; cursor:pointer; background:#fff; border:1px solid; }
  .rate .right { color:#14532d; border-color:#86c99a; } .rate .right[aria-pressed="true"] { background:#14532d; color:#fff; }
  .rate .wrong { color:#991b1b; border-color:#e49a9a; } .rate .wrong[aria-pressed="true"] { background:#991b1b; color:#fff; }
  .counts { display:flex; gap:14px; flex-wrap:wrap; font-size:14px; margin:8px 0 0; }
  details.else { margin-top:34px; border-top:1px solid var(--rule); padding-top:10px; } details.else > summary { cursor:pointer; font-weight:600; color:var(--navy); }
  .empty { color:var(--muted); font-style:italic; margin:2px 0; } ul.plain { margin:4px 0 0 18px; padding:0; }
  #co-start > h2 { color:var(--navy); margin-top:30px; }
  .howto { border-top:1px solid var(--rule); padding:2px 0 8px; margin:18px 0; max-width:780px; }
  .howto h3 { margin-top:16px; }
  .howto dt { margin-top:10px; } .howto dd { margin:3px 0 0 0; color:#434850; }
  .cards { display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:0 28px; margin-top:8px; }
  .card { text-align:left; font:inherit; background:#fff; border:0; border-top:1px solid var(--rule); border-radius:0; padding:12px 2px; cursor:pointer; color:var(--ink); }
  .card:hover { background:#f4f5f7; } .card b { display:block; font:700 16px/1.3 var(--serif); color:var(--navy); } .card span { color:var(--muted); font-size:13px; }
  .flash { background:#fdf6e3; }
  footer { margin-top:36px; font-size:12.5px; color:var(--muted); }
  @media (max-width: 760px) { .layout { display:block; } nav { width:auto; height:auto; position:static; border-right:0; border-bottom:1px solid var(--rule); padding:14px 16px; } main { padding:0 16px 40px; }
    .cohead { margin:0 -16px; padding:18px 16px 14px; } h2 { font-size:23px; }
    .grid, .cols { grid-template-columns:1fr; } .build { font-size:18px; } }
  @media print { nav { display:none; } .layout { display:block; } main { max-width:none; padding:0; } body { font-size:11pt; line-height:1.45; }
    .cohead { margin:0; padding:14px 0 10px 12px; } .founder, .keyfacts, .box, .fact { break-inside:avoid; } h3 { break-after:avoid; }
    .chip, .card { border-color:#999; } a { color:inherit; text-decoration:none; } .cohead a { color:#fff; } .rev { display:none !important; } }
</style>
</head>
<body>
<div class="layout">
  <nav aria-label="Companies"><h1>Founder dossiers</h1>
    <div id="tabs" role="tablist"></div>
    <label class="toggle"><input type="checkbox" id="review"> Review mode</label>
    <div class="rev"><button type="button" id="copy">Copy counts as text</button><span class="small" id="copystatus"></span>
      <textarea id="copytext" readonly aria-label="Counts as text"></textarea></div></nav>
  <main id="main"></main>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const ratings = {};          // "company index|claim id" -> "right" | "wrong"; in memory only
const claimsOf = [];         // per company: [{key, id, type}]
const factDetails = [];      // every fact <details>, so review mode can open them all

function el(tag, props, ...kids) {             // builds DOM nodes; text is always set as text, never as HTML
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) n.append(kid);
  return n;
}
function safeUrl(u) { try { const x = new URL(u); return /^https?:$/.test(x.protocol) ? x.href : null; } catch (e) { return null; } }
function shortUrl(u) { return u.replace(/^https?:\/\/(www\.)?/, "").replace(/\/$/, ""); }
function anchor(ci, id) { return "claim-" + ci + "-" + id; }
function segs(list) { const f = document.createDocumentFragment(); for (const s of list) f.append(s.b ? el("b", {text: s.t}) : document.createTextNode(s.t)); return f; }
function pill(p) { return el("span", {class: "pill p-" + p.kind, text: p.label}); }
function link(url) { const href = safeUrl(url); return href ? el("a", {href, target: "_blank", rel: "noopener noreferrer", text: shortUrl(href)}) : el("span", {text: url || "no link"}); }

function rating(c, ci) {
  const key = ci + "|" + c.id;
  if (!claimsOf[ci].some(x => x.key === key)) claimsOf[ci].push({key, id: c.id, type: c.type || ""});
  const right = el("button", {type: "button", class: "right", "aria-pressed": "false", text: "Right"});
  const wrong = el("button", {type: "button", class: "wrong", "aria-pressed": "false", text: "Wrong"});
  const rate = v => {
    if (ratings[key] === v) delete ratings[key]; else ratings[key] = v;
    right.setAttribute("aria-pressed", String(ratings[key] === "right")); wrong.setAttribute("aria-pressed", String(ratings[key] === "wrong"));
    updateCounts(ci);
  };
  right.addEventListener("click", () => rate("right")); wrong.addEventListener("click", () => rate("wrong"));
  return el("span", {class: "rate rev"}, right, wrong);
}
function evidence(f, ci) {        // the claims behind a fact: pill, quote, link, date; ids and ratings in review mode
  const box = el("div", {class: "evidence"});
  for (const c of f.claims) {
    const meta = el("div", {class: "meta"}, pill(c.pill), link(c.url));
    if (c.leadLabel) meta.append(el("span", {text: c.leadLabel}));
    if (c.date) meta.append(el("span", {text: "dated " + c.date}));
    if (c.correctedFrom) meta.append(el("span", {class: "rev", text: "origin set by code; the model said " + c.correctedFrom}));
    meta.append(el("span", {class: "rev", text: c.id}), rating(c, ci));
    box.append(el("div", {class: "row", id: anchor(ci, c.id)}, el("div", {}, segs(c.text)), el("blockquote", {text: c.quote || "(no quote)"}), meta));
  }
  return box;
}
function factRow(f, ci) {
  const text = el("div", {class: "ftext"});
  if (f.kindLabel) text.append(el("span", {class: "kind", text: f.kindLabel}));
  if (f.subject && f.subject !== "company" && f.showSubject) text.append(el("span", {class: "subjecttag", text: f.subject + " · "}));
  for (const line of f.lines) text.append(el("div", {}, segs(line)));
  const head = el("summary", {}, text, pill(f.pill));
  const details = el("details", {}, head, evidence(f, ci));
  factDetails.push(details);
  return el("div", {class: "fact" + (f.differ ? " differ" : "") + (f.snippet ? " snippet" : "")}, details);
}
function refs(ids, ci) {
  const span = el("span", {class: "refs rev"});
  for (const id of ids) span.append(el("a", {href: "#" + anchor(ci, id), text: id, onclick: e => { e.preventDefault(); jump(ci, id); }}));
  return span;
}
function jump(ci, id) {
  const row = document.getElementById(anchor(ci, id));
  if (!row) return;
  for (let n = row.parentElement; n; n = n.parentElement) if (n.tagName === "DETAILS") n.open = true;
  row.scrollIntoView({block: "center"}); row.classList.add("flash"); setTimeout(() => row.classList.remove("flash"), 1200);
}
function factList(list, ci, showSubject) {
  const box = el("div", {});
  for (const f of list) { f.showSubject = showSubject; box.append(factRow(f, ci)); }
  return box;
}
function unknownList(items, ci) {
  if (!items.length) return el("p", {class: "empty", text: "None recorded."});
  const ul = el("ul", {class: "plain"});
  for (const u of items) {
    const li = el("li", {}, el("b", {text: (u.subject && u.subject !== "company" ? u.subject + " · " : "") + u.type + ": "}), u.text);
    if (u.url) li.append(" ", link(u.url));
    if (u.claimId) li.append(" ", rating({id: u.claimId, type: u.type}, ci));          // a claim that states an absence is a claim too
    ul.append(li);
  }
  return ul;
}

function companyPage(co, ci) {
  claimsOf[ci] = co.claims.map(c => ({key: ci + "|" + c.id, id: c.id, type: c.type}));      // every claim on the page, opened or not
  const page = el("section", {id: "co-" + ci, hidden: ""});
  const slot = el("div", {class: "slot"});
  const showFact = (f, key) => { if (slot.dataset.key === key && slot.firstChild) { slot.replaceChildren(); slot.dataset.key = ""; return; }
    slot.dataset.key = key; slot.replaceChildren(evidence(f, ci)); };

  // 1. header strip
  const sub = el("p", {class: "sub"}, link(co.url));
  if (co.accelerator) { sub.append(" · "); const a = safeUrl(co.acceleratorUrl); sub.append(a ? el("a", {href: a, target: "_blank", rel: "noopener noreferrer", text: co.accelerator}) : co.accelerator); }
  if (co.batch) sub.append(" · " + co.batch);
  sub.append(el("span", {class: "rev", text: " · run " + co.runId + " · " + co.modeText}));
  page.append(el("header", {class: "cohead"}, el("h2", {text: co.name}), sub));
  for (const f of co.flags) page.append(el("div", {class: "flag"}, el("b", {text: "Check this company: "}), f + ". This can mean a rename, a pivot or a different company: the founders and facts read from that site need review."));
  if (co.build) {
    const b = el("p", {class: "build", title: "Click for the source"}, segs(co.build.lines[0]));
    b.style.cursor = "pointer"; b.addEventListener("click", () => showFact(co.build, "build"));
    page.append(b);
  }
  if (co.sell) {
    const sl = el("p", {class: "sell", title: "Click for the source"}, segs(co.sell.lines[0]));
    sl.addEventListener("click", () => showFact(co.sell, "sell"));
    page.append(sl);
  }
  const chips = el("div", {class: "chips"});
  co.chips.forEach((c, i) => chips.append(el("button", {type: "button", class: "chip" + (c.kind === "Sector" ? " sector" : "") + (c.differ ? " differ" : ""),
    title: c.differ ? "Sources differ; click for each" : "Click for the source", text: c.text, onclick: () => showFact(c.fact, "chip" + i)})));
  page.append(chips, slot, el("div", {class: "counts rev", id: "counts-" + ci}));

  // 1b. key facts
  if (co.keyFacts.length) {
    const kf = el("div", {class: "keyfacts"}, el("h3", {text: "Key facts"}));
    for (const k of co.keyFacts) {
      k.fact.showSubject = k.subject !== "company"; k.fact.kindLabel = k.kind;
      kf.append(factRow(k.fact, ci));
    }
    page.append(kf);
  }

  // 2. summary
  if (co.summary && co.summary.sentences.length) {
    const box = el("div", {class: "box summary"}, el("p", {class: "label", text: co.summary.label}));
    for (const s of co.summary.sentences) box.append(el("p", {}, segs(s.segs), refs(s.refs, ci)));
    page.append(box);
  }

  // 3. founders
  page.append(el("h3", {text: "Founders"}));
  for (const f of co.founders) {
    const head = el("div", {class: "fhead"}, el("span", {class: "name", text: f.name}));
    if (f.role) head.append(el("span", {class: "role", text: f.role}));
    const idBadge = el("details", {style: "display:inline-block"}, el("summary", {style: "list-style:none;cursor:pointer;display:inline"}, el("span", {class: "badge id-" + f.identityLevel, text: "Identity: " + f.identityLabel})));
    if (f.identity.length) { for (const x of f.identity) idBadge.append(evidence(x, ci)); } else idBadge.append(el("div", {class: "evidence", text: "No identity or role claim."}));
    head.append(idBadge);
    if (f.selfReportedOnly) head.append(el("span", {class: "badge self", title: "Every background fact comes from the founder's own pages", text: "background: own claims only"}));
    const card = el("div", {class: "founder"}, head);
    if (f.headline) card.append(el("div", {class: "headline"}, segs(f.headline.segs), refs(f.headline.refs, ci)));
    if (f.cells.length) {
      const grid = el("div", {class: "grid"});
      for (const cell of f.cells) grid.append(el("div", {class: "cell"}, el("h4", {text: cell.title}), factList(cell.facts, ci, false)));
      card.append(grid);
    } else card.append(el("p", {class: "empty", text: "No sourced facts."}));
    page.append(card);
  }

  // 4. signals | funding
  const cols = el("div", {class: "cols"});
  cols.append(el("div", {}, el("h3", {text: "Signals"}), co.signals.length ? factList(co.signals, ci, false) : el("p", {class: "empty", text: "None found."})),
              el("div", {}, el("h3", {text: "Funding"}), co.funding.length ? factList(co.funding, ci, false) : el("p", {class: "empty", text: "None found."})));
  page.append(cols);

  // 5. everything else
  const unknowns = [...co.unknowns, ...co.founders.flatMap(f => f.unknowns)];
  const leads = [...co.leads, ...co.founders.flatMap(f => f.leads)];
  const parts = [["Filler", co.filler], ["Other facts", co.other]];
  const total = co.filler.length + co.other.length + unknowns.length + leads.length + co.notes.length;
  const more = el("details", {class: "else"}, el("summary", {text: "Everything else (" + total + ")"}));
  for (const [title, list] of parts) if (list.length) more.append(el("h3", {text: title}), factList(list, ci, true));
  more.append(el("h3", {text: "Unknowns"}), unknownList(unknowns, ci));
  more.append(el("h3", {text: "Leads (not facts)"}));
  if (leads.length) more.append(factList(leads, ci, true)); else more.append(el("p", {class: "empty", text: "None."}));
  if (co.notes.length) { const ul = el("ul", {class: "plain"}); for (const n of co.notes) ul.append(el("li", {text: n})); more.append(el("h3", {text: "Assembly notes"}), ul); }
  page.append(more, el("footer", {class: "rev", text: "Ratings stay in this page's memory only and are lost on reload. Use “Copy counts as text” to keep them."}));
  return page;
}

function landing() {
  const page = el("section", {id: "co-start"});
  page.append(el("h2", {text: "Founder dossiers"}), el("p", {class: "sub", text: (DATA.label ? DATA.label + " · " : "") + DATA.companies.length + " companies. Pick one on the left, or below."}));
  const how = el("div", {class: "howto"}, el("h3", {text: "How to read this"}));
  how.append(el("p", {text: "Each page is a first-glance profile of a company and its founders, put together from public web pages. Every line is a fact that a page says. Click a line to see the exact quote and a link to the page."}));
  const dl = el("dl", {});
  const items = [
    [{label: "confirmed by YC", kind: "confirmed"}, "The organisation the fact is about states it itself: an accelerator, an investor, an employer or a university."],
    [{label: "corroborated", kind: "corroborated"}, "Two or more different kinds of source say the same, and none of them is the organisation itself."],
    [{label: "own claim", kind: "own"}, "Written by the founder: their bio, their own site or profile. Not independently checked."],
    [{label: "company site", kind: "site"}, "The company's own marketing text."],
    [{label: "media", kind: "media"}, "A news article or a third-party post. Media often repeats what the company said."],
    [{label: "database", kind: "database"}, "A startup database such as Tracxn or Crunchbase."],
    [{label: "LinkedIn snippet", kind: "snippet"}, "Only a search result's text was read; the page itself was not opened. The weakest tier. The same tier is called “X snippet”, “database snippet” (a startup database's result text) or “search snippet”."],
    [{label: "sources differ", kind: "differ"}, "Sources give different values. Both are shown; nothing is resolved."],
  ];
  for (const [p, t] of items) dl.append(el("dt", {}, pill(p)), el("dd", {text: t}));
  how.append(dl);
  how.append(el("p", {}, el("b", {text: "Identity badge. "}), "Each founder card says how well the person is tied to this company, in three levels. ",
    el("b", {text: "Confirmed by an independent source: "}), "a sourced identity or role claim (for example “Founder/CEO”) from an accelerator, database or media page on another domain than the company's. ",
    el("b", {text: "Named as founder by the company site and a snippet: "}), "the company's own site names the person as founder or in a role, and a search-result snippet from another domain says the same; no independent page was opened. ",
    el("b", {text: "Not confirmed: "}), "neither. Click the badge to see the claims behind it."));
  how.append(el("p", {}, el("b", {text: "Key facts. "}), "At extraction, the model tags each claim with at most one signal kind from a fixed list, or none: prior_exit (prior venture acquired or public), prior_venture_raised (money raised by a prior venture), quantified_traction (revenue, contracts, LOIs, pilots with a number), named_customer_or_partner, outside_investor (investor other than the accelerator), phd_or_publications, prior_role_same_field (prior role in the company's own field). Code picks 3-5 key facts per company in that priority order, preferring stronger status at equal priority, at most 2 per founder."));
  how.append(el("p", {}, el("b", {text: "Sourced does not mean true. "}), "It means a page says it and the quote is on that page. A source can be out of date or wrong, and a page can be about a namesake. Bold words are organisations, investors and amounts taken from structured fields."));
  how.append(el("p", {class: "sub", text: "Review mode (left, bottom) adds Right / Wrong buttons, counts, claim ids and run ids for checking the page."}));
  page.append(how);
  const cards = el("div", {class: "cards"});
  DATA.companies.forEach((co, ci) => cards.append(el("button", {type: "button", class: "card", onclick: () => show(ci)}, el("b", {text: co.name}),
    el("span", {text: (co.flags.length ? "Check: " + co.flags[0] + ". " : "") + (co.buildText || "No description found.")}))));
  page.append(cards);
  return page;
}

function count(ci) {
  let right = 0, wrong = 0; const wrongIds = [];
  for (const c of claimsOf[ci]) { if (ratings[c.key] === "right") right++; else if (ratings[c.key] === "wrong") { wrong++; wrongIds.push(c.id); } }
  return {right, wrong, total: claimsOf[ci].length, unrated: claimsOf[ci].length - right - wrong, wrongIds};
}
function updateCounts(ci) {
  const c = count(ci);
  document.getElementById("counts-" + ci).replaceChildren(el("span", {}, "Right ", el("b", {text: String(c.right)})),
    el("span", {}, "Wrong ", el("b", {text: String(c.wrong)})), el("span", {}, "Unrated ", el("b", {text: String(c.unrated)})),
    el("span", {class: "sub", text: "of " + c.total + " claims"}));
}
function countsText() {
  const lines = ["Ratings: Right / Wrong / Unrated of total claims"];
  DATA.companies.forEach((co, ci) => {
    const c = count(ci);
    lines.push(co.name + " (run " + co.runId + ", " + co.modeText + "): " + c.right + " / " + c.wrong + " / " + c.unrated + " of " + c.total);
    if (c.wrongIds.length) lines.push("  wrong: " + c.wrongIds.join(", "));
  });
  return lines.join("\n");
}
document.getElementById("copy").addEventListener("click", async () => {
  const text = countsText(), box = document.getElementById("copytext"), status = document.getElementById("copystatus");
  box.value = text; box.style.display = "block";
  try { await navigator.clipboard.writeText(text); status.textContent = "Copied."; }
  catch (e) { box.select(); status.textContent = document.execCommand && document.execCommand("copy") ? "Copied." : "Select the text below and copy it."; }
});
document.getElementById("review").addEventListener("change", e => {
  document.body.classList.toggle("review", e.target.checked);
  for (const d of factDetails) d.open = e.target.checked;
});
function show(ci) {      // ci = -1: the landing view
  document.getElementById("co-start").hidden = ci !== -1;
  DATA.companies.forEach((_, i) => { document.getElementById("co-" + i).hidden = i !== ci; document.getElementById("tab-" + i).setAttribute("aria-selected", String(i === ci)); });
  document.getElementById("tab-start").setAttribute("aria-selected", String(ci === -1));
  window.scrollTo(0, 0);
}
const tabs = document.getElementById("tabs"), main = document.getElementById("main");
tabs.append(el("button", {type: "button", class: "tab", id: "tab-start", role: "tab", "aria-selected": "true", onclick: () => show(-1), text: "How to read this"}));
main.append(landing());
DATA.companies.forEach((co, ci) => {
  main.append(companyPage(co, ci));
  tabs.append(el("button", {type: "button", class: "tab", id: "tab-" + ci, role: "tab", "aria-selected": "false", onclick: () => show(ci), text: co.name}));
  updateCounts(ci);
});
show(-1);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
