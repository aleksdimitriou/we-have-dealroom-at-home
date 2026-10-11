"""
verify.py: deterministic admissibility checks. No LLM, no network.

Every proposed claim comes out as a Claim with status `sourced` or `rejected` (with a reason).
Nothing is dropped. "Sourced" means the quote is on a page this run fetched or was given;
it says nothing about whether the claim is true.

Checks, in order (the first failure decides the rejection reason):
    1. source_url and quote present                     else no_source
    2. source_url in this run's fetch log               else snippet status / fetch_failed / robots_blocked /
                                                             quote_not_found / source_not_fetched
    3. quote on that page (after normalisation)         else quote_not_found
    4. sector value in SECTOR_TAGS                       else invalid_sector
    5. attribute matches the claim type                  else invalid_attribute
    6. a founder claim from a page found only by search names an anchor (company, co-founder, or an
       organisation already sourced for that founder)    else identity_unanchored
    Before step 2, two exclusions, log only, value and quote removed from the stored claim:
       a contact-data broker as the source              -> excluded_source
       a value or quote that states a person's age      -> excluded_personal_data (a person's age is never recorded)
    An email claim is admitted only from a page fetched on the company's own domain, with the address in the quote
    and on that domain; otherwise (another domain, a snippet, a company subject) excluded_personal_data.
    For a URL with a stored search result, step 2 checks the quote against the snippet first: if it is there the claim
    gets the snippet status (even when the page could not be fetched, e.g. robots.txt); if not, the failed fetch or
    quote_not_found decides.

Separately, for every claim: a source on the company's own domain can only be `company` or
`founder`. Any other origin is corrected to `company`, and the agent's label is kept in
`origin_corrected_from`. A claim whose quote matched a structured-data API response (OpenAlex) is corrected to
`database` the same way.
"""

from __future__ import annotations

import html
import re
import unicodedata
from urllib.parse import urlsplit

from schema import (
    ATTRIBUTES_BY_TYPE, SECTOR_TAGS,
    ActionKind, ActionOutcome, Claim, ClaimStatus, ClaimType, Evidence, ProposedClaim,
    Attribute, FoundVia, RejectionReason, Run, Source, SourceOrigin,
)
from src.orgnames import organisation_variants

# Contact-data brokers and people-search sites: never a source. agent.py drops their search results too.
CONTACT_BROKERS = (
    "rocketreach.co", "contactout.com", "zoominfo.com", "apollo.io", "lusha.com", "signalhire.com", "seamless.ai",
    "hunter.io", "kaspr.io", "snov.io", "adapt.io", "leadiq.com", "datanyze.com", "uplead.com", "cognism.com",
    "peopledatalabs.com", "aeroleads.com", "getprospect.com", "spokeo.com", "whitepages.com", "radaris.com",
)

# A person's age is never recorded. A founder claim is checked with all patterns; a company claim only where the age is tied to
# a person ("the 24-year-old founder"), so "a 5-year-old company" is not caught.
_AGE_ANY = re.compile(
    r"\baged\s+\d{1,2}\b|\b\d{1,2}[-\s]years?[-\s]old\b|\bage[d]?\s*:?\s*\d{1,2}\b"
    r"|\bborn\s+(?:in|on)?\s*(?:[A-Z][a-z]+\s+)?(?:\d{1,2},?\s+)?(?:19|20)\d{2}\b"
    r"|\b\d{1,2}\s+Jahre\s+alt\b|\bgeboren\b[^.]{0,30}\b(?:19|20)\d{2}\b|\bJahrgang\s+(?:19|20)\d{2}\b", re.I)
_AGE_OF_PERSON = re.compile(
    r"\baged\s+\d{1,2}\b|\b\d{1,2}[-\s]years?[-\s]old\s+(?:co-?)?(?:founder|ceo|cto|coo|entrepreneur|student|engineer)", re.I)
EMAIL_ADDRESS = re.compile(r"^[\w.+-]+@([\w-]+(?:\.[\w-]+)+)$")


def is_contact_broker(url: str) -> bool:
    return any(is_on_domain(url, d) for d in CONTACT_BROKERS)


def states_age(claim: ProposedClaim) -> bool:
    text = f"{claim.value} {claim.quote}"
    pattern = _AGE_OF_PERSON if claim.subject.strip().lower() == "company" else _AGE_ANY
    return bool(pattern.search(text))


def email_admissible(claim: ProposedClaim, page_url: str, company_domain: str) -> bool:
    """A founder's address, written on a page of the company's own domain, on that domain, and in the quote."""
    m = EMAIL_ADDRESS.match(claim.value.strip())
    return (bool(m) and claim.subject.strip().lower() != "company" and bool(company_domain)
            and is_on_domain(page_url, company_domain) and is_on_domain("https://" + m.group(1), company_domain)
            and claim.value.strip().lower() in claim.quote.lower())


# Curly and other typographic quotes -> straight quotes.
_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
})


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """The only transformations allowed before quote matching.
    Case is kept: matching is case-sensitive."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_QUOTE_MAP)
    return " ".join(text.split())


def normalise_url(url: str) -> str:
    """Ignore scheme, host case, a leading 'www.', a trailing '/' and the fragment. Keep the query."""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/")
    query = f"?{parts.query}" if parts.query else ""
    return f"{host}{path}{query}"


def domain_of(url: str) -> str:
    return urlsplit(url.strip()).netloc.lower().removeprefix("www.")


def registrable_domain(url: str) -> str:
    """Host without 'www.', reduced to the last two labels (three for co.uk-style endings): the site's own domain."""
    parts = domain_of(url).split(".")
    return ".".join(parts[-3:] if len(parts) >= 3 and parts[-2] in ("co", "com", "org") else parts[-2:])


def is_on_domain(url: str, domain: str) -> bool:
    """True for the domain itself and any subdomain of it (blog.example.com is on example.com)."""
    host = domain_of(url)
    return host == domain or host.endswith("." + domain)


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------

def _missing_source_reason(url: str, quote: str, run: Run) -> RejectionReason:
    """Why a cited URL has no fetched page in this run. If the quote is in the stored snippet of a search result for
    that URL, SNIPPET_ONLY: the claim gets the snippet status, whether the page was never fetched, was not chosen, or
    could not be fetched (robots.txt, an error). Otherwise a failed fetch, a quote missing from the snippet, or a URL
    seen nowhere."""
    key = normalise_url(url)
    results = [r for r in run.search_results if normalise_url(r.url) == key]
    q = normalise_text(quote)
    if any(q in normalise_text(r.snippet) for r in results):
        return RejectionReason.SNIPPET_ONLY
    failed_fetches = [
        a for a in run.actions
        if a.kind == ActionKind.FETCH and normalise_url(a.query_or_url) == key and a.outcome != ActionOutcome.OK
    ]
    if failed_fetches:
        last = failed_fetches[-1]
        if last.outcome == ActionOutcome.ROBOTS_BLOCKED:
            return RejectionReason.ROBOTS_BLOCKED
        return RejectionReason.FETCH_FAILED
    if results or key in {normalise_url(u) for u in run.search_result_urls}:
        return RejectionReason.QUOTE_NOT_FOUND
    return RejectionReason.SOURCE_NOT_FETCHED


_DEGREE_LEVELS = [
    (Attribute.PHD, re.compile(r"\b(ph\.?\s?d|doctor(ate)?|dphil)\b", re.I)),
    (Attribute.MASTERS, re.compile(r"\b(master'?s?|m\.?sc?|m\.?a|mba|m\.?eng|m\.?phil)\b", re.I)),
    (Attribute.BACHELORS, re.compile(r"\b(bachelor'?s?|b\.?sc?|b\.?a|b\.?eng|b\.?tech|undergraduate)\b", re.I)),
]
DEFAULT_DEGREE_LEVEL = Attribute.DEGREE_LEVEL_NOT_STATED


def default_degree_level(claim: ProposedClaim) -> Attribute:
    """The degree level an education claim without an attribute gets: read from its value and quote (PhD, master's,
    bachelor's), else "degree level not stated" (a set-valued attribute: it never makes sources disagree). LinkedIn snippets list education as "University, Degree Field" and the model often
    leaves the attribute out; that should not lose the fact."""
    text = f"{claim.value} {claim.quote}"
    return next((level for level, pattern in _DEGREE_LEVELS if pattern.search(text)), DEFAULT_DEGREE_LEVEL)


def _attribute_is_valid(claim: ProposedClaim) -> bool:
    allowed = ATTRIBUTES_BY_TYPE.get(claim.claim_type)
    if allowed is None:
        return claim.attribute is None
    return claim.attribute in allowed


def _corrected_origin(claim: ProposedClaim, company_domain: str) -> tuple[SourceOrigin, SourceOrigin | None]:
    """Returns (origin, origin_corrected_from)."""
    on_company_domain = bool(claim.source_url.strip()) and is_on_domain(claim.source_url, company_domain)
    if on_company_domain and claim.source_origin not in (SourceOrigin.COMPANY, SourceOrigin.FOUNDER):
        return SourceOrigin.COMPANY, claim.source_origin
    return claim.source_origin, None


def verify_claims(
    proposed: list[ProposedClaim],
    run: Run,
    page_texts: dict[str, str],
) -> tuple[list[Claim], list[Evidence]]:
    """Check every proposed claim against this run's log.

    page_texts maps source_id -> the page text stored for that source (a fetched page).
    Returns one Claim per proposed claim, in the same order, plus Evidence for each matched quote.
    """
    company_domain = domain_of(run.input.company_url)

    sources_by_url: dict[str, list[Source]] = {}
    for source in run.sources:
        sources_by_url.setdefault(normalise_url(source.url), []).append(source)

    claims: list[Claim] = []
    evidence: list[Evidence] = []
    from_search: list[tuple[Claim, Source]] = []          # founder claims sourced on pages found by search
    snippet_claims: list[Claim] = []                      # founder claims whose quote is only in a stored snippet

    for i, p in enumerate(proposed, start=1):
        origin, corrected_from = _corrected_origin(p, company_domain)
        claim = Claim(
            claim_id=f"c{i:03d}",
            subject=p.subject,
            claim_type=p.claim_type,
            attribute=p.attribute,
            value=p.value,
            value_normalized=p.value_normalized,
            source_date=p.source_date,
            source_origin=origin,
            origin_corrected_from=corrected_from,
            signal=p.signal,
            source_url=p.source_url,
            quote=p.quote,
        )
        claims.append(claim)

        # 1. source_url and quote present
        if not p.source_url.strip() or not p.quote.strip():
            _reject(claim, RejectionReason.NO_SOURCE)
            continue

        # exclusions: a contact-data broker, a person's age. Log only; the text is not kept.
        if is_contact_broker(p.source_url):
            _exclude(claim, RejectionReason.EXCLUDED_SOURCE)
            continue
        if states_age(p):
            _exclude(claim, RejectionReason.EXCLUDED_PERSONAL_DATA)
            continue

        # 2. source_url in this run's fetch log
        candidates = sources_by_url.get(normalise_url(p.source_url), [])
        if not candidates:
            reason = _missing_source_reason(p.source_url, p.quote, run)
            if p.claim_type == ClaimType.EMAIL:                 # an email is never taken from a snippet or a failed fetch
                _exclude(claim, RejectionReason.EXCLUDED_PERSONAL_DATA)
            elif reason == RejectionReason.SNIPPET_ONLY:
                claim.status = ClaimStatus.SNIPPET            # visible lowest tier; anchor-checked below if a founder claim
                if (p.claim_type == ClaimType.EDUCATION and p.attribute is None
                        and is_on_domain(p.source_url, "linkedin.com")):
                    p = p.model_copy(update={"attribute": default_degree_level(p)})     # a LinkedIn snippet without a level
                    claim.attribute = p.attribute
                if p.claim_type == ClaimType.SECTOR and p.value.strip() not in SECTOR_TAGS:
                    _reject(claim, RejectionReason.INVALID_SECTOR)
                elif not _attribute_is_valid(p):
                    _reject(claim, RejectionReason.INVALID_ATTRIBUTE)
                elif not same_name(p.subject, "company"):
                    snippet_claims.append(claim)
            else:
                _reject(claim, reason)
            continue

        # 3. quote on the page; any fetch of the same URL may hold it
        matched = None
        for source in candidates:
            if normalise_text(p.quote) in normalise_text(page_texts.get(source.source_id, "")):
                matched = source
                break
        if matched is None:
            _reject(claim, RejectionReason.QUOTE_NOT_FOUND)
            continue

        if p.claim_type == ClaimType.EMAIL and not email_admissible(p, matched.url, company_domain):
            _exclude(claim, RejectionReason.EXCLUDED_PERSONAL_DATA)
            continue

        if matched.found_via == FoundVia.API and claim.source_origin != SourceOrigin.DATABASE:
            claim.origin_corrected_from = claim.origin_corrected_from or p.source_origin     # a structured-data API
            claim.source_origin = SourceOrigin.DATABASE

        ev = Evidence(evidence_id=f"e{len(evidence) + 1:03d}", source_id=matched.source_id, quote=p.quote)
        evidence.append(ev)
        claim.evidence_id = ev.evidence_id

        # 4. sector value in SECTOR_TAGS
        if p.claim_type == ClaimType.SECTOR and p.value.strip() not in SECTOR_TAGS:
            _reject(claim, RejectionReason.INVALID_SECTOR)
            continue

        # 5. attribute matches the claim type
        if not _attribute_is_valid(p):
            _reject(claim, RejectionReason.INVALID_ATTRIBUTE)
            continue

        claim.status = ClaimStatus.SOURCED
        if matched.found_via == FoundVia.SEARCH and not same_name(p.subject, "company"):
            from_search.append((claim, matched))

    # 6. identity anchor, checked once every claim has a status, so anchors can come from any claim
    _check_identity_anchors(from_search, claims, evidence, run, page_texts)
    _check_snippet_anchors(snippet_claims, claims, evidence, run)
    return claims, evidence


def _anchor_names(claim: Claim, claims: list[Claim], evidence: list[Evidence], run: Run) -> list[str]:
    """What a search page (or snippet) may name to anchor a founder claim: the company, a co-founder, or an
    organisation already sourced for that founder from a page not found by search."""
    search_sources = {s.source_id for s in run.sources if s.found_via == FoundVia.SEARCH}
    evidence_source = {e.evidence_id: e.source_id for e in evidence}
    anchors = [run.input.company_name]
    anchors += [f for f in run.input.founder_names if not same_name(f, claim.subject)]
    for other in claims:
        org = (other.value_normalized or {}).get("organization")
        if (org and other.status == ClaimStatus.SOURCED and same_name(other.subject, claim.subject)
                and evidence_source.get(other.evidence_id) not in search_sources):
            anchors.append(org)
    return anchors


def _names_any(text: str, anchors: list[str]) -> bool:
    """True if the text names any anchor, in any of its usual spellings ("UC Berkeley" also finds "University of
    California, Berkeley"). Whole words, case ignored."""
    page = normalise_text(text).lower()
    return any(re.search(rf"(?<!\w){re.escape(normalise_text(v).lower())}(?!\w)", page)
               for a in anchors if a.strip() for v in organisation_variants(a))


def _check_snippet_anchors(snippet_claims: list[Claim], claims: list[Claim], evidence: list[Evidence], run: Run) -> None:
    """The same identity anchor for snippet-status founder claims, applied to the snippet text: a snippet that
    names no anchor may be about a namesake, so the claim is rejected (identity_unanchored) instead."""
    for claim in snippet_claims:
        key = normalise_url(claim.source_url or "")
        text = " ".join(r.snippet + " " + r.title for r in run.search_results if normalise_url(r.url) == key)
        if not _names_any(text, _anchor_names(claim, claims, evidence, run)):
            _reject(claim, RejectionReason.IDENTITY_UNANCHORED)


def same_name(a: str, b: str) -> bool:
    return " ".join(a.lower().split()) == " ".join(b.lower().split())


def _check_identity_anchors(from_search: list[tuple[Claim, Source]], claims: list[Claim], evidence: list[Evidence],
                            run: Run, page_texts: dict[str, str]) -> None:
    """A founder claim from a page found only by search stays sourced if that page names an anchor (the company,
    a co-founder, or an organisation already sourced for that founder from a page not found by search; the
    organisation comes from value_normalized["organization"] on prior_role, prior_venture and education claims).
    Otherwise the page may be about a namesake: the claim becomes a lead, identity_unanchored. Matching ignores
    case and needs whole words, so "cereal" does not match "cereals"."""
    for claim, source in from_search:
        if not _names_any(page_texts.get(source.source_id, ""), _anchor_names(claim, claims, evidence, run)):
            _reject(claim, RejectionReason.IDENTITY_UNANCHORED)


def _exclude(claim: Claim, reason: RejectionReason) -> None:
    """Reject and drop the text: excluded personal data and broker pages are not stored, only the URL and the reason."""
    _reject(claim, reason)
    claim.value, claim.quote, claim.value_normalized = "[removed]", "[removed]", None


def _reject(claim: Claim, reason: RejectionReason) -> None:
    claim.status = ClaimStatus.REJECTED
    claim.rejection_reason = reason
