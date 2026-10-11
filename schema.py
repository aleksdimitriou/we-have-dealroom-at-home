"""
schema.py: single source of truth for the data model.

Principle: the agent PROPOSES claims. The application decides which claims are
admissible (verify.py), and the dossier is ASSEMBLED by code from sourced claims
only (assemble.py). The agent never writes the dossier. "Sourced" means a page says
this (the quote is on the page); it does not mean the claim is true.

Data model (one JSON file per founder run):
    actions   -> every search/fetch the agent made, in order
    sources   -> each fetched page (one source, many evidence items)
    evidence  -> a verbatim quote from one source
    claims    -> a value of a given claim_type, pointing to evidence, with a status and an origin
    unknowns  -> what the agent looked for and could not find
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

class ClaimType(str, Enum):
    """Every claim must fill a dossier field. There is deliberately no OTHER:
    a claim that fits nowhere is out of scope."""
    # founder
    IDENTITY = "identity"                    # this person is linked to this company
    ROLE = "role"                            # role at the current company
    PRIOR_ROLE = "prior_role"                # employer + title (+ dates if stated)
    PRIOR_VENTURE = "prior_venture"          # company founded before + outcome if stated
    EDUCATION = "education"                  # institution + field (+ year if stated)
    TECHNICAL = "technical"                  # GitHub, papers, patents
    PUBLIC_PRESENCE = "public_presence"      # talks, podcasts, writing
    EMAIL = "email"                          # a founder's work email, admitted only from a page on the company's own
                                             # domain and on that domain; only in people.csv, never on the page
    # company
    COMPANY_DESCRIPTION = "company_description"
    SECTOR = "sector"                        # must be one of SECTOR_TAGS
    BUSINESS_MODEL = "business_model"        # B2B / B2C / B2B2C
    LOCATION = "location"
    STAGE_SIGNAL = "stage_signal"            # live product, pricing page, named customers
    FUNDING = "funding"                      # announced rounds


FOUNDER_CLAIM_TYPES = {
    ClaimType.IDENTITY, ClaimType.ROLE, ClaimType.PRIOR_ROLE,
    ClaimType.PRIOR_VENTURE, ClaimType.EDUCATION, ClaimType.TECHNICAL,
    ClaimType.PUBLIC_PRESENCE, ClaimType.EMAIL,
}

# At most 2 per company (primary + secondary). If nothing fits, record an unknown (no "other").
SECTOR_TAGS: list[str] = [
    "ai_software", "dev_infra", "enterprise_software", "fintech", "health_bio",
    "climate_energy", "mobility_logistics", "aerospace_defense", "industrial_hardware",
    "consumer", "commerce_marketplaces", "education_work",
    "real_estate_construction", "food_agri",
]


class Attribute(str, Enum):
    """Which part of a conceptual fact a claim states. The agent picks from this fixed list;
    code groups claims into facts by subject / claim_type / attribute.
    Only conflict-prone claim types have attributes; for the others, different values are different facts."""
    # funding
    TOTAL_RAISED = "total_raised"
    ROUND_LABEL = "round_label"
    ROUND_DATE = "round_date"
    INVESTOR = "investor"
    # stage_signal
    FOUNDED = "founded"
    TEAM_SIZE = "team_size"
    PRODUCT_STATUS = "product_status"
    CUSTOMER_OR_PILOT = "customer_or_pilot"
    PUBLIC_LISTING = "public_listing"          # pricing page, waitlist, app store listing, job postings
    # location
    HQ = "hq"
    OPERATING_SITES = "operating_sites"
    # role
    TITLE = "title"
    START_DATE = "start_date"
    # education (grouped by degree level)
    PHD = "phd"
    MASTERS = "masters"
    BACHELORS = "bachelors"
    LEFT_WITHOUT_DEGREE = "left_without_degree"
    DEGREE_LEVEL_NOT_STATED = "degree_level_not_stated"   # a degree whose level the source does not give; set-valued
    # sector
    PRIMARY = "primary"
    SECONDARY = "secondary"


ATTRIBUTES_BY_TYPE: dict[ClaimType, list[Attribute]] = {
    ClaimType.FUNDING: [Attribute.TOTAL_RAISED, Attribute.ROUND_LABEL, Attribute.ROUND_DATE, Attribute.INVESTOR],
    ClaimType.STAGE_SIGNAL: [Attribute.FOUNDED, Attribute.TEAM_SIZE, Attribute.PRODUCT_STATUS,
                             Attribute.CUSTOMER_OR_PILOT, Attribute.PUBLIC_LISTING],
    ClaimType.LOCATION: [Attribute.HQ, Attribute.OPERATING_SITES],
    ClaimType.ROLE: [Attribute.TITLE, Attribute.START_DATE],
    ClaimType.EDUCATION: [Attribute.PHD, Attribute.MASTERS, Attribute.BACHELORS, Attribute.LEFT_WITHOUT_DEGREE,
                          Attribute.DEGREE_LEVEL_NOT_STATED],
    ClaimType.SECTOR: [Attribute.PRIMARY, Attribute.SECONDARY],
}

# Set-valued attributes merge across claims; all others are scalar and can disagree.
SET_VALUED_ATTRIBUTES = {
    Attribute.INVESTOR, Attribute.CUSTOMER_OR_PILOT, Attribute.PUBLIC_LISTING, Attribute.OPERATING_SITES,
    Attribute.PRODUCT_STATUS,          # separate observations ("demo bookable", "SOC 2 in progress") are separate facts
    Attribute.DEGREE_LEVEL_NOT_STATED, # each distinct degree is its own fact and can never make sources disagree
}


class SourceOrigin(str, Enum):
    """Who wrote the content, not where it is hosted. Set per claim by the agent."""
    FOUNDER = "founder"                # founder LinkedIn, personal site, X (LinkedIn + personal site = one origin)
    COMPANY = "company"                # company site, company LinkedIn, press release
    COUNTERPARTY = "counterparty"      # investor portfolio page, accelerator page, employer, university, programme
    DATABASE = "database"              # Tracxn, Crunchbase
    MEDIA = "media"                    # news, newsletters, listicles, third-party posts


class ConfirmationStatus(str, Enum):
    """Derived by assemble.py per fact, never by the agent."""
    CONFIRMED = "confirmed"                # at least one counterparty source
    CORROBORATED = "corroborated"          # two or more different origins, none counterparty
    SINGLE_SOURCE = "single_source"        # one origin only
    SOURCES_DISAGREE = "sources_disagree"  # incompatible values; overrides the above
    LEAD = "lead"                          # only rejected-as-lead claims; shown in the Leads section
    SNIPPET = "snippet"                    # only a search snippet was checked; lowest tier, never confirmed/corroborated


class ClaimStatus(str, Enum):
    PROPOSED = "proposed"   # transient: only exists between agent output and verify.py
    SOURCED = "sourced"     # the quote is on the fetched page; says nothing about truth
    SNIPPET = "snippet"     # the quote is in the stored search snippet of a page that was not fetched; a visible
                            # lowest tier: never confirmed or corroborated, never in the CRM's primary fields
    REJECTED = "rejected"


class RejectionReason(str, Enum):
    NO_SOURCE = "no_source"                     # claim has no URL or quote
    SOURCE_NOT_FETCHED = "source_not_fetched"   # URL never appears in the fetch log
    QUOTE_NOT_FOUND = "quote_not_found"         # quote not present in the fetched page text
    INVALID_SECTOR = "invalid_sector"           # sector claim whose value is not in SECTOR_TAGS
    FETCH_FAILED = "fetch_failed"               # fetch attempted, page could not be retrieved
    ROBOTS_BLOCKED = "robots_blocked"           # fetch blocked by robots.txt
    SNIPPET_ONLY = "snippet_only"               # LEGACY: before the snippet tier a snippet-only claim was rejected with this reason;
                                                # verify.py now uses it only internally and gives such claims the snippet
                                                # status. Kept so stored runs that carry it (20261007_182518) still load.
    INVALID_ATTRIBUTE = "invalid_attribute"     # attribute missing, or not one of ATTRIBUTES_BY_TYPE[claim_type]
    IDENTITY_UNANCHORED = "identity_unanchored" # founder claim from a page found by search that names neither the
                                                # company, a co-founder, nor an employer already sourced
    EXCLUDED_PERSONAL_DATA = "excluded_personal_data"   # a person's age (never recorded); or an email not admitted. Log only;
                                                        # verify.py removes the value and the quote
    EXCLUDED_SOURCE = "excluded_source"         # a contact-data broker page. Log only; value and quote removed


# Rejections shown to the analyst in the Leads section. The others stay in the log only.
# quote_not_found is shown with a different label from the mechanical failures.
LEAD_REASONS = {
    RejectionReason.FETCH_FAILED, RejectionReason.ROBOTS_BLOCKED,
    RejectionReason.SNIPPET_ONLY, RejectionReason.QUOTE_NOT_FOUND, RejectionReason.IDENTITY_UNANCHORED,
}


class SignalKind(str, Enum):
    """What makes a claim worth a place among a company's key facts. Set by the model at extraction (at most one per
    claim, or none); code picks the key facts from them in this priority order (src/keyfacts.py)."""
    PRIOR_EXIT = "prior_exit"                                  # a prior venture acquired or public
    PRIOR_VENTURE_RAISED = "prior_venture_raised"              # money raised by a prior venture
    QUANTIFIED_TRACTION = "quantified_traction"                # revenue, contracts, LOIs, pilots with a number
    NAMED_CUSTOMER_OR_PARTNER = "named_customer_or_partner"
    OUTSIDE_INVESTOR = "outside_investor"                      # an investor other than the accelerator
    PHD_OR_PUBLICATIONS = "phd_or_publications"
    PRIOR_ROLE_SAME_FIELD = "prior_role_same_field"            # a prior role in the company's own field


class ActionKind(str, Enum):
    SEARCH = "search"
    FETCH = "fetch"
    API = "api"             # a request to a structured-data API (OpenAlex); not a page fetch


class ActionOutcome(str, Enum):
    OK = "ok"
    FETCH_FAILED = "fetch_failed"
    ROBOTS_BLOCKED = "robots_blocked"


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------

class BatchRow(BaseModel):
    """One row of the input CSV in data/."""
    company_name: str
    company_url: str
    accelerator_profile_url: Optional[str] = None
    founder_names: list[str]


# --------------------------------------------------------------------------
# What the agent submits (this model's JSON schema IS the submit tool's input schema)
# --------------------------------------------------------------------------

class ProposedClaim(BaseModel):
    subject: str = Field(description="Founder full name, or 'company'.")
    claim_type: ClaimType
    attribute: Optional[Attribute] = Field(
        default=None,
        description="Which part of the fact this claim states. Required for funding, stage_signal, "
                    "location, role, education and sector; leave empty for other claim types.")
    value: str = Field(description="The fact, stated plainly, one fact per claim. Copy dates and titles as written.")
    value_normalized: Optional[dict[str, str]] = Field(
        default=None,
        description="Optional normalised form next to the exact value, e.g. {'amount': '750000', 'currency': 'USD'}.")
    source_date: Optional[str] = Field(
        default=None, description="The date of the source page or post, copied as written. Leave empty if none.")
    source_url: str = Field(description="URL of a page you FETCHED in this run. Search snippets do not count.")
    quote: str = Field(description="Verbatim passage from that page supporting the value. Copy exactly.")
    source_origin: SourceOrigin = Field(
        description="Who WROTE this content, not where it is hosted. On a YC page, YC's own data is counterparty; "
                    "what the founders wrote there is founder or company.")
    signal: Optional[SignalKind] = Field(
        default=None,
        description="At most one signal kind, only if the claim states it; leave empty otherwise.")


class Unknown(BaseModel):
    subject: str
    claim_type: ClaimType
    searched_for: str = Field(description="What you looked for, the pages you fetched, and the links you "
                                          "considered but did not follow (with why).")


class Submission(BaseModel):
    """Input schema of the submit_claims tool. Calling it ends the run."""
    claims: list[ProposedClaim]
    unknowns: list[Unknown]


# --------------------------------------------------------------------------
# What the application stores (built by agent.py and verify.py, never by the model)
# --------------------------------------------------------------------------

class Action(BaseModel):
    action_id: str
    kind: ActionKind
    query_or_url: str
    timestamp: datetime
    outcome: ActionOutcome = ActionOutcome.OK
    for_subject: Optional[str] = None  # founder name this action was for (None = company-level); lets evaluate.py spot starvation


class FoundVia(str, Enum):
    """How the agent came to fetch a page."""
    INPUT = "input"           # the company URL or accelerator URL from the input row
    LINK = "link"             # a link on a page already fetched
    SEARCH = "search"         # only a web search result (identity anchor applies)
    API = "api"               # a structured-data API response stored as the source text (OpenAlex works)


class Source(BaseModel):
    source_id: str
    url: str
    fetched_at: datetime
    fetched_by_action: str            # action_id: lets you trace sourced claims back to actions
    content_hash: str
    page_text_path: Optional[str] = None   # path under runs/ (gitignored)
    found_via: Optional[FoundVia] = None   # None in runs made before search results were used


class SearchResult(BaseModel):
    """One web search result, stored so verify.py can check a snippet-only lead against its snippet."""
    action_id: str                    # the search action that returned it
    rank: int
    url: str
    title: str
    snippet: str


class Evidence(BaseModel):
    evidence_id: str
    source_id: str
    quote: str


class Claim(BaseModel):
    claim_id: str
    subject: str
    claim_type: ClaimType
    attribute: Optional[Attribute] = None
    value: str
    value_normalized: Optional[dict[str, str]] = None
    source_date: Optional[str] = None
    source_origin: SourceOrigin
    origin_corrected_from: Optional[SourceOrigin] = None   # the agent's label, when verify.py corrected it (company-domain check)
    signal: Optional[SignalKind] = None    # the model's signal tag (key facts); code decides what to do with it
    source_url: Optional[str] = None       # as submitted by the agent, kept for rejected claims too (leads need it)
    quote: Optional[str] = None            # as submitted
    evidence_id: Optional[str] = None      # None when rejected before a source was matched (no_source, source_not_fetched, ...)
    status: ClaimStatus = ClaimStatus.PROPOSED
    rejection_reason: Optional[RejectionReason] = None


class Budget(BaseModel):
    # The workflow's caps live in src/agent.py (fixed fetch and search steps). What is recorded here is what the run
    # actually did, and the one hard limit: the cost guard.
    max_searches: int = 0                  # search cap for this company: per founder and per company, set by agent.py
    actions_used: int = 0                  # searches + fetches
    searches_used: int = 0
    pages_fetched: int = 0
    cost_limit_usd: float = 0.50           # the hard cost guard this run used (a parameter)


class ModelCall(BaseModel):
    """Usage and cost of one model call, logged for every call in a run."""
    provider: str                          # "anthropic" | "openrouter"
    model: str                             # the model that served the call (can differ after a fallback)
    input_tokens: int = 0                  # uncached input
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0                  # estimate from the price table in src/llm.py
    seconds: float = 0.0
    stop_reason: Optional[str] = None
    error: Optional[str] = None


class Run(BaseModel):
    """Everything about one company's research run. Saved to runs/<run_id>/run.json."""
    run_id: str
    input: BatchRow
    model: str
    provider: str = "anthropic"
    started_at: datetime
    finished_at: Optional[datetime] = None
    stop_reason: Optional[str] = None      # "submitted" | "cost_limit" | "error"
    budget: Budget = Field(default_factory=Budget)
    actions: list[Action] = Field(default_factory=list)
    search_result_urls: list[str] = Field(default_factory=list)   # URLs seen in search results; lets verify.py tell snippet_only from source_not_fetched
    search_results: list[SearchResult] = Field(default_factory=list)  # every result with its snippet text
    sources: list[Source] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    unknowns: list[Unknown] = Field(default_factory=list)
    calls: list[ModelCall] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)   # things worth knowing about the run that are not actions: pages dropped, caps reached, cache reuse
    input_tokens: int = 0                  # totals over calls (uncached + cache writes + cache reads)
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0


# --------------------------------------------------------------------------
# Assembled output (built by assemble.py from SOURCED claims only)
# --------------------------------------------------------------------------

class Fact(BaseModel):
    """One conceptual fact: all claims sharing a fact_key, plus the confirmation status derived from them."""
    fact_key: str                          # subject / claim_type / attribute
    subject: str
    claim_type: ClaimType
    attribute: Optional[Attribute] = None
    claims: list[Claim]
    confirmation: ConfirmationStatus


class FounderDossier(BaseModel):
    name: str
    identity_confirmed: bool               # sourced IDENTITY claim, origin counterparty/database/media, domain other than the company's
    background_self_reported_only: bool    # True if all background claims are founder-origin; shown as a badge
    facts: dict[ClaimType, list[Fact]]     # sourced claims only, grouped by field
    unknowns: list[Unknown]
    leads: list[Fact] = Field(default_factory=list)   # rejected claims in LEAD_REASONS; never in the main fields


class CompanyDossier(BaseModel):
    company_name: str
    company_url: str
    facts: dict[ClaimType, list[Fact]]
    founders: list[FounderDossier]
    unknowns: list[Unknown] = Field(default_factory=list)
    leads: list[Fact] = Field(default_factory=list)
