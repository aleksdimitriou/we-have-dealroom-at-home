"""
agent.py: the fixed research workflow for one company. Code does the research; the model only reads.

    1. Code fetches the anchors: the company site, the accelerator page and one about/team page.
    2. Code searches "<founder>" "<company>" for each founder.
    3. First reading (one model call): the model reads the anchors and those snippets and submits claims, unknowns,
       the organisations named for each founder, and up to 2 links per founder chosen from the link lists.
    4. Code fetches the chosen links, then searches "<founder>" "<organisation>" (up to 3 per founder) and the
       company (funding and investors), filters the results on title and snippet, fetches the best ones (a cap per
       founder and per company) and drops every page that names neither the company nor a co-founder. A separate snippet
       budget searches each database by the company's domain (its name as a fallback) and LinkedIn for each founder; those are snippets, and a database page is fetched only where robots.txt allows.
    5. Second reading (one model call): the model reads the new pages and snippets and adds only new claims.
    6. Code drops exact duplicates and runs the claims through verify.py.

The model has no tools to browse with: it cannot fetch, search or choose a URL that was not shown to it. Code
enforces the caps below and the cost guard (default $0.50 per run), and logs every search, fetch and model call.

Usage (from the project root, venv active):
    python -m src.agent "Example Robotics" [--provider anthropic|openrouter] [--model ...] [--cost-limit 0.5]
        [--input batch.csv] [--discover-founders] [--fresh]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, ValidationError

from schema import (
    ATTRIBUTES_BY_TYPE, LEAD_REASONS, SECTOR_TAGS, SET_VALUED_ATTRIBUTES,
    Action, ActionKind, ActionOutcome, BatchRow, Budget, ClaimStatus, ClaimType, FoundVia, ModelCall, ProposedClaim, Run,
    SearchResult, Source, Unknown,
)
from src.assemble import BACKGROUND_TYPES, LEAD_LABELS
from src.cache import PageCache, SearchCache, cached_search
from src.fetch import FetchResult, fetch_page, fetch_sitemap_urls, robots_allows
from src.llm import ChatClient, ToolCall, ToolResult, ToolSpec, Turn, inline_refs, make_client
from src.openalex import OpenAlexResult, allowed_fields, find_works
from src.orgnames import organisation_variants
from src.search import CREDIT_PRICE_USD, SearchHit, SearchResponse, Site, search as web_search
from src.verify import domain_of, is_contact_broker, is_on_domain, normalise_text, normalise_url, registrable_domain, verify_claims

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Caps of the workflow. The cost guard is the only hard limit; these keep a run bounded and cheap.
COST_LIMIT_USD = 0.50
SOFT_COST_SHARE = 0.8          # past 80% of the cost limit, code stops searching and fetching and goes on reading
SEARCHES_PER_FOUNDER = 4       # "<name>" "<company>" plus up to 3 organisation queries
ENTITY_QUERIES_PER_FOUNDER = SEARCHES_PER_FOUNDER - 1
SEARCHES_PER_COMPANY = 1       # funding and investors (no news search; the database searches are the snippet budget)
MAX_SEARCHES = 16
# The snippet budget, counted apart from the caps above: one search per database by the company's domain (the
# company's name when the domain finds nothing on that site) and one linkedin.com search per founder (the
# company's own LinkedIn page is not searched). Results are snippets; LinkedIn is never fetched; a database page only where robots.txt allows.
FUNDING_SITES = ["pitchbook.com", "crunchbase.com", "dealroom.co", "tracxn.com"]
LINKEDIN = "linkedin.com"
RESULT_FETCHES_PER_FOUNDER = 2
RESULT_FETCHES_PER_COMPANY = 2
LINK_PICKS_PER_FOUNDER = 2
LINK_PICKS_PER_COMPANY = 2
ABOUT_PAGES = 1
SITE_PAGES_CAP = 4              # extra anchor pages from the company's own site (the about link, then the sitemap or common paths)
FALLBACK_TRIES = 5            # common paths tried when there is no usable sitemap
FETCH_ATTEMPTS_FACTOR = 2      # fetch attempts per subject, as a multiple of its cap (failed and dropped pages count)
MODEL_PAGE_CHARS = 20_000      # page text shown to the model (the full text is stored for verify.py)
MODEL_LINKS = 80               # links shown to the model per anchor page
MAX_NUDGES = 2                 # replies without a valid submission, per reading, before giving up
ASSUMED_FOUNDERS = 2           # search cap basis when founder names are not given (--discover-founders)
MAX_DISCOVERED_FOUNDERS = 4
PROMPT_TOKENS_ESTIMATE = 8_000  # system prompt and tool schema, for the cost guard before the first call
CHARS_PER_TOKEN = 3.5
_LOCALE = r"(?:[a-z]{2}(?:[-_][A-Za-z]{2})?/)?"            # an optional language prefix: /en/, /de-DE/, /en-GB/
# The pages worth reading on a company's own site, in priority order (one per kind); the prefix may be any locale.
PAGE_KINDS = [
    ("team", re.compile(rf"^/{_LOCALE}(team|our-team|the-team|founders?|people|leadership|management|who-we-are)(/|$)", re.I)),
    ("about", re.compile(rf"^/{_LOCALE}(about|about-us|company|our-story|story|ueber-uns|über-uns|wir)(/|$)", re.I)),
    ("press", re.compile(rf"^/{_LOCALE}(press|presse|newsroom|media|news)(/|$)", re.I)),
    ("imprint", re.compile(rf"^/{_LOCALE}(imprint|impressum|legal-notice|legal)(/|$)", re.I)),
]
COMMON_PATHS = ["/team", "/about", "/about-us", "/company", "/press", "/impressum", "/imprint"]
ABOUT_PATH = re.compile(r"/(about|about-us|team|our-team|founders?|people|company|our-story|who-we-are)(/|$)", re.I)


# --------------------------------------------------------------------------
# What the model submits (the workflow's own tool input; the stored data model is in schema.py)
# --------------------------------------------------------------------------

class EntityMention(BaseModel):
    subject: str = Field(description="The founder's full name exactly as given in the task.")
    name: str = Field(description="An organisation named for this founder, as the page writes it.")
    kind: Literal["venture", "employer", "university", "other"]


class LinkPick(BaseModel):
    url: str = Field(description="A URL copied from a link list shown with an anchor page.")
    for_subject: str = Field(description="The founder's full name this page is about, or 'company'.")


class Extraction(BaseModel):
    """Input schema of the submit_claims tool."""
    claims: list[ProposedClaim]
    unknowns: list[Unknown]
    entities: list[EntityMention] = Field(
        default_factory=list, description="First reading only: organisations named for each founder, most useful first.")
    follow_links: list[LinkPick] = Field(
        default_factory=list, description="First reading only: up to 2 links per founder, from the link lists shown.")


# --------------------------------------------------------------------------
# Prompt and tool (constant for the whole run, so the cached prefix stays valid)
# --------------------------------------------------------------------------

def _attribute_lines() -> str:
    lines = []
    for claim_type, attributes in ATTRIBUTES_BY_TYPE.items():
        names = ", ".join(a.value + (" (list: one claim per item)" if a in SET_VALUED_ATTRIBUTES else "")
                          for a in attributes)
        lines.append(f"  - {claim_type.value}: {names}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You read pages and search results about early-stage startups and their founders for a venture fund's
first-glance screen, and submit what they say as claims. You do not search, fetch, judge, score or summarise: code
has already collected everything you are shown.

Every claim you submit is checked by code:
- its source_url must be a page shown to you under "Pages", or the URL of a search result shown to you,
- its quote must appear word for word on that page (copy it exactly from the text shown; do not fix spelling,
  dates, titles or capitalisation) or, for a search result whose page was not fetched, in that result's snippet,
- otherwise the claim is rejected. A claim backed only by your own knowledge is rejected.

## Scope: first-glance profile, not due diligence
Per founder, about 10 facts: identity (a page that links the person to the company), role at the company,
prior ventures with their headline outcome, the 1-2 most relevant prior roles, highest or most notable
education, one line of technical evidence (GitHub, papers, patents), notable recognition or public presence.
No minor jobs, courses, schools, mentor roles or personal locations.
Per company, about 10 facts: what it does, sector, how it sells (business model), location, 2-3 stage
signals, funding and investors.
Never record age, nationality, family, personal social media or political views.
A founder's work email only as an email claim, and only when a page on the company's own website shows the address
(team page, imprint, contact line): copy it exactly; never construct, guess or complete one, and never take one from a
search result or another website.
If sources disagree, submit both claims; do not resolve the conflict.

## Claims
- One fact per claim: split every bundle. "$800K pre-seed" is two claims (total_raised, round_label).
  "Founded Heimdal (YC S21), raised $12M+" is three prior_venture claims: founded Heimdal, Heimdal was
  YC S21, Heimdal raised $12M+. A credential is technical ("Chartered engineer") and the scale is its own
  prior_role claim ("led $100M+ capital projects"). One employer, one degree, one award per claim.
- identity: the line on a page that names the person next to their title at this company, copied as one
  claim (value: the name and title as written, e.g. "<Name>, Founder/CEO"). Use the accelerator's own founder
  listing when the founder appears there (origin counterparty); fall back to the company's page or a
  founder-written page only if it does not list them. role: the person's title at this company as the page
  words it (attribute title), and its start date as its own claim (attribute start_date). When one line gives
  both name and title, submit the identity claim and the role claim from it.
- Titles, dates and degrees: for prior_role, prior_venture and education claims, copy the title, the dates and the
  degree with its field exactly as the source words them: "Founder & CEO, Acme (Feb 2023 - May 2026)", "Master of
  Science, Mechanical Engineering". Never cut them down to the bare organisation name or to an abbreviation. If
  the source gives no title or no dates, state only what it gives.
- A statement that a founder left, dropped out of or paused a programme is an education claim with attribute
  left_without_degree.
- company_description: when a page gives both a short tagline and a longer description, submit each as its own claim.
  Where the sources state them, also submit the product (what it is and does), the customer (who it is for) and the
  problem it solves as separate company_description claims, each with its own quote.
- Lists: when a sentence offers several things (separated by ";", "or" or "and": products, engagement types,
  customers, sales motions), submit one claim per thing. A business_model claim is one sales motion.
- Whenever you record a company_description, also record at least a primary sector tag.
- subject: the founder's full name exactly as given in the task, or "company".
- claim_type and attribute: attribute is required for these types and must come from this list
  (empty for every other type):
{_attribute_lines()}
- sector: the value must be one tag from this list, at most 2 per company (attribute primary, then
  secondary). Tag the customer's problem domain first; add ai_software as secondary only if AI is the
  core product. If nothing fits, record an unknown.
  Tags: {", ".join(SECTOR_TAGS)}
- stage_signal: observable facts only, quoted: founding date or year, team size, product status as the
  source words it, a named customer or case study or stated pilot, pricing page, live sign-up or
  waitlist, app store listing, job postings. Never judgements like "strong traction".
  An accelerator batch or an "Active" status is not a stage signal. Attributes: product_status is a
  statement of the product's state as the source words it ("coming soon", "in prototyping", "demo
  bookable", "SOC 2 in progress"), one claim per observation; job postings, a pricing page, a live
  waitlist or sign-up and an app store listing are public_listing, one claim each; a named customer or
  stated pilot is customer_or_pilot.
- funding: an accelerator's batch listing of the company (e.g. "Summer 2026" on a YC page, a company card
  in an accelerator's portfolio) is a funding claim with attribute investor, value the accelerator's name,
  origin counterparty.
- value: the fact stated plainly; copy dates and titles as written. value_normalized (optional): e.g.
  {{"amount": "750000", "currency": "USD"}}. source_date: the page's or post's own date if shown, as written.
- For prior_role, prior_venture and education claims, put the organisation's name in value_normalized as
  {{"organization": "<name as written on the page>"}}.
- Founder bios: claim every distinct first-glance fact in each founder's bio on a page shown to you: each
  employer or role, degree, fellowship or award, and military service, one claim each (prior_role,
  education, public_presence). Stay within about 10 facts per founder: skip minor jobs, short courses and
  personal details. Do not leave a fact in a bio you were shown unclaimed.
- source_origin is who WROTE the content, not where it is hosted:
  founder (written by the founder: their bio, personal site, LinkedIn, X, quotes from them),
  company (company marketing text: product descriptions, taglines, company site and company LinkedIn,
  press release), counterparty (investor portfolio page, accelerator's own data, employer, university,
  programme), database (Tracxn, Crunchbase), media (news, newsletters, listicles, third-party posts).
  On a YC company page, YC's own data (batch, status, team size, founder list, location, partner, tags) is
  counterparty; a founder's bio there is founder; the company description and product text is company.

## Signals
Set `signal` on a claim only when that claim itself states the signal (at most one per claim; otherwise leave it
empty). Code uses the tags to pick a company's key facts, so tag only what clearly fits and never filler:
- prior_exit: a founder's prior venture was acquired or went public;
- prior_venture_raised: a founder's prior venture raised money (an amount or a round);
- quantified_traction: revenue, contracts, letters of intent or pilots with a number (an amount, a count, a size);
- named_customer_or_partner: a named customer, partner or deployment;
- outside_investor: an investor other than the accelerator (a funding claim with attribute investor);
- phd_or_publications: a PhD (an education claim) or a published paper (a technical claim);
- prior_role_same_field: a prior role at an employer in the company's own field.

## Search results
- A search result is a URL, a title and a short snippet. If its page was fetched, it appears under "Pages" and you
  cite the page. If it was not fetched (LinkedIn and X are never fetched), you may still submit a fact from its
  snippet: cite the result URL and copy the quote exactly from the snippet. Such a claim is recorded with a snippet
  status, below facts from fetched pages. From X, use only a bio line's professional content (title, company);
  nothing from posts.
- A result from a startup database (Tracxn, Crunchbase, PitchBook, Dealroom) often carries funding facts in its
  snippet: submit them with origin database, citing the result URL and quoting the snippet exactly. If the page was
  also fetched, cite the page.
- A database result may describe another company with the same name: use it only if its snippet or page names the
  company's website (the domain in the task) or a founder.
- A search result may be about a namesake. Code drops fetched pages that do not name the person together with the
  company, a co-founder or an organisation already sourced for that person on an anchor page. For a snippet, quote only facts that sit next to the company, a co-founder or an organisation
  already shown on a page for that founder.

## The two readings
First reading: you are shown the anchor pages (company site, accelerator page, an about or team page) and the
snippets of a search for each founder. Submit_claims takes:
- claims and unknowns, as above (unknown: what is missing from what you were shown, said briefly);
- entities: for each founder, up to 5 organisations named for them in the pages or snippets, most useful first:
  companies they founded, then employers, then universities. Code searches the first three;
- follow_links: up to 2 links per founder (for_subject: the founder's name), copied from the link lists shown
  under the anchor pages, to a page about that founder (their own site, GitHub, papers, a bio on a partner's
  site); and up to 2 for the company (for_subject: "company"), to a page that states facts about it (legal notice or
  imprint, press, customers, pricing). Never LinkedIn or X. Empty if none.
Second reading: you are shown the new pages and snippets that code collected. A page titled "OpenAlex works for
<name>" is a structured list of that founder's publications. For each
work submit one technical claim about that founder: value the title, year and venue as given, quote the title exactly
as it is written in the text, origin database, signal phd_or_publications. Submit only claims that are not
already submitted (the same fact with the same quote counts as submitted), and the complete list of unknowns
that remain. Leave entities and follow_links empty.
"""

SUBMIT_TOOL = ToolSpec(
    name="submit_claims",
    description=("Submit claims, unknowns and (first reading only) entities and follow_links. Call it once per "
                 "reading, after you have read everything shown."),
    input_schema=inline_refs(Extraction.model_json_schema()),
)

UNFETCHABLE_SOCIAL = ("linkedin.com", "x.com", "twitter.com")
NEVER_FETCHED = UNFETCHABLE_SOCIAL + ("youtube.com", "youtu.be", "vimeo.com", "tiktok.com", "instagram.com",
                                      "facebook.com")


def is_unfetchable_social(url: str) -> bool:
    """LinkedIn and X: recorded as search results, never fetched."""
    return any(is_on_domain(url, d) for d in UNFETCHABLE_SOCIAL)


def skip_reason(url: str) -> Optional[str]:
    """Why code never fetches this URL: social profiles, videos, PDFs."""
    if is_unfetchable_social(url):
        return "LinkedIn/X"
    if is_contact_broker(url):
        return "contact-data broker"
    if any(is_on_domain(url, d) for d in NEVER_FETCHED):
        return "video or social"
    path = urlsplit(url).path.lower()
    if path.endswith(".pdf") or "/pdf/" in path:
        return "PDF"
    return None


def _whole_word(text: str, phrase: str) -> bool:
    phrase = normalise_text(phrase).lower()
    return bool(phrase) and bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalise_text(text).lower()))


def names_person(text: str, name: str) -> bool:
    """The full name as whole words, or the first and last word with at most one word between (a middle name or
    initial; a nickname in the input does not matter). Not any two words that happen to be on the page."""
    words = normalise_text(name).lower().split()
    if len(words) < 2:
        return _whole_word(text, name)
    pattern = rf"(?<!\w){re.escape(words[0])}\W+(?:\w+\W+)?{re.escape(words[-1])}(?!\w)"
    return _whole_word(text, name) or bool(re.search(pattern, normalise_text(text).lower()))


def pass_prefilter(hit: SearchHit, subject: Optional[str], company: str, domain: str = "") -> bool:
    """Before fetching: the title and snippet name the founder (founder query) or the company or its website's domain
    (company query)."""
    text = f"{hit.title} {hit.snippet}"
    if subject:
        return names_person(text, subject)
    return _whole_word(text, company) or bool(domain) and domain.lower() in text.lower()


def page_is_anchored(text: str, subject: Optional[str], row: BatchRow, founders: list[str],
                     organisations: tuple[str, ...] = ()) -> bool:
    """After fetching a page found by search. A founder's page names the founder and the company, a co-founder, or an
    organisation already sourced for that founder on an anchor page; a company page names the company and a founder
    or the company's own address. Otherwise it may be about a namesake and code drops it before the model sees it."""
    others = [f for f in founders if f != subject]
    if subject:
        return names_person(text, subject) and (_whole_word(text, row.company_name)
                                                or any(names_person(text, f) for f in others)
                                                or any(_whole_word(text, v) for o in organisations for v in organisation_variants(o)))
    host = urlsplit(row.company_url).netloc.lower().removeprefix("www.")
    return _whole_word(text, row.company_name) and (any(names_person(text, f) for f in founders)
                                                    or bool(host) and host in text.lower())


def page_kind(url: str) -> Optional[str]:
    path = urlsplit(url).path or "/"
    return next((kind for kind, pattern in PAGE_KINDS if pattern.search(path)), None)


def pick_site_pages(urls: list[str], site: str, skip_kinds: set[str], skip_urls: set[str], limit: int) -> list[str]:
    """From a sitemap's URLs: on the site's own domain, at most one page per kind (team, about, press, imprint) in that
    priority order, the shortest path first, never a kind already read or a URL already tried; at most `limit`."""
    own = [u for u in urls if registrable_domain(u) == registrable_domain(site) and normalise_url(u) not in skip_urls]
    picked: list[str] = []
    for kind, pattern in PAGE_KINDS:
        if kind in skip_kinds or len(picked) >= limit:
            continue
        matches = sorted((u for u in own if pattern.search(urlsplit(u).path or "/")), key=lambda u: (len(urlsplit(u).path), u))
        if matches:
            picked.append(matches[0])
    return picked


def task_header(row: BatchRow, raw_founders: list[str]) -> str:
    if row.founder_names:
        founders = "Founders (use these exact names as subject):\n" + "\n".join(
            f"- {name}" + (f" (input note: {raw})" if raw != name else "")
            for name, raw in zip(row.founder_names, raw_founders))
    else:                                   # founder discovery: the input gives no names
        founders = ("Founders: not given in the input. Identify the company's founders from the pages shown. "
                    "For each founder you find, use their full name exactly as the page writes it as subject, and "
                    "record an identity claim and a role claim linking them to the company; only people with such a "
                    "claim are treated as founders.")
    return (f"Company: {row.company_name}\n"
            f"Company URL: {row.company_url}\n"
            f"Accelerator profile URL: {row.accelerator_profile_url or 'none'}\n"
            f"{founders}")


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

class Query:
    """One search code ran: who it was for, what came back."""
    def __init__(self, text: str, subject: Optional[str], kind: str, action_id: str, hits: list[SearchHit]):
        self.text, self.subject, self.kind, self.action_id, self.hits = text, subject, kind, action_id, hits


class Researcher:
    """Holds one run's state; `research()` is the workflow."""

    def __init__(self, row: BatchRow, raw_founders: list[str], client: ChatClient, run_dir: Path,
                 fetch: Callable[[str], FetchResult] = fetch_page, cost_limit: float = COST_LIMIT_USD,
                 cache: Optional[PageCache] = None,
                 search: Callable[[str, Site], SearchResponse] = web_search,
                 openalex: Callable[..., OpenAlexResult] = find_works,
                 robots: Callable[[str], bool] = robots_allows,
                 sitemap: Optional[Callable[[str], list[str]]] = fetch_sitemap_urls):
        self.row, self.client, self.fetch, self.cost_limit = row, client, fetch, cost_limit
        self.search, self.openalex, self.robots = search, openalex, robots
        self.sitemap = sitemap                     # None: no sitemap and no common paths (tests)
        self.robots_by_host: dict[str, bool] = {}
        self.cache = cache
        self.run_dir = run_dir
        self.pages_dir = run_dir / "pages"
        self.raw_founders = raw_founders
        self.founders = list(row.founder_names)
        self.discovery = not row.founder_names     # founder discovery: the input gave no names
        self.run = Run(run_id=run_dir.name, input=row, model=client.model, provider=client.provider,
                       started_at=datetime.now(),
                       budget=Budget(cost_limit_usd=cost_limit))
        self.run.budget.max_searches = self._budget_total()
        self.page_texts: dict[str, str] = {}
        self.page_meta: dict[str, dict] = {}       # source_id -> url, title, links
        self.fetched_ok: dict[str, str] = {}       # normalised URL -> source_id
        self.attempted: set[str] = set()           # normalised URLs fetched or tried, so none is fetched twice
        self.shown_links: dict[str, str] = {}      # normalised URL -> URL, for every link shown to the model
        self._open_calls: list[str] = []           # tool call ids of the last reading, to answer
        self.queries: list[Query] = []
        self.searches_by_subject: Counter = Counter()
        self.main_searches = 0                     # searches outside the snippet budget
        self.snippet_searches = 0                  # searches of the snippet budget
        self.shown_snippets: set[str] = set()      # normalised result URLs whose snippet the model has seen
        self.anchor_organisations: dict[str, tuple[str, ...]] = {}   # founder -> organisations sourced on anchor pages
        self.sourced_organisations: dict[str, tuple[str, ...]] = {}  # founder -> organisations sourced on any page
        self.sourced_texts: dict[str, list[str]] = {}               # founder -> values of sourced background claims
        self.sector_tags: list[str] = []                            # the company's sourced sector tags

    @staticmethod
    def _search_cap(n_founders: int, discovery: bool = False) -> int:
        """Searches allowed per run outside the snippet budget; founder discovery adds one company search ("Company" founders)."""
        return min(MAX_SEARCHES, SEARCHES_PER_FOUNDER * n_founders + SEARCHES_PER_COMPANY + (1 if discovery else 0))

    @staticmethod
    def _snippet_cap(n_founders: int) -> int:
        """The snippet budget: a name fallback per database at most, one LinkedIn search per founder."""
        return 2 * len(FUNDING_SITES) + n_founders

    def _budget_total(self) -> int:
        n = len(self.founders) or ASSUMED_FOUNDERS
        return self._search_cap(n, self.discovery) + self._snippet_cap(n)

    # ---- workflow ----

    def research(self) -> Run:
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        anchors = self._fetch_anchors()
        for name in self.founders:
            self._search(f'"{name}" "{self.row.company_name}"', name, "name", qtype="name+company")
        if self.discovery:                           # no names yet: a search for who the founders are, seen by reading 1
            self._search(f'"{self.row.company_name}" founders', None, "company", qtype="company founders")

        first_message = self._first_message(anchors)
        (self.run_dir / "reading_1.txt").write_text(first_message, encoding="utf-8")
        if self._projected_cost(len(first_message), None) > self.cost_limit:
            return self._finish("cost_limit", claims=[])
        self.client.add_user_text(first_message)
        first, error = self._read()
        if first is None:
            return self._finish("error", error, claims=[])
        (self.run_dir / "submission_reading_1.json").write_text(
            first.model_dump_json(indent=2, exclude={"entities", "follow_links"}))
        self.run.events.append(f"reading 1: {len(first.claims)} claims, {len(first.entities)} organisations, "
                               f"{len(first.follow_links)} links chosen")
        if not self.founders:
            self._adopt_discovered_founders(first)
        self.anchor_organisations, self.sourced_organisations = self._sourced_organisations(first.claims)

        before = set(self.page_texts)
        self._fetch_picked_links(first.follow_links)
        self._run_openalex()
        self._run_searches(first.entities)
        self._fetch_results()
        new_sources = [s for s in self.page_texts if s not in before]
        new_snippets = self._unseen_snippets()

        claims, unknowns = list(first.claims), first.unknowns
        stop = "submitted"
        if new_sources or new_snippets:
            second_message = self._second_message(new_sources, new_snippets)
            (self.run_dir / "reading_2.txt").write_text(second_message, encoding="utf-8")
            if self._projected_cost(len(second_message), self.run.calls[-1] if self.run.calls else None) > self.cost_limit:
                stop = "cost_limit"
                self.run.events.append("second reading skipped: projected cost over the guard")
            else:
                self.client.add_tool_results(self._reply_to_first(second_message))
                second, error = self._read()
                if second is None:
                    stop = "error"
                    self.run.events.append(f"second reading failed: {error}")
                else:
                    claims, unknowns = self._merge(first.claims, second.claims), second.unknowns
        else:
            self.run.events.append("second reading skipped: nothing new was collected")
        return self._finish(stop, claims=claims, unknowns=unknowns)

    # ---- step 1: anchors ----

    def _fetch_anchors(self) -> list[str]:
        """Company site, accelerator page, then up to SITE_PAGES_CAP more pages of the company's own site: the about page
        its homepage links to, then pages its sitemap.xml lists (team, about, press, imprint, language prefixes
        included), or common paths when there is no usable sitemap. A homepage that redirects to another domain is
        flagged in the run's events."""
        ids = []
        for url in (self.row.company_url, self.row.accelerator_profile_url):
            if url:
                source_id = self._fetch_source(url, None, FoundVia.INPUT)
                if source_id:
                    ids.append(source_id)
        company_id = self.fetched_ok.get(normalise_url(self.row.company_url), "")
        company_page = self.page_meta.get(company_id, {})
        landed = company_page.get("url", "")
        if landed and registrable_domain(landed) != registrable_domain(self.row.company_url):
            self.run.events.append(f"redirect: {domain_of(self.row.company_url)} now redirects to {domain_of(landed)}")
        site = landed or self.row.company_url
        company_host = urlsplit(site).netloc.lower().removeprefix("www.")
        about = [link for link in company_page.get("links", [])
                 if ABOUT_PATH.search(urlsplit(link).path) and is_on_domain(link, company_host)
                 and normalise_url(link) not in self.attempted and not skip_reason(link)]
        extra = 0
        for link in about[:ABOUT_PAGES]:
            source_id = self._fetch_source(link, None, FoundVia.LINK)
            if source_id:
                ids.append(source_id)
                extra += 1
        if self.sitemap is not None and site:
            read_kinds = {page_kind(src.url) for src in self.run.sources} - {None}
            listed = self.sitemap(site)
            picks = pick_site_pages(listed, site, read_kinds, self.attempted, SITE_PAGES_CAP - extra)
            how = f"sitemap: {len(listed)} urls, {len(picks)} pages picked"
            if not picks:                                    # no sitemap, or nothing useful in it: common paths
                origin = urlunsplit(urlsplit(site)[:2] + ("", "", ""))
                picks = [origin + path for path in COMMON_PATHS
                         if page_kind(path) not in read_kinds and normalise_url(origin + path) not in self.attempted][:FALLBACK_TRIES]
                how += f"; common paths tried: {len(picks)}"
            kept = 0
            for url in picks:
                if extra + kept >= SITE_PAGES_CAP:
                    break
                source_id = self._fetch_source(url, None, FoundVia.LINK)
                if source_id:
                    ids.append(source_id)
                    kept += 1
            self.run.events.append(f"{how}; pages kept: {kept}")
        return ids

    def _fetch_source(self, url: str, subject: Optional[str], found_via: FoundVia,
                      anchor_check: bool = False) -> Optional[str]:
        """Fetch one page (the cache first), log the action, store the page. Returns the source id, or None when the
        fetch failed or (anchor_check) the page names neither the company nor a co-founder: then it is dropped."""
        key = normalise_url(url)
        self.attempted.add(key)
        cached = self.cache.get(url) if self.cache else None
        if cached:
            result, fetched_at = cached
            self.run.events.append(f"cache: reused page fetched {fetched_at:%Y-%m-%d %H:%M}: {url}")
        else:
            result, fetched_at = self.fetch(url), datetime.now()
            if self.cache:
                self.cache.put(url, result, fetched_at)
        action = Action(action_id=f"a{len(self.run.actions) + 1:03d}", kind=ActionKind.FETCH, query_or_url=url,
                        timestamp=datetime.now(), outcome=result.outcome, for_subject=subject)
        self.run.actions.append(action)
        self.run.budget.actions_used += 1
        if result.outcome != ActionOutcome.OK:
            return None
        if anchor_check and not page_is_anchored(result.text, subject, self.row, self.founders,
                                                  self.anchor_organisations.get(subject or "", ())):
            self.run.events.append(f"dropped (names neither the company nor a co-founder): {url}")
            return None
        self.run.budget.pages_fetched += 1
        source_id = self._store_source(url, result, action.action_id, fetched_at, found_via)
        return source_id

    def _store_source(self, url: str, result: FetchResult, action_id: str, fetched_at: datetime,
                      found_via: FoundVia) -> str:
        """Save the page text and register it as a Source. If the fetch redirected, the requested URL is
        registered too, so a claim citing either URL matches the same text."""
        source_id = f"s{len({s.source_id for s in self.run.sources}) + 1:03d}"
        text_path = self.pages_dir / f"{source_id}.txt"
        text_path.write_text(result.text, encoding="utf-8")
        (self.pages_dir / f"{source_id}.links.json").write_text(json.dumps(result.links, indent=1))
        content_hash = hashlib.sha256(result.text.encode("utf-8")).hexdigest()
        relative = str(text_path.relative_to(PROJECT_ROOT)) if text_path.is_relative_to(PROJECT_ROOT) else str(text_path)
        for u in dict.fromkeys([result.final_url or url, url]):          # final URL first, then the requested one
            if any(normalise_url(s.url) == normalise_url(u) for s in self.run.sources):
                continue
            self.run.sources.append(Source(source_id=source_id, url=u, fetched_at=fetched_at,
                                           fetched_by_action=action_id, content_hash=content_hash,
                                           page_text_path=relative, found_via=found_via))
            self.fetched_ok[normalise_url(u)] = source_id
            self.attempted.add(normalise_url(u))
        self.page_texts[source_id] = result.text
        self.page_meta[source_id] = {"url": result.final_url or url, "title": result.title, "links": result.links,
                                     "truncated": result.truncated or len(result.text) > MODEL_PAGE_CHARS}
        return source_id

    # ---- searches ----

    def _search_allowed(self, subject: Optional[str], snippet: bool = False) -> Optional[str]:
        """None when a search may run, otherwise why not. The snippet budget has its own count and no other cap."""
        if snippet:
            if self.snippet_searches >= self._snippet_cap(len(self.founders) or ASSUMED_FOUNDERS):
                return "snippet budget reached"
        else:
            cap = SEARCHES_PER_COMPANY + (1 if self.discovery else 0) if subject is None else SEARCHES_PER_FOUNDER
            if self.main_searches >= self._search_cap(len(self.founders) or ASSUMED_FOUNDERS, self.discovery):
                return "search cap for the company reached"
            if self.searches_by_subject[subject] >= cap:
                return f"search cap for {subject or 'the company'} reached"
        if self.run.cost_usd >= SOFT_COST_SHARE * self.cost_limit:
            return "cost near the guard"
        return None

    def _search(self, query: str, subject: Optional[str], kind: str, sites: Site = None, qtype: str = "",
                position: Optional[int] = None, snippet: bool = False) -> bool:
        why_not = self._search_allowed(subject, snippet)
        if why_not:
            self.run.events.append(f"search skipped ({why_not}): {query}")
            return False
        response = self.search(query, sites)
        if snippet:
            self.snippet_searches += 1
        else:
            self.main_searches += 1
            self.searches_by_subject[subject] += 1
        action = Action(action_id=f"a{len(self.run.actions) + 1:03d}", kind=ActionKind.SEARCH,
                        query_or_url=query + (f" [sites: {', '.join(sites)}]" if sites else ""),
                        timestamp=datetime.now(),
                        outcome=ActionOutcome.FETCH_FAILED if response.error else ActionOutcome.OK, for_subject=subject)
        self.run.actions.append(action)
        self.run.budget.actions_used += 1
        self.run.budget.searches_used += 1
        self.run.cost_usd += response.cost_usd
        self.run.events.append(f"search {'reused from cache' if response.cached else 'fee'}: {response.credits} "
                               f"credit(s), ${response.credits * CREDIT_PRICE_USD:.4f}" + (" (not billed)" if response.cached else ""))
        self.run.events.append(f"search {action.action_id} type={qtype or kind}" + (f" position={position}" if position else ""))
        if response.error:
            self.run.events.append(f"search failed: {query}: {response.error}")
            return False
        hits = [h for h in response.hits if not is_contact_broker(h.url)]        # never a source: not stored, not shown
        if len(hits) < len(response.hits):
            self.run.events.append(f"search {action.action_id}: {len(response.hits) - len(hits)} contact-data broker result(s) removed")
        for hit in hits:
            self.run.search_results.append(SearchResult(action_id=action.action_id, rank=hit.rank, url=hit.url,
                                                        title=hit.title, snippet=hit.snippet))
            self.run.search_result_urls.append(hit.url)
        self.queries.append(Query(query, subject, kind, action.action_id, hits))
        return True

    def _run_searches(self, entities: list[EntityMention]) -> None:
        """Step 4: name+company for founders found only now (discovery), up to 3 organisation queries per founder,
        then the company queries."""
        company = self.row.company_name
        already = {q.subject for q in self.queries if q.kind == "name"}
        for name in self.founders:
            if name not in already:
                self._search(f'"{name}" "{company}"', name, "name", qtype="name+company")
        per_founder: dict[str, list[str]] = {name: [] for name in self.founders}
        for e in entities:
            subject = self._subject(e.subject)
            organisation = " ".join(e.name.split())
            if (subject not in per_founder or not organisation or _whole_word(organisation, company)
                    or organisation.lower() in (o.lower() for o in per_founder[subject])):
                continue
            per_founder[subject].append(organisation)
        for name, organisations in per_founder.items():
            for position, organisation in enumerate(organisations[:ENTITY_QUERIES_PER_FOUNDER], start=1):
                self._search(f'"{name}" "{organisation}"', name, "entity", qtype="name+organisation", position=position)
        self._search(f'"{company}" funding investors', None, "company", qtype="company funding/investors")
        self._run_snippet_searches()

    def _snippet_search(self, query: str, subject: Optional[str], site: str, qtype: str) -> int:
        """One search of the snippet budget; the number of results it returned (0 when it failed)."""
        before = len(self.queries)
        ok = self._search(query, subject, "snippet", sites=[site], qtype=qtype, snippet=True)
        return len(self.queries[-1].hits) if ok and len(self.queries) > before else 0

    def _run_snippet_searches(self) -> None:
        """The snippet budget: each database by the company's domain (a generic name finds other companies; the
        website does not), by the company's name only when the domain finds nothing there; each founder's LinkedIn profile. Their snippets are the evidence; nothing here fetches a LinkedIn page."""
        company, domain = self.row.company_name, domain_of(self.row.company_url)
        for site in FUNDING_SITES:
            found = self._snippet_search(f'"{domain}"', None, site, f"database {site} (domain)") if domain else 0
            if not found:
                self._snippet_search(f'"{company}"', None, site, f"database {site} (name)")
        for name in self.founders:
            self._snippet_search(f'"{name}" "{company}"', name, LINKEDIN, "linkedin founder")

    # ---- fetching what the model chose, and what the searches found ----

    def _fetch_picked_links(self, picks: list[LinkPick]) -> None:
        """Links the model chose from the link lists. Code keeps only URLs that were shown to it, are fetchable, not
        fetched yet, and at most LINK_PICKS_PER_FOUNDER per founder and LINK_PICKS_PER_COMPANY for the company."""
        taken: Counter = Counter()
        for pick in picks:
            subject, key = self._subject(pick.for_subject), normalise_url(pick.url)
            cap = LINK_PICKS_PER_COMPANY if subject is None else LINK_PICKS_PER_FOUNDER
            if subject is not None and subject not in self.founders:
                self.run.events.append(f"link refused (not a founder of this company): {pick.url}")
            elif key not in self.shown_links:
                self.run.events.append(f"link refused (not on a page shown to the model): {pick.url}")
            elif skip_reason(pick.url):
                self.run.events.append(f"link refused ({skip_reason(pick.url)}): {pick.url}")
            elif key in self.attempted:
                continue
            elif taken[subject] >= cap:
                self.run.events.append(f"link refused (cap of {cap} for {subject or 'the company'}): {pick.url}")
            else:
                taken[subject] += 1
                self._fetch_source(self.shown_links[key], subject, FoundVia.LINK)

    def _fetch_results(self) -> None:
        """Fetch the best search results per founder and for the company: the top hit of each query first, then
        the second hits, and so on. Filter on title and snippet first; the fetched page must pass the anchor check."""
        for subject in [*self.founders, None]:
            cap = RESULT_FETCHES_PER_COMPANY if subject is None else RESULT_FETCHES_PER_FOUNDER
            queries = [q for q in self.queries if q.subject == subject]
            candidates = sorted(((hit.rank, i, hit) for i, q in enumerate(queries) for hit in q.hits),
                                key=lambda t: (t[0], t[1]))
            kept = attempts = 0
            for _, _, hit in candidates:
                if kept >= cap or attempts >= cap * FETCH_ATTEMPTS_FACTOR:
                    break
                key = normalise_url(hit.url)
                if key in self.attempted or skip_reason(hit.url) or not pass_prefilter(hit, subject, self.row.company_name, domain_of(self.row.company_url)):
                    continue
                if not self._robots_allow(hit.url):
                    self.run.events.append(f"robots.txt disallows {domain_of(hit.url)}: the snippet is used, not the page")
                    self.attempted.add(key)
                    continue
                if self.run.cost_usd >= SOFT_COST_SHARE * self.cost_limit:
                    self.run.events.append("fetching stopped: cost near the guard")
                    return
                attempts += 1
                if self._fetch_source(hit.url, subject, FoundVia.SEARCH, anchor_check=True):
                    kept += 1

    def _robots_allow(self, url: str) -> bool:
        """robots.txt decides before any fetch of a search result, once per host: a disallowed page is not requested
        (no failed fetch is logged); its snippet still goes to the model."""
        host = domain_of(url)
        if host not in self.robots_by_host:
            self.robots_by_host[host] = bool(self.robots(url))
        return self.robots_by_host[host]

    # ---- what the model is shown ----

    def _page_block(self, source_id: str, with_links: bool) -> str:
        meta = self.page_meta[source_id]
        text = self.page_texts[source_id]
        cut = f"\n[text cut after {MODEL_PAGE_CHARS} characters]" if len(text) > MODEL_PAGE_CHARS else ""
        links = ""
        if with_links and meta["links"]:
            shown = meta["links"][:MODEL_LINKS]
            self.shown_links.update({normalise_url(link): link for link in shown})
            links = "\nLinks on this page (follow_links may only use these):\n" + "\n".join(f"- {link}" for link in shown)
        return (f"--- [{source_id}] {meta['url']}\nTitle: {meta['title']}\n{text[:MODEL_PAGE_CHARS]}{cut}{links}")

    def _snippet_block(self, queries: list[Query]) -> str:
        lines = []
        for q in queries:
            lines.append(f"Search for {q.subject or 'the company'}: {q.text}")
            for hit in q.hits:
                key = normalise_url(hit.url)
                if key in self.fetched_ok:                      # the page itself is shown under Pages
                    continue
                note = " [not fetched: LinkedIn/X]" if is_unfetchable_social(hit.url) else " [not fetched]"
                lines.append(f"  {hit.rank}. {hit.title} | {hit.url}{note}\n     {hit.snippet}")
                self.shown_snippets.add(key)
        return "\n".join(lines)

    def _first_message(self, anchor_ids: list[str]) -> str:
        pages = "\n\n".join(self._page_block(s, with_links=True) for s in anchor_ids) or "(no anchor page could be fetched)"
        snippets = self._snippet_block(self.queries) or "(no search was run)"
        return (task_header(self.row, self.raw_founders)
                + "\n\nFIRST READING.\n\n=== Pages ===\n" + pages
                + "\n\n=== Search results (snippets) ===\n" + snippets)

    def _unseen_snippets(self) -> list[Query]:
        """Queries run since the first reading that have a hit the model has not seen and whose page was not fetched."""
        return [q for q in self.queries
                if any(normalise_url(h.url) not in self.shown_snippets and normalise_url(h.url) not in self.fetched_ok
                       for h in q.hits)]

    def _second_message(self, new_sources: list[str], new_queries: list[Query]) -> str:
        pages = "\n\n".join(self._page_block(s, with_links=False) for s in new_sources) or "(no new page)"
        unseen = [Query(q.text, q.subject, q.kind, q.action_id,
                        [h for h in q.hits if normalise_url(h.url) not in self.shown_snippets]) for q in new_queries]
        snippets = self._snippet_block(unseen) or "(no new search result)"
        return ("SECOND READING. Submit only claims that are not already submitted, and the complete list of "
                "unknowns that remain. Leave entities and follow_links empty.\n\n=== New pages ===\n" + pages
                + "\n\n=== New search results (snippets) ===\n" + snippets)

    # ---- model calls ----

    def _reply_to_first(self, second_message: str) -> list[ToolResult]:
        """Answer the first reading's tool call with the second reading, so the conversation stays valid."""
        return [ToolResult(call_id, second_message if i == 0 else "Ignored: one submit_claims per reading.")
                for i, call_id in enumerate(self._open_calls)]

    def _read(self) -> tuple[Optional[Extraction], Optional[str]]:
        """One reading: call the model until it submits a valid Extraction (a few nudges), or give up."""
        nudges = 0
        while True:
            turn = self.client.step()
            self._log_call(turn)
            if turn.error and not turn.tool_calls:
                return None, turn.error
            calls = [c for c in turn.tool_calls if c.name == "submit_claims"]
            if not calls:
                nudges += 1
                if nudges > MAX_NUDGES:
                    return None, "model stopped without submitting claims"
                self.client.add_user_text("Call submit_claims with your claims and unknowns.")
                continue
            self._open_calls = [c.id for c in turn.tool_calls]
            problem = self._invalid(calls[0])
            if problem is None:
                return Extraction.model_validate(calls[0].input), None
            nudges += 1
            if nudges > MAX_NUDGES:
                return None, "submission invalid after retries"
            self.client.add_tool_results([ToolResult(c.id, problem, is_error=True) for c in turn.tool_calls])

    def _invalid(self, call: ToolCall) -> Optional[str]:
        if call.input is None:
            self.run.events.append("submit_claims with invalid JSON arguments")
            return "submit_claims arguments were not valid JSON. Submit again."
        try:
            Extraction.model_validate(call.input)
        except ValidationError as e:
            self.run.events.append(f"submit_claims failed validation: {e.error_count()} errors")
            return f"Submission did not validate; fix and submit again:\n{str(e)[:3000]}"
        return None

    def _projected_cost(self, new_chars: int, last: Optional[ModelCall]) -> float:
        """Spend so far plus an estimate of the next call: everything it will read, uncached, plus the same output
        as the last call produced. Errs high, so the guard stops before crossing the limit, not after."""
        if last is None:
            tokens_in, tokens_out = PROMPT_TOKENS_ESTIMATE + new_chars / CHARS_PER_TOKEN, 4_000
        else:
            tokens_in = last.input_tokens + last.cache_write_tokens + last.cache_read_tokens + new_chars / CHARS_PER_TOKEN
            tokens_out = last.output_tokens
        return (self.run.cost_usd + tokens_in * self.client.input_price_per_token
                + tokens_out * self.client.output_price_per_token)

    def _log_call(self, turn: Turn) -> None:
        self.run.calls.append(ModelCall(
            provider=self.client.provider,
            model=turn.model or f"(no response; requested {self.client.model})",   # the model the API reported
            input_tokens=turn.usage.input_tokens, cache_write_tokens=turn.usage.cache_write_tokens,
            cache_read_tokens=turn.usage.cache_read_tokens, output_tokens=turn.usage.output_tokens,
            cost_usd=turn.cost_usd, seconds=round(turn.seconds, 2), stop_reason=turn.stop_reason, error=turn.error))
        self.run.input_tokens += turn.usage.total_input
        self.run.output_tokens += turn.usage.output_tokens
        self.run.cache_read_tokens += turn.usage.cache_read_tokens
        self.run.cost_usd += turn.cost_usd

    # ---- merging and finishing ----

    @staticmethod
    def _claim_key(c: ProposedClaim) -> tuple:
        return (c.subject.lower().strip(), c.claim_type, c.attribute, normalise_text(c.value).lower(),
                normalise_url(c.source_url), normalise_text(c.quote))

    def _merge(self, first: list[ProposedClaim], second: list[ProposedClaim]) -> list[ProposedClaim]:
        """The second reading adds only new claims: exact duplicates of the first are dropped."""
        seen = {self._claim_key(c) for c in first}
        added = []
        for c in second:
            key = self._claim_key(c)
            if key not in seen:
                seen.add(key)
                added.append(c)
        if len(second) != len(added):
            self.run.events.append(f"dropped {len(second) - len(added)} duplicate claims from the second reading")
        return list(first) + added

    def _adopt_discovered_founders(self, first: Extraction) -> None:
        """Founder discovery: a subject of the first reading becomes a founder only with a sourced identity or role
        claim (the quote is on a page this run fetched), so an investor, a customer or a co-author named in a claim is
        not adopted. The founders are written back into the run (input row included), so assembly, the page and the
        export see them. At most MAX_DISCOVERED_FOUNDERS, in the order they first appear."""
        claims, _ = verify_claims(first.claims, self.run, self.page_texts)
        found = []
        for c in claims:
            name = " ".join(c.subject.split())
            if (c.status == ClaimStatus.SOURCED and c.claim_type in (ClaimType.IDENTITY, ClaimType.ROLE)
                    and name and name.lower() != "company" and name.lower() not in (f.lower() for f in found)):
                found.append(name)
        self.founders = found[:MAX_DISCOVERED_FOUNDERS]
        self.raw_founders = list(self.founders)
        self.row = self.row.model_copy(update={"founder_names": self.founders})
        self.run.input = self.row
        self.run.budget.max_searches = self._budget_total()
        self.run.events.append(f"founders discovered: {len(self.founders)} (each with a sourced identity or role claim)")

    def _sourced_organisations(self, proposed: list[ProposedClaim]) -> tuple[dict, dict]:
        """Organisations (value_normalized.organization) of the first reading's claims that passed verify.py, per
        founder: (those whose page is an anchor page, an input URL or a link found on one; those from any page).
        The first kind anchors pages found by search (a page naming the founder and one of these organisations is not
        about a namesake); the second is what OpenAlex affiliations are compared with."""
        claims, evidence = verify_claims(proposed, self.run, self.page_texts)
        source_of = {e.evidence_id: e.source_id for e in evidence}
        search_sources = {s.source_id for s in self.run.sources if s.found_via == FoundVia.SEARCH}
        anchor: dict[str, list[str]] = {}
        anywhere: dict[str, list[str]] = {}
        self.sourced_texts, self.sector_tags = {}, []
        for c in claims:
            if c.status == ClaimStatus.SOURCED and c.claim_type == ClaimType.SECTOR:
                self.sector_tags.append(c.value.strip())
            if c.status == ClaimStatus.SOURCED and c.subject.strip().lower() != "company" and c.claim_type in BACKGROUND_TYPES:
                self.sourced_texts.setdefault(self._subject(c.subject) or "", []).append(c.value)
            organisation = (c.value_normalized or {}).get("organization")
            if c.status != ClaimStatus.SOURCED or not organisation or c.subject.strip().lower() == "company":
                continue
            name, organisation = self._subject(c.subject) or "", " ".join(organisation.split())
            anywhere.setdefault(name, []).append(organisation)
            if source_of.get(c.evidence_id) not in search_sources:
                anchor.setdefault(name, []).append(organisation)
        pack = lambda d: {name: tuple(dict.fromkeys(orgs)) for name, orgs in d.items()}
        return pack(anchor), pack(anywhere)

    def _run_openalex(self) -> None:
        """Publications per founder from OpenAlex: an author is accepted only with two independent signals (affiliation
        overlaps with sourced institutions, a co-founder as co-author, topic fit). The reduced response becomes a source
        (found_via api). Its institutions are not added to the anchor organisations."""
        for name in self.founders:
            result = self.openalex(name, list(self.sourced_organisations.get(name, ())),
                                   [f for f in self.founders if f != name], aliases=self._aliases(name),
                                   fields=allowed_fields(self.sourced_texts.get(name, []), self.sector_tags))
            for url in result.requests:
                self.run.actions.append(Action(
                    action_id=f"a{len(self.run.actions) + 1:03d}", kind=ActionKind.API, query_or_url=url,
                    timestamp=datetime.now(), outcome=ActionOutcome.FETCH_FAILED if result.error else ActionOutcome.OK,
                    for_subject=name))
                self.run.budget.actions_used += 1
            self.run.cost_usd += result.cost_usd
            if not result.requests:
                if result.error:                     # no key: the same for every founder, so say it once
                    self.run.events.append(f"openalex skipped: {result.error}")
                    return
                continue
            self.run.events.append(
                f"openalex: {name}: {len(result.requests)} requests (cost ${result.cost_usd:.2f}), {result.candidates} "
                f"candidate authors ({result.name_mismatch} name mismatch, {result.no_overlap} with fewer than two "
                f"signals; " + "; ".join(f"{', '.join(sig) or 'no signal'}{' = accepted' if ok else ''}" for _, sig, ok in result.decisions)
                + f"), accepted by "
                f"{result.accepted_by or 'none'}, {len(result.works)} works" + (f"; error {result.error}" if result.error else ""))
            if result.works and result.source_text:
                action_id = self.run.actions[-1].action_id
                fetched = FetchResult(requested_url=result.source_url, outcome=ActionOutcome.OK, final_url=result.source_url,
                                      status=200, title=f"OpenAlex works for {name} (author matched on {result.accepted_by.replace('_', ' ').replace('+', ' and ')})",
                                      text=result.source_text)
                self._store_source(result.source_url, fetched, action_id, datetime.now(), FoundVia.API)

    def _aliases(self, name: str) -> tuple[str, ...]:
        """Other names a founder goes by, from the input's brackets: "Mira (Maya) Lindqvist" -> "Maya Lindqvist";
        "Sam Okafor (registered as Samuel Okafor)" -> "Samuel Okafor"."""
        raw = next((r for n, r in zip(self.row.founder_names, self.raw_founders) if n == name), "")
        found = []
        for inner in re.findall(r"\(([^)]*)\)", raw):
            inner = re.sub(r"^(registered|also known|aka|a\.k\.a\.)( as)?\s+", "", inner.strip(), flags=re.I)
            words = inner.split()
            if words:
                found.append(inner if len(words) >= 2 else f"{inner} {name.split()[-1]}")
        return tuple(a for a in dict.fromkeys(found) if a.lower() != name.lower())

    def _subject(self, value: Optional[str]) -> Optional[str]:
        """Founder name as in the founder list, or None for company-level actions."""
        if not value or value.strip().lower() == "company":
            return None
        for name in self.founders:
            if name.lower() == value.strip().lower():
                return name
        return value.strip()

    def _finish(self, stop_reason: str, error: Optional[str] = None, claims: Optional[list[ProposedClaim]] = None,
                unknowns: Optional[list[Unknown]] = None) -> Run:
        claims = claims or []
        (self.run_dir / "submission.json").write_text(
            json.dumps({"claims": [c.model_dump(mode="json") for c in claims],
                        "unknowns": [u.model_dump(mode="json") for u in unknowns or []]}, indent=2, ensure_ascii=False))
        self.run.claims, self.run.evidence = verify_claims(claims, self.run, self.page_texts)
        self.run.unknowns = unknowns or []
        self.run.stop_reason = stop_reason
        if error:
            self.run.events.append(f"stopped: {error}")
        self.run.finished_at = datetime.now()
        (self.run_dir / "run.json").write_text(self.run.model_dump_json(indent=2))
        (self.run_dir / "summary.txt").write_text(summarise(self.run))
        return self.run


def find_row(company: str, input_path: Path) -> tuple[BatchRow, list[str]]:
    """The batch row for a company name (case-insensitive) from an input CSV or the golden workbook."""
    from src.golden import read_input_rows
    if not Path(input_path).exists():
        raise SystemExit(f"{input_path} not found. golden/golden_input.csv is private (real founders); give --input with a "
                         "CSV in the format of data/golden_input.example.csv, or use `python -m src.batch` with a list of URLs.")
    rows = {row.company_name.lower(): (row, raw) for row, raw in read_input_rows(input_path)}
    if company.lower() not in rows:
        raise SystemExit(f"{company!r} not in {input_path}: {', '.join(r.company_name for r, _ in rows.values())}")
    return rows[company.lower()]


def without_founder_names(row: BatchRow) -> BatchRow:
    """The same row with the founder names removed, for founder discovery."""
    return row.model_copy(update={"founder_names": []})


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def summarise(run: Run) -> str:
    sourced = [c for c in run.claims if c.status == ClaimStatus.SOURCED]
    snippets = [c for c in run.claims if c.status == ClaimStatus.SNIPPET]
    rejected = [c for c in run.claims if c.status == ClaimStatus.REJECTED]
    leads = [c for c in rejected if c.rejection_reason in LEAD_REASONS]
    by_type = Counter(c.claim_type.value for c in sourced)
    by_reason = Counter(c.rejection_reason.value for c in rejected)
    per_subject = Counter(a.for_subject or "company" for a in run.actions)
    corrected = [c for c in run.claims if c.origin_corrected_from]
    fresh = sum(e.startswith("search fee") for e in run.events)
    reused = sum(e.startswith("search reused") for e in run.events)

    lines = [
        f"Run {run.run_id}: {run.input.company_name} ({run.provider}, {run.model})",
        f"Stopped: {run.stop_reason}",
        f"Claims: {len(run.claims)} submitted, {len(sourced)} sourced, {len(snippets)} snippet, "
        f"{len(rejected)} rejected; {len(run.unknowns)} unknowns",
        "Sourced by type: " + (", ".join(f"{t} {n}" for t, n in sorted(by_type.items())) or "none"),
        "Rejected by reason: " + (", ".join(f"{r} {n}" for r, n in sorted(by_reason.items())) or "none"),
        f"Leads ({len(leads)}):",
        *[f"  - [{LEAD_LABELS[c.rejection_reason]}] {c.subject} / {c.claim_type.value}: {c.value[:90]} <{c.source_url}>"
          for c in leads],
        f"Origins corrected to company (company-domain check): {len(corrected)}",
        f"Actions: {run.budget.actions_used} ({run.budget.searches_used} searches, {run.budget.pages_fetched} pages "
        f"kept; " + ", ".join(f"{s} {n}" for s, n in per_subject.items()) + ")",
        f"Searches: {run.budget.searches_used} of {run.budget.max_searches} ({fresh} billed, {reused} from the cache); "
        f"{len(run.search_results)} results stored; cost guard ${run.budget.cost_limit_usd:.2f}",
        f"Pages dropped or links refused: {sum(e.startswith(('dropped (', 'link refused')) for e in run.events)}",
        f"Model calls: {len(run.calls)}; tokens: {run.input_tokens} in "
        f"({run.cache_read_tokens} cache reads, {sum(c.cache_write_tokens for c in run.calls)} cache writes), "
        f"{run.output_tokens} out",
        f"Estimated cost: ${run.cost_usd:.4f} (model ${sum(c.cost_usd for c in run.calls):.4f}, billed searches "
        f"${run.cost_usd - sum(c.cost_usd for c in run.calls):.4f})",
        "Models returned by the API: " + ", ".join(f"{m} x{n}" for m, n in Counter(c.model for c in run.calls).items()),
    ]
    if run.events and any(e.startswith("stopped") for e in run.events):
        lines.append(next(e for e in run.events if e.startswith("stopped")))
    return "\n".join(lines) + "\n"


def claims_report(run: Run, run_dir: Path) -> str:
    """Claim table and actions."""
    submission_path = run_dir / "submission.json"
    proposed = json.loads(submission_path.read_text())["claims"] if submission_path.exists() else []
    rows = [("subject", "type", "attribute", "value", "origin", "status", "source URL")]
    for i, c in enumerate(run.claims):
        status = ("sourced" if c.status == ClaimStatus.SOURCED else "snippet" if c.status == ClaimStatus.SNIPPET
                  else f"rejected: {c.rejection_reason.value}")
        url = proposed[i].get("source_url", "") if i < len(proposed) else ""
        rows.append((c.subject, c.claim_type.value, c.attribute.value if c.attribute else "", c.value[:60],
                     c.source_origin.value, status, url))
    widths = [max(len(r[k]) for r in rows) for k in range(len(rows[0]))]
    lines = ["Claims:"] + [" | ".join(v.ljust(w) for v, w in zip(r, widths)).rstrip() for r in rows]

    lines.append("Actions:")
    for a in run.actions:
        lines.append(f"  {a.action_id} {a.kind.value:7} {a.outcome.value:14} {a.for_subject or 'company':14} {a.query_or_url}")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def new_run_dir() -> Path:
    """runs/<timestamp>, with a numeric suffix when two runs start in the same second (a batch)."""
    base = PROJECT_ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}_{n}")
    return path


def run_company(row: BatchRow, raw_founders: list[str], provider: str = "anthropic", model: Optional[str] = None,
                cost_limit: float = COST_LIMIT_USD, fresh: bool = False, run_dir: Optional[Path] = None) -> Run:
    """One company end to end with the real model, search, fetch and cache. Used by the CLI and the batch runner."""
    run_dir = run_dir or new_run_dir()
    client = make_client(provider, SYSTEM_PROMPT, [SUBMIT_TOOL], model=model)
    cache = PageCache(read=not fresh)
    search = cached_search(SearchCache(read=not fresh), web_search)
    return Researcher(row, raw_founders, client, run_dir, cache=cache, search=search, cost_limit=cost_limit).research()


def main() -> None:
    from src.golden import GOLDEN_INPUT

    parser = argparse.ArgumentParser(description="Research one company from a batch input CSV.")
    parser.add_argument("company", help="company_name as in the input, e.g. 'Example Robotics'")
    parser.add_argument("--input", default=str(GOLDEN_INPUT),
                        help="batch CSV with the golden input's columns (default: golden/golden_input.csv, which is private; "
                             "data/golden_input.example.csv shows the format)")
    parser.add_argument("--discover-founders", action="store_true",
                        help="drop the founder names from the input; the model must find the founders")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openrouter"])
    parser.add_argument("--model", default=None, help="override the provider's default model")
    parser.add_argument("--cost-limit", type=float, default=COST_LIMIT_USD,
                        help="hard cost guard per company in USD, search fees included")
    parser.add_argument("--fresh", action="store_true",
                        help="fetch and search everything again instead of reusing pages and searches from the last 7 days")
    args = parser.parse_args()

    row, raw_founders = find_row(args.company, Path(args.input))
    if args.discover_founders:
        row, raw_founders = without_founder_names(row), []
        print("Founder names removed from the input: the model must find the founders.")

    run_dir = new_run_dir()
    run = run_company(row, raw_founders, args.provider, args.model, args.cost_limit, args.fresh, run_dir)
    print(summarise(run), end="")
    report = claims_report(run, run_dir)
    (run_dir / "claims_report.txt").write_text(report)
    print(report, end="")
    print(f"Saved: {run_dir.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
