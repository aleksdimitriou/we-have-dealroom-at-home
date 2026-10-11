"""
antler.py: a batch input from Antler's public portfolio directory.

    python -m src.intake --antler-portfolio --country Germany --years 2025-2026 [--exclude cereal] [--limit 10] [--out DIR]

antler.co/portfolio lists every company as a card (name, one-line description, country, sector, cohort year) and
links the card to the company's own site. The list is paged; the pages are linked as `?<key>_page=N`. This module
crawls the pages in order (robots.txt first, the 10-second Crawl-delay honoured by fetch.py, pages through the page
cache), reads the cards from each page's text, filters them locally by country and cohort years and writes:

    urls.csv    url, name, accelerator_url (+ country, sector, year): a list `python -m src.batch` reads as it is
    rows.json   the same rows, resolved, for `python -m src.batch --rows`
    skipped.txt cards that matched the filter but whose company site could not be tied to the card

The accelerator URL of a row is the portfolio page that holds the card, so "Investor: Antler" can be confirmed like a
YC batch listing. The cards carry no founder names; the workflow's founder discovery finds them. The order is
the site's own (alphabetical); the cards are not ranked or judged. Nothing here calls a model.

How a card's site is found: the page text and the page's link list are separate, so the links are matched to the cards
in order by name (the domain must contain the card's name, or the other way round); a card that matches nothing is
filled from its neighbours only when exactly as many unclaimed links lie between them as there are cards unmatched.
Anything still unmatched is listed in skipped.txt rather than guessed.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

from schema import ActionOutcome, BatchRow
from src.cache import PageCache
from src.fetch import FetchResult, fetch_page
from src.intake import IntakeError, Resolved, save_rows
from src.verify import normalise_url

PORTFOLIO_URL = "https://www.antler.co/portfolio"
MAX_PAGES = 40                                   # a crawl is bounded even if the pagination links loop
LOOKAHEAD = 8                                    # links searched ahead of the last matched one, per card
SECTORS = ("Real Estate and PropTech", "Industrials", "Health and BioTech", "FinTech", "Energy and ClimateTech",
           "ConsumerTech", "B2B Software")
_SHOWING = re.compile(r"^Showing \d+ of \d+$")
_YEAR = re.compile(r"^(20\d\d)$")
_PAGE_LINK = re.compile(r"^https?://(?:www\.)?antler\.co/portfolio\?\w+_page=(\d+)$")
_NOT_A_SITE = re.compile(r"(^|\.)(antler\.co|linkedin\.com|facebook\.com|instagram\.com|youtube\.com|x\.com|twitter\.com|"
                         r"efrontcloud\.com)$")
_PREFIXES = ("get", "use", "try", "meet", "join", "hello", "with", "the", "my", "go")


@dataclass
class Card:
    name: str
    description: str
    country: str
    sector: str
    year: Optional[int]
    page_url: str                                # the portfolio page that holds the card
    site: str = ""                               # the company's own site, "" when it could not be tied to the card


# --------------------------------------------------------------------------
# Reading one page
# --------------------------------------------------------------------------

def _filter_countries(lines: list[str], showing: int) -> set[str]:
    """The countries the page's own Location filter offers (the lines between "Location" and "Sector")."""
    try:
        loc = max(i for i in range(showing) if lines[i] == "Location")
        return set(lines[loc + 2:lines.index("Sector", loc)])
    except ValueError:
        return set()


def parse_page_cards(text: str) -> tuple[list[Card], list[str]]:
    """The cards of one portfolio page, in order, and a list of problems (blocks that did not look like a card).
    A card is name, description, country, sector, and a year when the site has one; the sector line ends the card.
    Some cards lack the description or the country: with two lines before the sector the second is read as the country
    when the page's Location filter offers it, else as the description (and the country stays empty)."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if _SHOWING.match(l)), None)
    if start is None:
        return [], ["no card list found on the page"]
    countries = _filter_countries(lines, start)
    body: list[str] = []
    for l in lines[start + 1:]:
        if l in ("Previous", "Next", "Load more") or l.startswith("No results"):
            break
        if l != "Tag" or body:
            body.append(l)
    cards: list[Card] = []
    problems: list[str] = []
    buffer: list[str] = []
    i = 0
    while i < len(body):
        bare_year = _YEAR.match(body[i]) and buffer                    # a card with a year but no sector line
        if body[i] in SECTORS or bare_year:
            sector = body[i] if body[i] in SECTORS else ""
            i += 1 if sector else 0
            year = None
            if i < len(body) and _YEAR.match(body[i]):
                year, i = int(body[i]), i + 1
            if 3 <= len(buffer) <= 4:                                  # name, description (1-2 lines), country
                cards.append(Card(buffer[0], " ".join(buffer[1:-1]), buffer[-1], sector, year, ""))
            elif len(buffer) == 2:                                     # a card without a description or without a country
                has_country = buffer[1] in countries
                cards.append(Card(buffer[0], "" if has_country else buffer[1], buffer[1] if has_country else "", sector, year, ""))
            elif len(buffer) == 1:                                     # a card with a name only
                cards.append(Card(buffer[0], "", "", sector, year, ""))
            else:
                problems.append(f"block of {len(buffer)} lines before sector '{sector}' not read as a card: {' / '.join(buffer)[:80]}")
            buffer = []
        else:
            buffer.append(body[i])
            i += 1
    if buffer:
        problems.append(f"{len(buffer)} trailing lines not read as a card")
    return cards, problems


def site_links(links: list[str]) -> list[str]:
    """The page's links that can be a company's own site, in page order (no Antler, social or login links)."""
    return [l for l in links if (host := urlsplit(l).netloc.lower().removeprefix("www.")) and not _NOT_A_SITE.search(host)]


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def name_keys(name: str) -> list[str]:
    """Comparable forms of a card's name: the name, the name before a bracket, and the bracket's text ("fka Jenni.ai")."""
    outer = re.sub(r"\([^)]*\)", " ", name)
    inner = re.findall(r"\(([^)]*)\)", name)
    forms = [outer, *(re.sub(r"^(fka|aka)\s+", "", i, flags=re.I) for i in inner)]
    keys = []
    for form in forms:
        for k in (_key(form), _key(re.sub(r"\.(ai|io|com|app|co|eu|de|net|org)$", "", form.strip(), flags=re.I))):
            if len(k) >= 3 and k not in keys:
                keys.append(k)
    return keys


def host_key(url: str) -> str:
    labels = urlsplit(url).netloc.lower().removeprefix("www.").split(".")
    if len(labels) >= 3 and labels[-2] in ("co", "com", "org", "ac"):
        labels = labels[:-2]
    else:
        labels = labels[:-1]
    return _key("".join(labels))


def matches(name: str, url: str) -> bool:
    host = host_key(url)
    if not host:
        return False
    stripped = next((host[len(p):] for p in _PREFIXES if host.startswith(p) and len(host) > len(p) + 2), host)
    for k in name_keys(name):
        if k in host or (len(host) >= 4 and host in k) or (len(stripped) >= 4 and stripped in k):
            return True
    first = re.split(r"[\s(]", name.strip())[0]
    return len(_key(first)) >= 4 and host.startswith(_key(first))


def assign_sites(cards: list[Card], links: list[str]) -> None:
    """Fill card.site from the page's links: in order, by name, with a short lookahead; then a gap between two matched
    cards is filled by position when exactly as many unclaimed links lie between them as cards are unmatched."""
    claimed: dict[int, int] = {}                 # card index -> link index
    pointer = 0
    for ci, card in enumerate(cards):
        window = len(links) if not claimed else LOOKAHEAD
        for li in range(pointer, min(len(links), pointer + window)):
            if matches(card.name, links[li]):
                claimed[ci], pointer = li, li + 1
                break
    matched = sorted(claimed)
    for a, b in zip(matched, matched[1:]):
        cards_between = list(range(a + 1, b))
        links_between = list(range(claimed[a] + 1, claimed[b]))
        if cards_between and len(cards_between) == len(links_between):
            claimed.update(dict(zip(cards_between, links_between)))
    for ci, li in claimed.items():
        cards[ci].site = links[li]


# --------------------------------------------------------------------------
# The crawl
# --------------------------------------------------------------------------

def _fetch(url: str, fetch: Callable[[str], FetchResult], cache: Optional[PageCache]) -> FetchResult:
    cached = cache.get(url) if cache else None
    if cached:
        return cached[0]
    result = fetch(url)
    if cache:
        cache.put(url, result, datetime.now())
    return result


@dataclass
class Crawl:
    cards: list[Card] = field(default_factory=list)
    pages: int = 0
    problems: list[str] = field(default_factory=list)


def crawl_portfolio(fetch: Callable[[str], FetchResult] = fetch_page, cache: Optional[PageCache] = None,
                    log: Callable[[str], None] = print, start: str = PORTFOLIO_URL) -> Crawl:
    """Every card of the portfolio, page by page in the order of the pagination links. Duplicates (same name and
    country) are dropped. A page that cannot be fetched stops the crawl: a partial list would look complete."""
    out = Crawl()
    queue, seen, page_numbers = [start], {normalise_url(start)}, {start: 1}
    seen_cards: set[tuple[str, str]] = set()
    while queue and out.pages < MAX_PAGES:
        url = queue.pop(0)
        page = _fetch(url, fetch, cache)
        if page.outcome != ActionOutcome.OK:
            raise IntakeError(f"{url}: page could not be fetched ({page.outcome.value}: {page.error})")
        out.pages += 1
        cards, problems = parse_page_cards(page.text)
        assign_sites(cards, site_links(page.links))
        added = 0
        for card in cards:
            card.page_url = url
            if (card.name.lower(), card.country) not in seen_cards:
                seen_cards.add((card.name.lower(), card.country))
                out.cards.append(card)
                added += 1
        out.problems += [f"{url}: {p}" for p in problems]
        log(f"page {out.pages}: {len(cards)} cards ({added} new), {sum(bool(c.site) for c in cards)} with a site")
        for link in page.links:
            m = _PAGE_LINK.match(link)
            if m and int(m.group(1)) > 1 and normalise_url(link) not in seen:                     # page 1 is the start page
                seen.add(normalise_url(link))
                page_numbers[link] = int(m.group(1))
                queue.append(link)
        queue.sort(key=lambda u: page_numbers[u])
    return out


# --------------------------------------------------------------------------
# Filter and output
# --------------------------------------------------------------------------

def parse_years(spec: str) -> tuple[int, int]:
    """'2026', '2025-2026' or '2025,2026' -> (first, last)."""
    numbers = [int(n) for n in re.findall(r"\d{4}", spec)]
    if not numbers:
        raise ValueError(f"years must look like 2026 or 2025-2026, not {spec!r}")
    return min(numbers), max(numbers)


def filter_cards(cards: list[Card], country: Optional[str], years: Optional[tuple[int, int]],
                 exclude: tuple[str, ...] = (), limit: Optional[int] = None) -> list[Card]:
    """Cards in the site's order that match the country and the cohort years, minus excluded names. A card with no
    year never matches a year filter."""
    skip = {e.lower() for e in exclude}
    kept = [c for c in cards
            if (not country or c.country.lower() == country.lower())
            and (not years or (c.year is not None and years[0] <= c.year <= years[1]))
            and c.name.lower() not in skip]
    return kept[:limit] if limit else kept


def write_outputs(cards: list[Card], out_dir: Path) -> tuple[list[Card], list[Card]]:
    """urls.csv, rows.json and skipped.txt; returns (written, skipped)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [c for c in cards if c.site]
    skipped = [c for c in cards if not c.site]
    with (out_dir / "urls.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "name", "accelerator_url", "country", "sector", "year"])
        for c in written:
            writer.writerow([c.site, c.name, c.page_url, c.country, c.sector, c.year or ""])
    resolved = []
    for c in written:
        row = BatchRow(company_name=c.name, company_url=c.site, accelerator_profile_url=c.page_url, founder_names=[])
        resolved.append(Resolved(row, [], c.site, [
            f"Antler portfolio card: {c.country}, {c.sector}" + (f", {c.year}" if c.year else "") + "; the site was matched to the card by name",
            "company site: name from the input list; no founder names, discovery will find them"]))
    save_rows(resolved, out_dir / "rows.json")
    (out_dir / "skipped.txt").write_text("".join(f"{c.name} ({c.country}, {c.year}): no company site tied to the card on {c.page_url}\n"
                                                for c in skipped), encoding="utf-8")
    return written, skipped
