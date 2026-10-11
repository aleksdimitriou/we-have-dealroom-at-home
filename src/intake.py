"""
intake.py: a batch as a plain list of URLs.

Input file: one URL per line (`#` starts a comment), or a CSV with a header row and the columns
    url                 required: a YC company page (ycombinator.com/companies/<slug>) or a company's own site
    name                optional but recommended for company sites: the company's name from the list it came from
    founders            optional, separated by ";"
    accelerator_url     optional: the accelerator's page for the company (a YC page is its own accelerator page; for an
                        Antler company, the portfolio page that holds its card, which makes "Investor: Antler" a
                        counterparty fact)
For each URL, `resolve` fetches the page (through the page cache, so the workflow's own anchor fetch costs nothing
extra) and fills the BatchRow:
    YC page      company name from the breadcrumb, company site from the website line, founder names from the
                 "Active Founders" block (a name line followed by a role line such as "Founder/CEO"), accelerator
                 URL = the YC page. The names are as YC writes them.
    other URL    company site = the URL, name from the `name` column, no founder names: the workflow's founder
                 discovery finds them.
The company name never comes from a page title (titles are taglines, product names or garbled); without a name column
a company site's name is its domain's main label and the row says so in its notes.
The resolved rows are saved as rows.json, so the analyst can read and edit them before the run (`--rows`).
Nothing here calls a model.

    python -m src.intake --antler-portfolio --country Germany --years 2025-2026 [--exclude cereal] [--limit 10] [--out DIR]
is the filter command for Antler's portfolio (src/antler.py): it crawls the portfolio pages and writes a url list
(`urls.csv`, readable as an input file) and `rows.json` for the companies of that country and cohort years.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

from schema import ActionOutcome, BatchRow
from src.cache import PageCache
from src.fetch import FetchResult, fetch_page
from src.golden import clean_founder_name

_ROLE = re.compile(r"^(co-?founders?|founders?)(\s*(/|&|and|,)\s*[A-Za-z][\w .-]{0,24})?$", re.I)
_NAME_WORD = re.compile(r"^[A-ZÀ-Ý][\w'’.-]*$")
_BATCH = re.compile(r"^(Summer|Winter|Spring|Fall)\s+20\d\d$")


class IntakeError(Exception):
    """A URL that cannot be turned into a row (the page could not be fetched, or it is not a usable page)."""


@dataclass
class Entry:
    url: str
    founders: list[str] = field(default_factory=list)
    accelerator_url: Optional[str] = None
    name: Optional[str] = None


@dataclass
class Resolved:
    row: BatchRow
    raw_founders: list[str]
    source_url: str
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"source_url": self.source_url, "company_name": self.row.company_name, "company_url": self.row.company_url,
                "accelerator_profile_url": self.row.accelerator_profile_url, "founder_names": self.raw_founders,
                "notes": self.notes}

    @staticmethod
    def from_json(d: dict) -> "Resolved":
        raw = list(d.get("founder_names") or [])
        row = BatchRow(company_name=d["company_name"], company_url=d["company_url"],
                       accelerator_profile_url=d.get("accelerator_profile_url") or None,
                       founder_names=[clean_founder_name(f) for f in raw])
        return Resolved(row, raw, d.get("source_url", d["company_url"]), list(d.get("notes") or []))


# --------------------------------------------------------------------------
# Reading the input file
# --------------------------------------------------------------------------

def read_entries(path: Path) -> list[Entry]:
    text = path.read_text(encoding="utf-8-sig")
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    if lines and lines[0].lower().split(",")[0].strip() == "url":
        out = []
        for r in csv.DictReader(lines):
            url = (r.get("url") or "").strip()
            if url:
                out.append(Entry(url, [f.strip() for f in (r.get("founders") or "").split(";") if f.strip()],
                                 (r.get("accelerator_url") or "").strip() or None, (r.get("name") or "").strip() or None))
        return out
    return [Entry(l.split()[0]) for l in lines]


def is_yc_company_page(url: str) -> bool:
    parts = urlsplit(url)
    return (parts.netloc.lower().removeprefix("www.") == "ycombinator.com"
            and re.fullmatch(r"/companies/[^/?#]+/?", parts.path) is not None)


# --------------------------------------------------------------------------
# Parsing pages
# --------------------------------------------------------------------------

def _looks_like_name(line: str) -> bool:
    """2-5 capitalised words; a nickname in brackets ("Jonas (Joni) Weber") is allowed and ignored here."""
    words = re.sub(r"\([^)]*\)", " ", line).split()
    return 2 <= len(words) <= 5 and len(line) <= 55 and not any(ch.isdigit() for ch in line) \
        and all(_NAME_WORD.match(w) or w.lower() in ("de", "van", "von", "da", "di", "bin", "al") for w in words)


def parse_yc_page(text: str, links: list[str]) -> dict:
    """{name, website, batch, founders} from the text of a YC company page. Empty values when not found."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name = website = batch = ""
    for i, l in enumerate(lines):
        if l.startswith("Home›Companies›"):
            name = l.split("›")[-1].strip()
            for m in lines[i + 1:i + 14]:
                if _BATCH.match(m) and not batch:
                    batch = m
                if not website and re.match(r"^https?://", m) and "ycombinator.com" not in m:
                    website = m.split()[0]
            break
    if not website:
        website = next((l for l in links if not any(h in urlsplit(l).netloc for h in
                                                    ("ycombinator.com", "linkedin.com", "twitter.com", "x.com", "github.com",
                                                     "facebook.com", "crunchbase.com"))), "")
    founders: list[str] = []
    if "Active Founders" in lines:
        block = lines[lines.index("Active Founders") + 1:]
        for prev, cur in zip(block, block[1:]):
            if _ROLE.match(cur) and _looks_like_name(prev) and prev not in founders:
                founders.append(prev)
    return {"name": name, "website": website, "batch": batch, "founders": founders}


def name_from_domain(url: str) -> str:
    """The main label of the site's domain, capitalised: a last-resort name when the list gave none."""
    labels = urlsplit(url).netloc.lower().removeprefix("www.").split(".")
    stem = labels[-3] if len(labels) >= 3 and labels[-2] in ("co", "com", "org") else labels[-2] if len(labels) >= 2 else labels[0]
    return stem.capitalize()


def name_from_slug(url: str) -> str:
    return urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").title()


# --------------------------------------------------------------------------
# Resolving one entry
# --------------------------------------------------------------------------

def _fetch(url: str, fetch: Callable[[str], FetchResult], cache: Optional[PageCache]) -> FetchResult:
    cached = cache.get(url) if cache else None
    if cached:
        return cached[0]
    result = fetch(url)
    if cache:
        cache.put(url, result, datetime.now())
    return result


def resolve(entry: Entry, fetch: Callable[[str], FetchResult] = fetch_page, cache: Optional[PageCache] = None) -> Resolved:
    notes: list[str] = []
    page = _fetch(entry.url, fetch, cache)
    if page.outcome != ActionOutcome.OK:
        raise IntakeError(f"{entry.url}: page could not be fetched ({page.outcome.value}: {page.error})")
    if is_yc_company_page(entry.url):
        info = parse_yc_page(page.text, page.links)
        if not info["website"]:
            raise IntakeError(f"{entry.url}: no company website found on the YC page")
        name = entry.name or info["name"] or name_from_slug(entry.url)
        raw = list(entry.founders) or info["founders"]
        notes.append(f"YC page: {len(info['founders'])} active founder(s) read from the page"
                     + (f", batch {info['batch']}" if info["batch"] else ""))
        if entry.founders:
            notes.append("founder names taken from the input file, not from the page")
        company_url, accelerator = info["website"], entry.accelerator_url or entry.url
    else:
        name = entry.name or name_from_domain(entry.url)
        raw, company_url, accelerator = list(entry.founders), entry.url, entry.accelerator_url
        notes.append("company site: name " + ("from the input list" if entry.name else "from the domain (no name given)")
                     + ("" if raw else "; no founder names, discovery will find them"))
    row = BatchRow(company_name=name, company_url=company_url, accelerator_profile_url=accelerator,
                   founder_names=[clean_founder_name(f) for f in raw])
    return Resolved(row, raw, entry.url, notes)


def save_rows(resolved: list[Resolved], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": [r.to_json() for r in resolved]}, indent=2, ensure_ascii=False), encoding="utf-8")


def load_rows(path: Path) -> list[Resolved]:
    return [Resolved.from_json(d) for d in json.loads(path.read_text(encoding="utf-8"))["rows"]]


# --------------------------------------------------------------------------
# CLI: a batch input from Antler's portfolio
# --------------------------------------------------------------------------

def main() -> None:
    import argparse
    from src.antler import PORTFOLIO_URL, crawl_portfolio, filter_cards, parse_years, write_outputs

    parser = argparse.ArgumentParser(description="Make a batch input (url list and rows.json) from Antler's public portfolio.")
    parser.add_argument("--antler-portfolio", action="store_true", help=f"crawl {PORTFOLIO_URL} (robots.txt and its crawl delay are honoured)")
    parser.add_argument("--country", help="keep companies of this country, e.g. Germany")
    parser.add_argument("--years", help="keep these cohort years, e.g. 2026 or 2025-2026")
    parser.add_argument("--exclude", action="append", default=[], help="a company name to leave out (repeatable)")
    parser.add_argument("--limit", type=int, help="keep the first N companies of the filtered list, in the site's order")
    parser.add_argument("--out", help="output folder (default inputs/antler_<country>_<years>)")
    parser.add_argument("--fresh", action="store_true", help="bypass the page cache")
    args = parser.parse_args()
    if not args.antler_portfolio:
        raise SystemExit("nothing to do: give --antler-portfolio. Any other list is a plain text file with one URL per line "
                         "(or a CSV with a url column) that `python -m src.batch` reads directly.")
    years = parse_years(args.years) if args.years else None
    crawl = crawl_portfolio(cache=PageCache(read=not args.fresh))
    chosen = filter_cards(crawl.cards, args.country, years, tuple(args.exclude), args.limit)
    label = "_".join(x for x in (args.country, args.years) if x) or "all"
    out = Path(args.out or f"inputs/antler_{re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_').lower()}")
    written, skipped = write_outputs(chosen, out)
    print(f"Crawled {crawl.pages} pages, {len(crawl.cards)} companies. Matching the filter: {len(chosen)}; "
          f"written: {len(written)}; no company site tied to the card: {len(skipped)}.")
    no_country = sum(not c.country for c in crawl.cards)
    if no_country:
        print(f"  note: {no_country} cards show no country, so a country filter cannot match them")
    for problem in crawl.problems[:10]:
        print(f"  warning: {problem}")
    print(f"{out}/urls.csv, {out}/rows.json" + (f", {out}/skipped.txt" if skipped else ""))


if __name__ == "__main__":
    main()
