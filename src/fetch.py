"""
fetch.py: client-side page fetch. Downloads one page and returns its visible text.

Standard library only. robots.txt is checked first; a disallowed page is never requested, and a Crawl-delay in
robots.txt is honoured: consecutive requests to the same host are at least that many seconds apart (capped at
MAX_CRAWL_DELAY_SECONDS so one slow site cannot stall a batch).
The HTTP call is a parameter (`http_get`) so tests can pass fake pages and never touch the network.
"""

from __future__ import annotations

import socket
import time
import urllib.error
import xml.etree.ElementTree as ET
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Callable, Optional
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

from schema import ActionOutcome

USER_AGENT_TOKEN = "FounderDossierAgent"
USER_AGENT = f"{USER_AGENT_TOKEN}/0.1 (research prototype; fetches single public pages; respects robots.txt)"
TIMEOUT_SECONDS = 15
MAX_DOWNLOAD_BYTES = 2_000_000      # stop reading the body after this
MAX_TEXT_CHARS = 100_000            # cap on the visible text returned
MAX_LINKS = 200                     # cap on outbound links returned
MAX_CRAWL_DELAY_SECONDS = 30        # a larger Crawl-delay is shortened to this
SUPPORTED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


@dataclass
class HttpResponse:
    final_url: str
    status: int
    content_type: str          # full header value, e.g. "text/html; charset=utf-8"
    body: bytes
    body_truncated: bool       # True if the body was longer than the download cap


# A function that GETs a URL. Raises urllib.error.HTTPError for 4xx/5xx and other exceptions on network errors.
HttpGet = Callable[[str], HttpResponse]


@dataclass
class FetchResult:
    requested_url: str
    outcome: ActionOutcome
    final_url: Optional[str] = None
    status: Optional[int] = None
    content_type: Optional[str] = None
    title: Optional[str] = None
    text: str = ""
    truncated: bool = False             # True if the body or the text hit a cap
    text_chars_before_cap: Optional[int] = None
    links: list[str] = field(default_factory=list)   # absolute http(s) links found on the page, in page order, deduplicated
    error: Optional[str] = None


def _http_get(url: str) -> HttpResponse:
    """The real network call. Follows redirects; reads at most MAX_DOWNLOAD_BYTES (+1 to detect overflow)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        body = response.read(MAX_DOWNLOAD_BYTES + 1)
        return HttpResponse(
            final_url=response.geturl(),
            status=response.status,
            content_type=response.headers.get("Content-Type", ""),
            body=body[:MAX_DOWNLOAD_BYTES],
            body_truncated=len(body) > MAX_DOWNLOAD_BYTES,
        )


# --------------------------------------------------------------------------
# robots.txt
# --------------------------------------------------------------------------

_robots_parsers: dict[str, tuple[Optional[RobotFileParser], bool]] = {}   # host -> (parser, usable)
_last_request: dict[str, float] = {}                      # host -> clock time of the last request we made to it


def clear_politeness_state() -> None:
    """Forget the robots.txt cache and the request times (tests, and a fresh batch)."""
    _robots_parsers.clear()
    _last_request.clear()


def merge_robots_groups(lines: list[str]) -> list[str]:
    """RFC 9309: groups for the same user-agent are merged. Python's parser keeps only the first "*" group, so a
    robots.txt that repeats "User-agent: *" (as many large sites do) would lose its later Disallow and Crawl-delay
    lines. Groups with the same set of user-agents are joined here before parsing."""
    groups: list[tuple[list[str], list[str]]] = []
    agents: list[str] = []
    rules: list[str] = []
    in_rules = False

    def close():
        nonlocal agents, rules, in_rules
        if agents:
            groups.append((agents, rules))
        agents, rules, in_rules = [], [], False

    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key == "user-agent":
            if in_rules:
                close()
            agents.append(value.strip())
        elif agents:
            in_rules = True
            rules.append(f"{key}: {value.strip()}")
    close()
    merged: dict[tuple[str, ...], list[str]] = {}
    for names, group_rules in groups:
        merged.setdefault(tuple(sorted(n.lower() for n in names)), []).extend(group_rules)
    out: list[str] = []
    for names, group_rules in merged.items():
        out += [f"User-agent: {n}" for n in names] + group_rules + [""]
    return out


def _robots(url: str, http_get: HttpGet) -> tuple[Optional[RobotFileParser], bool]:
    """The parsed robots.txt of the URL's host, and whether it could be read at all. Cached per host for the real
    network call only (a test's fake getter is never cached). RFC 9309: a 4xx robots.txt means no rules (everything
    allowed); a 5xx or an unreachable robots.txt means assume everything is disallowed."""
    parts = urlsplit(url)
    host = parts.netloc.lower()
    if http_get is _http_get and host in _robots_parsers:
        return _robots_parsers[host]
    robots_url = urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))
    try:
        response = http_get(robots_url)
        lines = response.body.decode("utf-8", errors="replace").splitlines()
        parser = RobotFileParser()
        parser.parse(merge_robots_groups(lines))
        answer: tuple[Optional[RobotFileParser], bool] = (parser, True)
    except urllib.error.HTTPError as e:
        e.close()
        answer = (None, 400 <= e.code < 500)                   # no robots.txt: no rules
    except Exception:
        answer = (None, False)
    if http_get is _http_get:
        _robots_parsers[host] = answer
    return answer


def _allowed(url: str, rules: tuple[Optional[RobotFileParser], bool]) -> bool:
    parser, usable = rules
    return usable if parser is None else parser.can_fetch(USER_AGENT_TOKEN, url)


def _delay(rules: tuple[Optional[RobotFileParser], bool]) -> float:
    delay = rules[0].crawl_delay(USER_AGENT_TOKEN) if rules[0] is not None else None
    return min(float(delay), MAX_CRAWL_DELAY_SECONDS) if delay else 0.0


def robots_allows(url: str, http_get: HttpGet = _http_get) -> bool:
    return _allowed(url, _robots(url, http_get))


def crawl_delay(url: str, http_get: HttpGet = _http_get) -> float:
    """Seconds robots.txt asks us to wait between requests to this host (0 if none), capped."""
    return _delay(_robots(url, http_get))


def _wait_for_turn(url: str, delay: float, sleep: Callable[[float], None], clock: Callable[[], float]) -> None:
    """Sleep until `delay` seconds have passed since our last request to this host, then record this request."""
    host = urlsplit(url).netloc.lower()
    if delay > 0 and host in _last_request:
        wait = _last_request[host] + delay - clock()
        if wait > 0:
            sleep(wait)
    _last_request[host] = clock()


# --------------------------------------------------------------------------
# sitemap.xml
# --------------------------------------------------------------------------

MAX_SITEMAPS = 3                    # the sitemap itself plus at most two children of a sitemap index
MAX_SITEMAP_URLS = 3000


def fetch_sitemap_urls(site_url: str, http_get: HttpGet = _http_get, sleep: Callable[[float], None] = time.sleep,
                       clock: Callable[[], float] = time.monotonic) -> list[str]:
    """The page URLs a site lists in its sitemap (the `Sitemap:` lines of robots.txt, else /sitemap.xml). robots.txt
    must allow the sitemap URL, the Crawl-delay is honoured, an index is followed one level (two children). Never
    raises: no sitemap, a blocked one or broken XML gives an empty list. Pages are not fetched here."""
    parts = urlsplit(site_url)
    origin = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
    rules = _robots(site_url, http_get)
    listed = list(rules[0].site_maps() or []) if rules[0] is not None else []
    queue = [u for u in listed if urlsplit(u).scheme in ("http", "https")][:2] or [origin + "/sitemap.xml"]
    urls: list[str] = []
    fetched = 0
    while queue and fetched < MAX_SITEMAPS and len(urls) < MAX_SITEMAP_URLS:
        sitemap = queue.pop(0)
        if not _allowed(sitemap, rules):
            continue
        _wait_for_turn(sitemap, _delay(rules), sleep, clock)
        fetched += 1
        try:
            body = http_get(sitemap).body
            root = ET.fromstring(body)
        except Exception:                                   # no sitemap, an error page, XML that does not parse
            continue
        locs = [(e.text or "").strip() for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "loc" and (e.text or "").strip()]
        if root.tag.rsplit("}", 1)[-1] == "sitemapindex":
            queue += locs[:MAX_SITEMAPS - 1]
        else:
            urls += locs
    return list(dict.fromkeys(urls))[:MAX_SITEMAP_URLS]


# --------------------------------------------------------------------------
# HTML -> visible text
# --------------------------------------------------------------------------

_HIDDEN_TAGS = {"script", "style", "noscript", "template", "svg", "head"}
_BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th",
               "section", "article", "header", "footer", "nav", "main", "aside", "blockquote", "pre",
               "table", "dt", "dd", "figcaption"}


class _VisibleTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)   # entities like &rsquo; are decoded
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.hidden_depth = 0
        self.in_title = False
        self.title_done = False   # only the first <title> counts; inline SVG icons carry their own <title>
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href.strip())
        if tag == "title":
            self.in_title = not self.title_done
        elif tag in _HIDDEN_TAGS:
            self.hidden_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            if self.in_title:
                self.title_done = True
            self.in_title = False
        elif tag in _HIDDEN_TAGS:
            self.hidden_depth = max(0, self.hidden_depth - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        elif self.hidden_depth == 0:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[Optional[str], str]:
    """Returns (title, visible text). Lines are kept; spaces within a line are collapsed; blank lines removed."""
    title, text, _ = parse_html(html, base_url="")
    return title, text


def parse_html(html: str, base_url: str) -> tuple[Optional[str], str, list[str]]:
    """Returns (title, visible text, outbound links). Links are made absolute against base_url,
    stripped of #fragments, limited to http(s), deduplicated in page order, capped at MAX_LINKS."""
    parser = _VisibleTextParser()
    parser.feed(html)
    parser.close()
    title = " ".join("".join(parser.title_parts).split()) or None
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    text = "\n".join(line for line in lines if line)

    links: list[str] = []
    seen: set[str] = set()
    for href in parser.hrefs:
        absolute, _ = urldefrag(urljoin(base_url, href))
        if urlsplit(absolute).scheme in ("http", "https") and absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return title, text, links[:MAX_LINKS]


def _charset(content_type: str) -> str:
    for piece in content_type.split(";")[1:]:
        name, _, value = piece.strip().partition("=")
        if name.lower() == "charset" and value:
            return value.strip("\"'")
    return "utf-8"


# --------------------------------------------------------------------------
# fetch_page
# --------------------------------------------------------------------------

def fetch_page(url: str, http_get: HttpGet = _http_get, max_text_chars: int = MAX_TEXT_CHARS,
               sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> FetchResult:
    """Fetch one page. Never raises: failures come back as outcome fetch_failed or robots_blocked."""
    result = FetchResult(requested_url=url, outcome=ActionOutcome.FETCH_FAILED)

    if urlsplit(url).scheme not in ("http", "https"):
        result.error = "only http and https URLs are fetched"
        return result

    rules = _robots(url, http_get)                       # one robots.txt request, used for the check and the delay
    if not _allowed(url, rules):
        result.outcome = ActionOutcome.ROBOTS_BLOCKED
        result.error = "disallowed by robots.txt (or robots.txt unreachable)"
        return result

    _wait_for_turn(url, _delay(rules), sleep, clock)
    try:
        response = http_get(url)
    except urllib.error.HTTPError as e:
        e.close()
        result.status = e.code
        result.final_url = e.geturl() if hasattr(e, "geturl") else None
        result.error = f"HTTP {e.code}"
        return result
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as e:
        result.error = f"network error: {e}"
        return result

    result.final_url = response.final_url
    result.status = response.status
    result.content_type = response.content_type

    media_type = response.content_type.split(";")[0].strip().lower()
    if media_type not in SUPPORTED_CONTENT_TYPES:
        result.error = f"unsupported content type: {media_type or 'none'}"
        return result

    try:
        decoded = response.body.decode(_charset(response.content_type), errors="replace")
    except LookupError:   # unknown charset name in the header
        decoded = response.body.decode("utf-8", errors="replace")
    if media_type == "text/plain":
        title, text = None, decoded
    else:
        title, text, result.links = parse_html(decoded, base_url=response.final_url)

    result.title = title
    result.text_chars_before_cap = len(text)
    result.truncated = response.body_truncated or len(text) > max_text_chars
    result.text = text[:max_text_chars]
    result.outcome = ActionOutcome.OK
    return result
