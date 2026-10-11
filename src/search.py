"""
search.py: client-side web search through Tavily's search API, called with the standard library.

Search results only supply URLs and snippets. A snippet is never evidence for a fact: the agent must fetch a
result with fetch_page before claiming anything from it. The snippets are stored with the run anyway, so that
verify.py can check that a snippet-only lead quotes its snippet exactly.

Tavily may use queries to improve its service (no opt-out below enterprise plans): queries contain founder and
company names, so only professional searches are made. Key: the line tavily_api_key (any case) in .env.

    search(query, site=None) -> SearchResponse     never raises; errors come back in SearchResponse.error
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from src.llm import read_env_key

TAVILY_URL = "https://api.tavily.com/search"
MAX_RESULTS = 5
TIMEOUT_SECONDS = 30
CREDIT_PRICE_USD = 0.008          # Tavily pay-as-you-go price per credit (Oct 2026); a basic search uses 1 credit
SNIPPET_CHARS = 1_000             # cap per stored snippet (Tavily's are about 500 characters at most)


@dataclass
class SearchHit:
    rank: int
    url: str
    title: str
    snippet: str


Site = Union[str, list[str], None]       # one domain or several (Tavily's include_domains)


def site_domains(site: Site) -> list[str]:
    return [site] if isinstance(site, str) and site else list(site or [])


@dataclass
class SearchResponse:
    query: str
    site: Site = None
    hits: list[SearchHit] = field(default_factory=list)
    credits: int = 0              # what the call was billed, for the cost guard (for a cached reply: what it cost once)
    error: Optional[str] = None
    cached: bool = False          # True when the reply came from the search cache: nothing was billed this time

    @property
    def cost_usd(self) -> float:
        return 0.0 if self.cached else self.credits * CREDIT_PRICE_USD


# Sends a JSON body to a URL and returns the decoded JSON reply. A parameter so tests never touch the network.
HttpPost = Callable[[str, dict, str], dict]


def _http_post(url: str, body: dict, key: str) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def search(query: str, site: Site = None, http_post: HttpPost = _http_post,
           key: Optional[str] = None) -> SearchResponse:
    """One basic-depth search. `site` restricts results to a domain or a list of domains (Tavily's include_domains)."""
    out = SearchResponse(query=query, site=site)
    body = {"query": query, "search_depth": "basic", "max_results": MAX_RESULTS,
            "include_answer": False, "include_raw_content": False}
    if site_domains(site):
        body["include_domains"] = site_domains(site)
    try:
        data = http_post(TAVILY_URL, body, key or read_env_key("TAVILY_API_KEY"))
    except urllib.error.HTTPError as e:
        e.close()
        out.error = f"HTTP {e.code}"
        return out
    except Exception as e:                                # network errors, timeouts, bad JSON, missing key
        out.error = f"{type(e).__name__}: {e}"
        return out
    usage = data.get("usage") or {}
    out.credits = int(usage.get("credits", 1))           # a basic search is 1 credit when the reply doesn't say
    for rank, r in enumerate(data.get("results") or [], start=1):
        if r.get("url"):
            out.hits.append(SearchHit(rank=rank, url=r["url"], title=(r.get("title") or "")[:300],
                                      snippet=(r.get("content") or "")[:SNIPPET_CHARS]))
    return out
