"""
cache.py: successful page fetches and search replies kept on disk, so re-running a company reuses the pages and
searches it already paid for instead of downloading or buying them again.

One JSON file per URL under runs/_cache/ (gitignored, like the rest of runs/: it holds page text), named by a hash
of the normalised URL. Only successful fetches are stored; failed ones are retried next time. Entries older than
MAX_AGE_DAYS are ignored, because pages change. A reused page still counts as a research action in the run, exactly
like a live fetch, so what the analyst sees does not change; the run's events say which pages came from the cache.
Search replies are kept the same way (runs/_cache/search/, keyed by query and site): a reused reply counts as a search
in the run but is not billed again.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from schema import ActionOutcome
from src.fetch import FetchResult
from src.search import SearchHit, SearchResponse, Site, site_domains
from src.verify import normalise_url

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "runs" / "_cache"
MAX_AGE_DAYS = 7


class PageCache:
    def __init__(self, directory: Path = CACHE_DIR, max_age_days: int = MAX_AGE_DAYS, read: bool = True,
                 now: Callable[[], datetime] = datetime.now):
        self.directory, self.max_age, self.read, self.now = directory, timedelta(days=max_age_days), read, now

    def _path(self, url: str) -> Path:
        return self.directory / (hashlib.sha256(normalise_url(url).encode("utf-8")).hexdigest() + ".json")

    def get(self, url: str) -> Optional[tuple[FetchResult, datetime]]:
        """The stored result and when it was fetched, or None if absent, too old, or reading is off (--fresh)."""
        path = self._path(url)
        if not self.read or not path.exists():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(entry["fetched_at"])
        if self.now() - fetched_at > self.max_age:
            return None
        result = entry["result"]
        result["outcome"] = ActionOutcome(result["outcome"])
        return FetchResult(**result), fetched_at

    def put(self, url: str, result: FetchResult, fetched_at: datetime) -> None:
        if result.outcome != ActionOutcome.OK:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        stored = asdict(result)
        stored["outcome"] = result.outcome.value
        self._path(url).write_text(json.dumps({"url": url, "fetched_at": fetched_at.isoformat(), "result": stored},
                                              ensure_ascii=False), encoding="utf-8")


class SearchCache:
    """Successful search replies on disk, same rules as PageCache: a failed search is retried next time."""

    def __init__(self, directory: Path = CACHE_DIR / "search", max_age_days: int = MAX_AGE_DAYS, read: bool = True,
                 now: Callable[[], datetime] = datetime.now):
        self.directory, self.max_age, self.read, self.now = directory, timedelta(days=max_age_days), read, now

    def _path(self, query: str, site: Site) -> Path:
        key = " ".join(query.lower().split()) + "|" + ",".join(sorted(d.lower() for d in site_domains(site)))
        return self.directory / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json")

    def get(self, query: str, site: Site = None) -> Optional[SearchResponse]:
        path = self._path(query, site)
        if not self.read or not path.exists():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        if self.now() - datetime.fromisoformat(entry["stored_at"]) > self.max_age:
            return None
        return SearchResponse(query=query, site=site, credits=entry["credits"], cached=True,
                              hits=[SearchHit(**h) for h in entry["hits"]])

    def put(self, response: SearchResponse) -> None:
        if response.error or response.cached:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(response.query, response.site).write_text(json.dumps({
            "query": response.query, "site": response.site, "stored_at": self.now().isoformat(),
            "credits": response.credits, "hits": [asdict(h) for h in response.hits]}, ensure_ascii=False),
            encoding="utf-8")


def cached_search(cache: SearchCache, inner: Callable[..., SearchResponse]) -> Callable[[str, Site], SearchResponse]:
    """A search function that answers from the cache when it can and stores every successful live reply."""
    def search_with_cache(query: str, site: Site = None) -> SearchResponse:
        hit = cache.get(query, site)
        if hit:
            return hit
        response = inner(query, site)
        cache.put(response)
        return response
    return search_with_cache
