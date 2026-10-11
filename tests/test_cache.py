"""
Tests for the page cache in src/cache.py (the workflow's use of it, and the search cache, are in test_agent.py).
Run from the project root:  python -m unittest discover tests
"""

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from schema import ActionOutcome
from src.cache import PageCache
from src.fetch import FetchResult

T0 = datetime(2026, 10, 6, 12, 0)
OK_PAGE = FetchResult(requested_url="https://a.com", outcome=ActionOutcome.OK, final_url="https://a.com/",
                      status=200, title="A", text="Hello", links=["https://a.com/team"])


class PageCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_and_url_normalisation(self):
        PageCache(self.dir).put("https://a.com", OK_PAGE, T0)
        hit = PageCache(self.dir, now=lambda: T0 + timedelta(days=1)).get("http://www.a.com/")
        self.assertEqual(hit, (OK_PAGE, T0))

    def test_stale_entries_are_ignored(self):
        PageCache(self.dir).put("https://a.com", OK_PAGE, T0)
        self.assertIsNone(PageCache(self.dir, now=lambda: T0 + timedelta(days=8)).get("https://a.com"))

    def test_failed_fetches_are_not_stored(self):
        failed = FetchResult(requested_url="https://a.com", outcome=ActionOutcome.FETCH_FAILED, error="HTTP 503")
        PageCache(self.dir).put("https://a.com", failed, T0)
        self.assertIsNone(PageCache(self.dir).get("https://a.com"))

    def test_fresh_mode_does_not_read_but_still_writes(self):
        PageCache(self.dir, read=False).put("https://a.com", OK_PAGE, T0)
        self.assertIsNone(PageCache(self.dir, read=False).get("https://a.com"))
        self.assertIsNotNone(PageCache(self.dir, now=lambda: T0).get("https://a.com"))
