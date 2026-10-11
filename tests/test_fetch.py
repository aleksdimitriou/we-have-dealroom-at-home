"""
Tests for src/fetch.py with fake pages. No network: every test passes a fake http_get.
Run from the project root:  python -m unittest discover tests
"""

import unittest
import urllib.error

from schema import ActionOutcome
from src.fetch import HttpResponse, fetch_page, html_to_text

HTML_PAGE = """<!doctype html>
<html><head><title>  Torus &amp; Co </title>
<style>body { color: red; }</style>
<script>var secret = "not visible";</script>
</head>
<body>
  <nav>Home</nav>
  <h1>Torus</h1>
  <p>We build   AI for   grid operators.</p>
  <p>We&rsquo;re based in San Francisco.</p>
  <noscript>Enable JavaScript</noscript>
</body></html>"""

ALLOW_ALL = "User-agent: *\nAllow: /\n"


class FakeWeb:
    """Maps URL -> HttpResponse, or an exception to raise. Records every URL requested."""

    def __init__(self, pages: dict):
        self.pages = pages
        self.requested: list[str] = []

    def __call__(self, url: str) -> HttpResponse:
        self.requested.append(url)
        page = self.pages.get(url)
        if page is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        if isinstance(page, Exception):
            raise page
        return page


def html_response(url: str, body: str = HTML_PAGE, content_type: str = "text/html; charset=utf-8",
                  final_url: str | None = None, status: int = 200) -> HttpResponse:
    return HttpResponse(final_url=final_url or url, status=status, content_type=content_type,
                        body=body.encode("utf-8"), body_truncated=False)


def robots(body: str) -> HttpResponse:
    return HttpResponse(final_url="", status=200, content_type="text/plain", body=body.encode(), body_truncated=False)


class HtmlToTextTests(unittest.TestCase):
    def test_strips_scripts_styles_and_keeps_visible_text(self):
        title, text = html_to_text(HTML_PAGE)
        self.assertEqual(title, "Torus & Co")
        self.assertIn("We build AI for grid operators.", text)
        self.assertIn("We’re based in San Francisco.", text)       # entity decoded
        for hidden in ("color: red", "not visible", "Enable JavaScript", "Torus & Co"):
            self.assertNotIn(hidden, text)

    def test_svg_titles_do_not_leak_into_page_title_or_text(self):
        html = "<head><title>Real title</title></head><body><svg><title>Twitter</title></svg><p>Body</p></body>"
        title, text = html_to_text(html)
        self.assertEqual(title, "Real title")
        self.assertEqual(text, "Body")

    def test_outbound_links_are_absolute_deduplicated_and_http_only(self):
        from src.fetch import parse_html
        html = ('<a href="/team">Team</a> <a href="https://x.com/a#top">X</a> <a href="/team#b">again</a>'
                '<a href="mailto:a@b.c">mail</a> <a href="javascript:void(0)">js</a> <a href="sub/page">rel</a>')
        _, _, links = parse_html(html, base_url="https://www.example.com/about/")
        self.assertEqual(links, ["https://www.example.com/team", "https://x.com/a", "https://www.example.com/about/sub/page"])

    def test_block_tags_become_line_breaks(self):
        _, text = html_to_text("<p>one</p><p>two</p><span>three</span><span>four</span>")
        self.assertEqual(text.splitlines(), ["one", "two", "threefour"])


class FetchPageTests(unittest.TestCase):
    URL = "https://example.com/companies/torus"
    ROBOTS = "https://example.com/robots.txt"

    def test_ok_page(self):
        web = FakeWeb({self.ROBOTS: robots(ALLOW_ALL), self.URL: html_response(self.URL)})
        r = fetch_page(self.URL, http_get=web)
        self.assertEqual(r.outcome, ActionOutcome.OK)
        self.assertEqual((r.status, r.title), (200, "Torus & Co"))
        self.assertTrue(r.content_type.startswith("text/html"))
        self.assertIn("grid operators", r.text)
        self.assertFalse(r.truncated)
        self.assertEqual(r.links, [])
        self.assertEqual(web.requested, [self.ROBOTS, self.URL])     # robots.txt first

    def test_links_resolve_against_final_url_after_redirect(self):
        final = "https://www.example.com/new/"
        page = html_response(self.URL, body='<a href="team">Team</a>', final_url=final)
        web = FakeWeb({self.ROBOTS: robots(ALLOW_ALL), self.URL: page})
        self.assertEqual(fetch_page(self.URL, http_get=web).links, ["https://www.example.com/new/team"])

    def test_redirect_reports_final_url(self):
        final = "https://www.example.com/companies/torus/"
        web = FakeWeb({self.ROBOTS: robots(ALLOW_ALL), self.URL: html_response(self.URL, final_url=final)})
        self.assertEqual(fetch_page(self.URL, http_get=web).final_url, final)

    def test_robots_disallow_blocks_without_fetching_the_page(self):
        web = FakeWeb({self.ROBOTS: robots("User-agent: *\nDisallow: /companies\n"), self.URL: html_response(self.URL)})
        r = fetch_page(self.URL, http_get=web)
        self.assertEqual(r.outcome, ActionOutcome.ROBOTS_BLOCKED)
        self.assertEqual(web.requested, [self.ROBOTS])

    def test_robots_rule_for_our_user_agent(self):
        rules = "User-agent: FounderDossierAgent\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
        web = FakeWeb({self.ROBOTS: robots(rules), self.URL: html_response(self.URL)})
        self.assertEqual(fetch_page(self.URL, http_get=web).outcome, ActionOutcome.ROBOTS_BLOCKED)

    def test_missing_robots_txt_allows(self):
        web = FakeWeb({self.URL: html_response(self.URL)})     # robots.txt -> 404
        self.assertEqual(fetch_page(self.URL, http_get=web).outcome, ActionOutcome.OK)

    def test_robots_server_error_blocks(self):
        web = FakeWeb({self.ROBOTS: urllib.error.HTTPError(self.ROBOTS, 503, "Unavailable", {}, None),
                       self.URL: html_response(self.URL)})
        self.assertEqual(fetch_page(self.URL, http_get=web).outcome, ActionOutcome.ROBOTS_BLOCKED)

    def test_http_error_is_fetch_failed_with_status(self):
        web = FakeWeb({self.ROBOTS: robots(ALLOW_ALL)})          # page -> 404
        r = fetch_page(self.URL, http_get=web)
        self.assertEqual((r.outcome, r.status, r.error), (ActionOutcome.FETCH_FAILED, 404, "HTTP 404"))

    def test_network_error_is_fetch_failed(self):
        web = FakeWeb({self.ROBOTS: robots(ALLOW_ALL), self.URL: TimeoutError("timed out")})
        r = fetch_page(self.URL, http_get=web)
        self.assertEqual(r.outcome, ActionOutcome.FETCH_FAILED)
        self.assertIn("timed out", r.error)

    def test_unsupported_content_type(self):
        pdf = HttpResponse(final_url=self.URL, status=200, content_type="application/pdf", body=b"%PDF-1.4",
                           body_truncated=False)
        web = FakeWeb({self.ROBOTS: robots(ALLOW_ALL), self.URL: pdf})
        r = fetch_page(self.URL, http_get=web)
        self.assertEqual(r.outcome, ActionOutcome.FETCH_FAILED)
        self.assertIn("application/pdf", r.error)

    def test_plain_text_is_returned_as_is(self):
        web = FakeWeb({self.ROBOTS: robots(ALLOW_ALL),
                       self.URL: html_response(self.URL, body="line one\nline two", content_type="text/plain")})
        r = fetch_page(self.URL, http_get=web)
        self.assertEqual((r.text, r.title), ("line one\nline two", None))

    def test_text_cap_is_recorded(self):
        long_page = "<p>" + "word " * 1000 + "</p>"
        web = FakeWeb({self.ROBOTS: robots(ALLOW_ALL), self.URL: html_response(self.URL, body=long_page)})
        r = fetch_page(self.URL, http_get=web, max_text_chars=100)
        self.assertTrue(r.truncated)
        self.assertEqual(len(r.text), 100)
        self.assertGreater(r.text_chars_before_cap, 100)

    def test_download_cap_is_recorded(self):
        page = HttpResponse(final_url=self.URL, status=200, content_type="text/html", body=b"<p>partial",
                            body_truncated=True)
        web = FakeWeb({self.ROBOTS: robots(ALLOW_ALL), self.URL: page})
        self.assertTrue(fetch_page(self.URL, http_get=web).truncated)

    def test_unknown_charset_falls_back_to_utf8(self):
        web = FakeWeb({self.ROBOTS: robots(ALLOW_ALL),
                       self.URL: html_response(self.URL, content_type="text/html; charset=not-a-charset")})
        self.assertEqual(fetch_page(self.URL, http_get=web).outcome, ActionOutcome.OK)

    def test_non_http_scheme_is_refused(self):
        web = FakeWeb({})
        r = fetch_page("file:///etc/passwd", http_get=web)
        self.assertEqual(r.outcome, ActionOutcome.FETCH_FAILED)
        self.assertEqual(web.requested, [])


class CrawlDelayTests(unittest.TestCase):
    """A Crawl-delay in robots.txt spaces consecutive requests to the same host."""

    def setUp(self):
        from src.fetch import clear_politeness_state
        clear_politeness_state()
        self.now = 100.0
        self.slept: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(round(seconds, 3))
        self.now += seconds

    def fetch(self, web, url):
        return fetch_page(url, http_get=web, sleep=self.sleep, clock=self.clock)

    def test_the_second_request_to_a_host_waits_for_the_delay(self):
        web = FakeWeb({"https://a.example/robots.txt": robots("User-agent: *\nCrawl-delay: 10\n"),
                       "https://a.example/one": html_response("https://a.example/one"),
                       "https://a.example/two": html_response("https://a.example/two"),
                       "https://b.example/robots.txt": robots(ALLOW_ALL),
                       "https://b.example/x": html_response("https://b.example/x")})
        self.fetch(web, "https://a.example/one")
        self.assertEqual(self.slept, [])                              # the first request never waits
        self.now += 4
        self.fetch(web, "https://a.example/two")
        self.assertEqual(self.slept, [6.0])                           # 10 s since the first, 4 s have passed
        self.fetch(web, "https://b.example/x")
        self.assertEqual(self.slept, [6.0])                           # another host, no delay asked

    def test_no_delay_means_no_waiting_and_a_huge_delay_is_capped(self):
        from src.fetch import MAX_CRAWL_DELAY_SECONDS
        plain = FakeWeb({"https://c.example/robots.txt": robots(ALLOW_ALL), "https://c.example/1": html_response("https://c.example/1"),
                         "https://c.example/2": html_response("https://c.example/2")})
        self.fetch(plain, "https://c.example/1")
        self.fetch(plain, "https://c.example/2")
        self.assertEqual(self.slept, [])
        slow = FakeWeb({"https://d.example/robots.txt": robots("User-agent: *\nCrawl-delay: 600\n"),
                        "https://d.example/1": html_response("https://d.example/1"), "https://d.example/2": html_response("https://d.example/2")})
        self.fetch(slow, "https://d.example/1")
        self.fetch(slow, "https://d.example/2")
        self.assertEqual(self.slept, [float(MAX_CRAWL_DELAY_SECONDS)])

    def test_a_blocked_page_is_not_requested_and_does_not_count_as_a_request(self):
        web = FakeWeb({"https://e.example/robots.txt": robots("User-agent: *\nCrawl-delay: 10\nDisallow: /private\n"),
                       "https://e.example/ok": html_response("https://e.example/ok")})
        self.assertEqual(self.fetch(web, "https://e.example/private").outcome, ActionOutcome.ROBOTS_BLOCKED)
        self.fetch(web, "https://e.example/ok")
        self.assertEqual(self.slept, [])

    def test_repeated_user_agent_groups_are_merged(self):
        from src.fetch import merge_robots_groups
        from urllib.robotparser import RobotFileParser
        text = "User-agent: *\nDisallow: /admin\n\nUser-agent: *\nCrawl-delay: 10\n\nUser-agent: *\nDisallow: /portfolio/company/\n"
        parser = RobotFileParser()
        parser.parse(merge_robots_groups(text.splitlines()))
        self.assertEqual(parser.crawl_delay("FounderDossierAgent"), 10)
        self.assertFalse(parser.can_fetch("FounderDossierAgent", "https://x.example/portfolio/company/a"))
        self.assertTrue(parser.can_fetch("FounderDossierAgent", "https://x.example/portfolio"))


class SitemapTests(unittest.TestCase):
    URLSET = ('<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
              '<url><loc>https://s.example/en-GB/about</loc></url><url><loc>https://s.example/en-GB/press/</loc></url>'
              '<url><loc> https://s.example/blog/1 </loc></url></urlset>')

    def setUp(self):
        from src.fetch import clear_politeness_state
        clear_politeness_state()

    def urls(self, web):
        from src.fetch import fetch_sitemap_urls
        return fetch_sitemap_urls("https://s.example/", http_get=web, sleep=lambda s: None, clock=lambda: 0.0)

    def xml(self, body):
        return HttpResponse(final_url="", status=200, content_type="application/xml", body=body.encode(), body_truncated=False)

    def test_the_default_sitemap_is_read(self):
        web = FakeWeb({"https://s.example/robots.txt": robots(ALLOW_ALL), "https://s.example/sitemap.xml": self.xml(self.URLSET)})
        self.assertEqual(self.urls(web), ["https://s.example/en-GB/about", "https://s.example/en-GB/press/", "https://s.example/blog/1"])

    def test_a_sitemap_line_in_robots_txt_wins_and_an_index_is_followed_one_level(self):
        index = ('<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><sitemap><loc>https://s.example/pages.xml</loc></sitemap>'
                 '<sitemap><loc>https://s.example/posts.xml</loc></sitemap></sitemapindex>')
        web = FakeWeb({"https://s.example/robots.txt": robots("User-agent: *\nAllow: /\nSitemap: https://s.example/map-index.xml\n"),
                       "https://s.example/map-index.xml": self.xml(index),
                       "https://s.example/pages.xml": self.xml(self.URLSET),
                       "https://s.example/posts.xml": self.xml('<urlset><url><loc>https://s.example/post-1</loc></url></urlset>')})
        self.assertEqual(self.urls(web), ["https://s.example/en-GB/about", "https://s.example/en-GB/press/", "https://s.example/blog/1",
                                          "https://s.example/post-1"])

    def test_no_sitemap_a_blocked_sitemap_and_broken_xml_give_nothing(self):
        self.assertEqual(self.urls(FakeWeb({"https://s.example/robots.txt": robots(ALLOW_ALL)})), [])
        blocked = FakeWeb({"https://s.example/robots.txt": robots("User-agent: *\nDisallow: /sitemap.xml\n"),
                           "https://s.example/sitemap.xml": self.xml(self.URLSET)})
        self.assertEqual(self.urls(blocked), [])
        self.assertNotIn("https://s.example/sitemap.xml", blocked.requested)                  # robots.txt decided first
        broken = FakeWeb({"https://s.example/robots.txt": robots(ALLOW_ALL), "https://s.example/sitemap.xml": self.xml("<html>not xml")})
        self.assertEqual(self.urls(broken), [])
