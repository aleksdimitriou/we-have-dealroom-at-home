"""
Tests for src/antler.py (the Antler portfolio filter) with fake pages in the portfolio page's text format.
No network. Run from the project root:  python -m unittest discover tests
"""

import json
import tempfile
import unittest
from pathlib import Path

from schema import ActionOutcome
from src.antler import (Card, assign_sites, crawl_portfolio, filter_cards, matches, parse_page_cards, parse_years,
                        site_links, write_outputs)
from src.fetch import FetchResult
from src.intake import IntakeError, load_rows, read_entries

FILTER = """Skip to main content
BROWSE OUR
—PORTFOLIO
Search
Location
All
Germany
Singapore
US
Sector
All
FinTech
B2B Software
Year
All
2026
2025
Showing 0 of 100
Tag
"""


def portfolio_text(cards, tail="Previous\nLoad more\nNo results found."):
    """Cards as the page writes them: lines name, description, country, sector, year (any of them may be left out)."""
    return FILTER + "\n".join("\n".join(lines) for lines in cards) + "\n" + tail + "\nBUILD YOUR STARTUP"


def page(url, text, links=()):
    return FetchResult(requested_url=url, outcome=ActionOutcome.OK, final_url=url, status=200, text=text, links=list(links))


CARDS_1 = [
    ["Aiden", "AI for sales", "Germany", "B2B Software", "2024"],
    ["Kinetic", "Zero-party data", "Germany", "B2B Software", "2025"],
    ["Meaningful CX", "Customer experience", "Germany", "B2B Software", "2026"],
    ["Old Co", "Old", "Germany", "FinTech", "2021"],
    ["Singa", "Payments", "Singapore", "FinTech", "2025"],
]
LINKS_1 = ["https://www.antler.co/", "https://getaiden.com", "https://www.usekinetic.com/", "https://meaningful.app",
           "https://oldco.example", "https://singa.example", "https://www.antler.co/portfolio?abc_page=2",
           "https://www.linkedin.com/company/antlerglobal/"]


class ParseTests(unittest.TestCase):
    def test_cards_are_read_in_order_with_their_fields(self):
        cards, problems = parse_page_cards(portfolio_text(CARDS_1))
        self.assertEqual(problems, [])
        self.assertEqual([c.name for c in cards], ["Aiden", "Kinetic", "Meaningful CX", "Old Co", "Singa"])
        self.assertEqual((cards[1].description, cards[1].country, cards[1].sector, cards[1].year),
                         ("Zero-party data", "Germany", "B2B Software", 2025))

    def test_a_card_without_a_year_and_a_country_the_filter_does_not_list_are_read(self):
        cards, _ = parse_page_cards(portfolio_text([["nolemon", "Repair shops", "Spain", "B2B Software"],
                                                    ["NoMa", "Buses", "Kenya", "B2B Software", "2023"]]))
        self.assertEqual([(c.name, c.country, c.year) for c in cards], [("nolemon", "Spain", None), ("NoMa", "Kenya", 2023)])

    def test_a_card_missing_its_description_or_its_country_is_still_a_card(self):
        cards, problems = parse_page_cards(portfolio_text([["phablo.ai", "Singapore", "B2B Software", "2025"],
                                                           ["Paismo", "Redefining work.", "B2B Software", "2023"]]))
        self.assertEqual(problems, [])
        self.assertEqual([(c.name, c.description, c.country) for c in cards],
                         [("phablo.ai", "", "Singapore"), ("Paismo", "Redefining work.", "")])

    def test_a_card_with_a_year_but_no_sector_and_a_card_with_only_a_name_are_cards(self):
        cards, problems = parse_page_cards(portfolio_text([["Unison", "Team workspace", "Germany", "2026"],
                                                           ["Unplex", "Build things", "Germany", "FinTech", "2025"],
                                                           ["HeyAmos", "B2B Software"]]))
        self.assertEqual(problems, [])
        self.assertEqual([(c.name, c.sector, c.year) for c in cards],
                         [("Unison", "", 2026), ("Unplex", "FinTech", 2025), ("HeyAmos", "B2B Software", None)])
        self.assertEqual(cards[2].country, "")

    def test_a_page_without_cards_and_a_block_that_is_not_a_card_are_reported(self):
        self.assertEqual(parse_page_cards("nothing here")[0], [])
        cards, problems = parse_page_cards(portfolio_text([["A", "b", "c", "d", "e", "f", "B2B Software"]]))
        self.assertEqual(cards, [])
        self.assertEqual(len(problems), 1)

    def test_the_featured_block_and_the_footer_are_not_cards(self):
        text = "Featured\nLovable\nAn app builder\nSweden\n" + portfolio_text(CARDS_1[:1])
        cards, _ = parse_page_cards(text)
        self.assertEqual([c.name for c in cards], ["Aiden"])


class SiteTests(unittest.TestCase):
    def test_a_site_must_carry_the_cards_name(self):
        self.assertTrue(matches("Kinetic", "https://www.usekinetic.com/"))
        self.assertTrue(matches("Meaningful CX", "https://meaningful.app"))
        self.assertTrue(matches("Papayo.ai", "https://papayo.ai"))
        self.assertTrue(matches("PCB Labs", "https://pcblabs.com"))
        self.assertTrue(matches("Altum (fka Jenni.ai)", "https://jenni.ai/?utm_source=x"))
        self.assertFalse(matches("Leetify", "https://lemonado.io/"))

    def test_links_are_matched_in_order_and_an_unmatched_card_is_filled_from_its_neighbours_by_position(self):
        cards = [Card(n, "", "Germany", "FinTech", 2025, "") for n in ("Alpha", "Zeta Corp", "Omega")]
        assign_sites(cards, ["https://alpha.example", "https://totally-different.example", "https://omega.example"])
        self.assertEqual([c.site for c in cards], ["https://alpha.example", "https://totally-different.example", "https://omega.example"])

    def test_an_unmatched_card_with_no_certain_link_stays_empty(self):
        cards = [Card(n, "", "Germany", "FinTech", 2025, "") for n in ("Alpha", "Zeta Corp", "Omega")]
        assign_sites(cards, ["https://alpha.example", "https://omega.example"])
        self.assertEqual([c.site for c in cards], ["https://alpha.example", "", "https://omega.example"])

    def test_antler_social_and_login_links_are_not_sites(self):
        self.assertEqual(site_links(LINKS_1), ["https://getaiden.com", "https://www.usekinetic.com/", "https://meaningful.app",
                                              "https://oldco.example", "https://singa.example"])


class FilterTests(unittest.TestCase):
    def cards(self):
        cards, _ = parse_page_cards(portfolio_text(CARDS_1))
        assign_sites(cards, site_links(LINKS_1))
        return cards

    def test_years_are_a_range(self):
        self.assertEqual(parse_years("2025-2026"), (2025, 2026))
        self.assertEqual(parse_years("2026"), (2026, 2026))
        self.assertEqual(parse_years("2024,2026"), (2024, 2026))
        with self.assertRaises(ValueError):
            parse_years("recent")

    def test_country_years_exclude_and_limit(self):
        cards = self.cards()
        names = lambda cs: [c.name for c in cs]
        self.assertEqual(names(filter_cards(cards, "Germany", (2024, 2026))), ["Aiden", "Kinetic", "Meaningful CX"])
        self.assertEqual(names(filter_cards(cards, "germany", (2025, 2026), exclude=("kinetic",))), ["Meaningful CX"])
        self.assertEqual(names(filter_cards(cards, "Germany", None, limit=2)), ["Aiden", "Kinetic"])
        self.assertEqual(names(filter_cards(cards, None, (2025, 2025))), ["Kinetic", "Singa"])

    def test_a_card_with_no_year_never_matches_a_year_filter(self):
        card = Card("Yearless", "", "Germany", "FinTech", None, "")
        self.assertEqual(filter_cards([card], "Germany", (2020, 2026)), [])
        self.assertEqual(filter_cards([card], "Germany", None), [card])


class CrawlTests(unittest.TestCase):
    URL = "https://www.antler.co/portfolio"
    P2 = "https://www.antler.co/portfolio?abc_page=2"
    P3 = "https://www.antler.co/portfolio?abc_page=3"

    def pages(self):
        second = [["Zed", "Last", "Germany", "FinTech", "2026"], ["Aiden", "again", "Germany", "B2B Software", "2024"]]
        return {self.URL: page(self.URL, portfolio_text(CARDS_1), LINKS_1),
                self.P2: page(self.P2, portfolio_text(second), ["https://zed.example", "https://getaiden.com", self.URL, self.P3]),
                self.P3: page(self.P3, portfolio_text([["Third", "x", "US", "FinTech", "2025"]]), ["https://third.example", self.P2])}

    def test_the_pages_are_followed_in_order_and_duplicates_are_dropped(self):
        pages, fetched = self.pages(), []

        def fetch(url):
            fetched.append(url)
            return pages[url]

        log = []
        crawl = crawl_portfolio(fetch=fetch, log=log.append)
        self.assertEqual(fetched, [self.URL, self.P2, self.P3])
        self.assertEqual(crawl.pages, 3)
        self.assertEqual([c.name for c in crawl.cards], ["Aiden", "Kinetic", "Meaningful CX", "Old Co", "Singa", "Zed", "Third"])
        self.assertEqual(crawl.cards[0].page_url, self.URL)
        self.assertEqual(crawl.cards[5].page_url, self.P2)
        self.assertEqual(crawl.cards[5].site, "https://zed.example")
        self.assertEqual(len(log), 3)

    def test_a_page_that_cannot_be_fetched_stops_the_crawl(self):
        failing = FetchResult(requested_url=self.URL, outcome=ActionOutcome.ROBOTS_BLOCKED, error="disallowed by robots.txt")
        with self.assertRaises(IntakeError):
            crawl_portfolio(fetch=lambda url: failing, log=lambda _: None)

    def test_a_loop_in_the_pagination_ends(self):
        loop = page(self.URL, portfolio_text(CARDS_1), [*LINKS_1, self.URL + "?abc_page=1"])
        crawl = crawl_portfolio(fetch=lambda url: loop if url == self.URL else page(url, portfolio_text([]), [self.URL + "?abc_page=1"]),
                                log=lambda _: None)
        self.assertEqual(crawl.pages, 2)                  # the base page and ?abc_page=2 once; ?abc_page=1 is the base page


class OutputTests(unittest.TestCase):
    def test_urls_csv_is_readable_as_an_input_and_rows_json_as_rows(self):
        cards, _ = parse_page_cards(portfolio_text(CARDS_1))
        assign_sites(cards, site_links(LINKS_1))
        for c in cards:
            c.page_url = "https://www.antler.co/portfolio?abc_page=7"
        cards.append(Card("No Site", "x", "Germany", "FinTech", 2025, "https://www.antler.co/portfolio?abc_page=7"))
        chosen = filter_cards(cards, "Germany", (2024, 2026))
        with tempfile.TemporaryDirectory() as tmp:
            written, skipped = write_outputs(chosen, Path(tmp))
            self.assertEqual([c.name for c in written], ["Aiden", "Kinetic", "Meaningful CX"])
            self.assertEqual([c.name for c in skipped], ["No Site"])
            entries = read_entries(Path(tmp) / "urls.csv")
            self.assertEqual([(e.name, e.url) for e in entries],
                             [("Aiden", "https://getaiden.com"), ("Kinetic", "https://www.usekinetic.com/"), ("Meaningful CX", "https://meaningful.app")])
            self.assertEqual(entries[0].accelerator_url, "https://www.antler.co/portfolio?abc_page=7")
            rows = load_rows(Path(tmp) / "rows.json")
            self.assertEqual([r.row.company_name for r in rows], ["Aiden", "Kinetic", "Meaningful CX"])
            self.assertEqual(rows[1].row.company_url, "https://www.usekinetic.com/")
            self.assertEqual(rows[1].row.accelerator_profile_url, "https://www.antler.co/portfolio?abc_page=7")
            self.assertEqual(rows[1].row.founder_names, [])                              # discovery finds the founders
            self.assertIn("No Site", (Path(tmp) / "skipped.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
