"""
Tests for src/openalex.py with fake HTTP. No network. Run from the project root:  python -m unittest discover tests
"""

import json
import unittest

from src.openalex import (
    MAX_REQUESTS, SECTOR_FIELDS, Work, allowed_fields, find_works, same_person_name, search_query, top_works,
)
from schema import SECTOR_TAGS


def author(aid, name, works_count, institutions=(), alternatives=()):
    return {"id": f"https://openalex.org/{aid}", "display_name": name, "display_name_alternatives": list(alternatives),
            "works_count": works_count, "affiliations": [{"institution": {"display_name": i}} for i in institutions],
            "last_known_institutions": []}


def work(title, cited, date, authors=("Jane Doe",), venue="Journal X", field="Computer Science"):
    return {"id": "https://openalex.org/W1", "display_name": title, "publication_year": int(date[:4]),
            "publication_date": date, "cited_by_count": cited, "doi": "https://doi.org/10.1/x",
            "primary_location": {"source": {"display_name": venue}},
            "primary_topic": {"field": {"display_name": field}},
            "authorships": [{"author": {"display_name": a}} for a in authors]}


class FakeApi:
    def __init__(self, authors, works_by_author):
        self.authors, self.works_by_author, self.urls = authors, works_by_author, []

    def __call__(self, url):
        self.urls.append(url)
        if "/authors?" in url:
            return {"results": self.authors}
        for aid, works in self.works_by_author.items():
            if f"authorships.author.id:{aid}" in url:
                return {"results": works}
        return {"results": []}


PAPERS = [work("Old paper", 5, "2019-01-01"), work("Big paper", 90, "2020-01-01"), work("New paper", 5, "2024-01-01"),
          work("Mid paper", 30, "2021-01-01"), work("Tiny", 0, "2025-01-01")]


class OpenAlexTests(unittest.TestCase):
    def test_names(self):
        self.assertTrue(same_person_name("J. Doe", "Jane Doe"))
        self.assertTrue(same_person_name("Jane Q. Doe", "Jane Doe"))
        self.assertFalse(same_person_name("John Doe", "Jane Doe"))
        self.assertFalse(same_person_name("Jane Smith", "Jane Doe"))

    def test_two_affiliations_are_two_signals_and_the_works_are_the_top_three(self):
        api = FakeApi([author("A1", "Jane Doe", 40, ["The University of Texas at Austin", "Los Alamos National Laboratory"]),
                       author("A2", "Jane Doe", 900, ["Other University"])], {"A1": PAPERS})
        r = find_works("Jane Doe", ["UT Austin", "Los Alamos National Lab"], ["John Roe"], http_get=api, key="SECRET")
        self.assertEqual((r.accepted_by, r.author_id, r.candidates), ("affiliation+second_affiliation", "A1", 2))
        self.assertEqual([w.title for w in r.works], ["Big paper", "Mid paper", "New paper"])   # most cited, then most recent
        self.assertFalse(any("SECRET" in u for u in r.requests))                                 # the key is never logged
        self.assertTrue(all("api_key=SECRET" in u for u in api.urls))                            # but it is sent
        self.assertEqual(r.cost_usd, 0)
        text = json.loads(r.source_text)
        self.assertEqual(text["works"][0]["title"], "Big paper")
        self.assertEqual(text["works"][0]["field"], "Computer Science")
        self.assertEqual(text["accepted_by"], "affiliation+second_affiliation")

    def test_an_affiliation_plus_a_coauthor_or_plus_topic_fit_is_enough(self):
        institutions = ["UT Austin"]
        co = FakeApi([author("A1", "Jane Doe", 40, ["The University of Texas at Austin"])],
                     {"A1": [work("Joint paper", 10, "2022-01-01", authors=("Jane Doe", "J. Roe"), field="Chemistry")]})
        r = find_works("Jane Doe", institutions, ["John Roe"], http_get=co, key="k", fields=("Engineering",))
        self.assertEqual(r.accepted_by, "affiliation+coauthor")
        topic = FakeApi([author("A1", "Jane Doe", 40, ["The University of Texas at Austin"])],
                        {"A1": [work(f"P{i}", i, "2022-01-01", field="Engineering") for i in range(3)]})
        r = find_works("Jane Doe", institutions, ["John Roe"], http_get=topic, key="k", fields=("Engineering", "Energy"))
        self.assertEqual(r.accepted_by, "affiliation+topic")

    def test_a_coauthor_and_topic_fit_are_enough_without_any_affiliation(self):
        api = FakeApi([author("A1", "Jane Doe", 40, ["Somewhere Else"])],
                      {"A1": [work(f"P{i}", i, "2022-01-01", authors=("Jane Doe", "John Roe"), field="Engineering") for i in range(3)]})
        r = find_works("Jane Doe", ["UT Austin"], ["John Roe"], http_get=api, key="k", fields=("Engineering",))
        self.assertEqual(r.accepted_by, "coauthor+topic")

    def test_one_signal_is_never_enough(self):
        # one overlap with a large institution, works in another field, no co-author: the namesake case
        api = FakeApi([author("A1", "Jane Doe", 11, ["University of Oxford", "Hunan Agricultural University"])],
                      {"A1": [work(f"Biochar {i}", i, "2024-01-01", field="Environmental Science") for i in range(3)]})
        r = find_works("Jane Doe", ["University of Oxford"], ["John Roe"], http_get=api, key="k", fields=("Computer Science", "Engineering"))
        self.assertEqual((r.accepted_by, r.works, r.source_text, r.no_overlap), ("", [], "", 1))
        self.assertEqual(r.decisions, [("Jane Doe", ["affiliation"], False)])
        # a co-author alone, topic alone: also rejected
        co_only = FakeApi([author("A1", "Jane Doe", 3, ["Elsewhere"])], {"A1": [work("P", 1, "2022-01-01", authors=("Jane Doe", "John Roe"), field="Chemistry")]})
        self.assertEqual(find_works("Jane Doe", ["UT Austin"], ["John Roe"], http_get=co_only, key="k", fields=("Engineering",)).decisions,
                         [("Jane Doe", ["coauthor"], False)])
        topic_only = FakeApi([author("A1", "Jane Doe", 3, ["Elsewhere"])], {"A1": [work("P", 1, "2022-01-01", field="Engineering")]})
        self.assertEqual(find_works("Jane Doe", ["UT Austin"], ["John Roe"], http_get=topic_only, key="k", fields=("Engineering",)).decisions,
                         [("Jane Doe", ["topic"], False)])

    def test_topic_needs_most_of_the_kept_works(self):
        api = FakeApi([author("A1", "Jane Doe", 3, ["The University of Texas at Austin"])],
                      {"A1": [work("A", 9, "2022-01-01", field="Engineering"), work("B", 8, "2022-01-01", field="Chemistry"),
                              work("C", 7, "2022-01-01", field="Chemistry")]})
        r = find_works("Jane Doe", ["UT Austin"], [], http_get=api, key="k", fields=("Engineering",))
        self.assertEqual(r.decisions[0][1], ["affiliation"])                 # 1 of 3 is not most

    def test_the_search_stops_fetching_works_once_an_author_is_accepted(self):
        api = FakeApi([author("A1", "Jane Doe", 40, ["UT Austin", "Los Alamos National Laboratory"]),
                       author("A2", "Jane Doe", 30, ["UT Austin"])], {"A1": PAPERS, "A2": PAPERS})
        r = find_works("Jane Doe", ["UT Austin", "Los Alamos National Lab"], [], http_get=api, key="k")
        self.assertEqual(len(r.requests), 2)                                 # the search and one works request

    def test_the_request_cap_and_errors(self):
        authors = [author(f"A{i}", "Jane Doe", i, ["Elsewhere"]) for i in range(5)]
        api = FakeApi(authors, {})
        r = find_works("Jane Doe", ["UT Austin"], ["John Roe"], http_get=api, key="k", max_requests=3)
        self.assertEqual(len(r.requests), 3)
        self.assertLessEqual(MAX_REQUESTS, 5)
        full = find_works("Jane Doe", ["UT Austin"], ["John Roe"], http_get=FakeApi(authors, {}), key="k", aliases=("J Doe",))
        self.assertLessEqual(len(full.requests), 5)                          # two searches and at most three works requests

        def broken(url):
            raise TimeoutError("timed out")

        r = find_works("Jane Doe", [], [], http_get=broken, key="k")
        self.assertIn("timed out", r.error)
        self.assertEqual((r.works, len(r.requests)), ([], 1))

    def test_initials_are_searched_without_dots(self):
        self.assertEqual(search_query("A.B. Carter"), "A B Carter")
        self.assertEqual(search_query("Jane Doe"), "Jane Doe")
        api = FakeApi([author("A1", "A. B. Carter", 3, ["UC Berkeley", "Rockwell Automation"])],
                      {"A1": [work("A paper", 1, "2020-01-01", authors=("A. B. Carter",))]})
        r = find_works("A.B. Carter", ["University of California, Berkeley", "Rockwell Automation"], [], http_get=api, key="k")
        self.assertIn("search=A+B+Carter", r.requests[0])
        self.assertEqual(r.accepted_by, "affiliation+second_affiliation")

    def test_an_alias_is_searched_too_and_the_rule_is_unchanged(self):
        class ByQuery(FakeApi):
            def __call__(self, url):
                self.urls.append(url)
                if "/authors?" in url:
                    return {"results": self.authors} if "search=Maya+Lindqvist" in url else {"results": []}
                return super().__call__(url)

        papers = [work(f"P{i}", i, "2020-01-01", authors=("Maya Lindqvist",), field="Computer Science") for i in range(3)]
        api = ByQuery([author("A1", "Maya Lindqvist", 11, ["University of Oxford"])], {"A1": papers})
        r = find_works("Mira Lindqvist", ["Oxford"], [], http_get=api, key="k", aliases=("Maya Lindqvist",), fields=("Computer Science",))
        self.assertEqual((r.accepted_by, r.queries, len(r.works)), ("affiliation+topic", 2, 3))
        # the alias gives a name match, not an acceptance: one overlap and no topic fit is still rejected
        api2 = ByQuery([author("A1", "Maya Lindqvist", 11, ["University of Oxford"])],
                       {"A1": [work(f"P{i}", i, "2020-01-01", authors=("Maya Lindqvist",), field="Chemistry") for i in range(3)]})
        r2 = find_works("Mira Lindqvist", ["Oxford"], [], http_get=api2, key="k", aliases=("Maya Lindqvist",), fields=("Computer Science",))
        self.assertEqual((r2.accepted_by, r2.no_overlap), ("", 1))

    def test_name_variants_from_openalex_alternatives(self):
        api = FakeApi([author("A1", "Maya Lindqvist", 12, ["MIT", "Google"], alternatives=["Mira Lindqvist"])],
                      {"A1": [work("A paper", 3, "2023-01-01", authors=("Maya Lindqvist",))]})
        r = find_works("Mira Lindqvist", ["Massachusetts Institute of Technology", "Google"], [], http_get=api, key="k")
        self.assertEqual(r.accepted_by, "affiliation+second_affiliation")

    def test_top_works_tiebreak(self):
        ws = [Work("a", 2020, "2020-01-01", 5, "", "", []), Work("b", 2024, "2024-01-01", 5, "", "", []), Work("c", 2019, "2019-01-01", 9, "", "", [])]
        self.assertEqual([w.title for w in top_works(ws, 2)], ["c", "b"])


class FieldTableTests(unittest.TestCase):
    def test_fields_come_from_fixed_tables(self):
        self.assertEqual(allowed_fields(["PhD Candidate in Computer Science, University of Oxford"], []), ("Computer Science",))
        self.assertIn("Engineering", allowed_fields(["MS in Robotics from UT Austin"], []))
        self.assertEqual(allowed_fields([], ["aerospace_defense"]), SECTOR_FIELDS["aerospace_defense"])
        self.assertEqual(allowed_fields(["Nothing relevant"], []), ())
        self.assertEqual(allowed_fields(["Engineering"], ["industrial_hardware"]).count("Engineering"), 1)   # no duplicates

    def test_every_sector_tag_has_fields(self):
        self.assertEqual(set(SECTOR_FIELDS), set(SECTOR_TAGS))


if __name__ == "__main__":
    unittest.main()
