"""
Tests for src/intake.py and src/batch.py with fake pages and a fake workflow. No network, no API calls.
Run from the project root:  python -m unittest discover tests
"""

import json
import tempfile
import unittest
from pathlib import Path

from schema import ActionOutcome, BatchRow, Claim, ClaimStatus, ClaimType, Run, SourceOrigin
from src.batch import Deps, MIN_ROW_COST, batch_dir, build_outputs, load_state, projected_cost, run_batch
from src.fetch import FetchResult
from src.intake import (
    Entry, IntakeError, Resolved, is_yc_company_page, load_rows, name_from_domain, name_from_slug, parse_yc_page,
    read_entries, resolve, save_rows,
)

YC_TEXT = """Open menu
Home›Companies›Torus Labs
Torus Labs
AI agents for engineering firms
Summer 2026
Active
Enterprise Software
San Francisco
Company
Jobs4
https://www.torus-labs.example
AI agents for engineering firms
We help engineering firms work faster.
Active Founders
Jane Doe
Founder/CEO
Co-founder and CEO. Previously at Initech.
John (Johnny) Roe
Co-Founder
Builds the platform.
Company Launches
Latest News
Sam Smith
Investor
"""


def page(url, text="", title="", links=()):
    return FetchResult(requested_url=url, outcome=ActionOutcome.OK, final_url=url, status=200, title=title, text=text, links=list(links))


YC_URL = "https://www.ycombinator.com/companies/torus-labs"
SITE_URL = "https://www.acme-robots.example/"


class IntakeTests(unittest.TestCase):
    def test_a_plain_list_and_a_csv_are_both_read(self):
        with tempfile.TemporaryDirectory() as d:
            txt = Path(d) / "urls.txt"
            txt.write_text("# the first batch\n" + YC_URL + "\n\n" + SITE_URL + " # a site\n", encoding="utf-8")
            self.assertEqual([e.url for e in read_entries(txt)], [YC_URL, SITE_URL])
            csv = Path(d) / "urls.csv"
            csv.write_text(f"url,name,founders,accelerator_url\n{SITE_URL},Acme Robots,Jane Doe; John Roe,https://www.antler.co/portfolio\n", encoding="utf-8")
            entry = read_entries(csv)[0]
            self.assertEqual((entry.url, entry.name, entry.founders, entry.accelerator_url),
                             (SITE_URL, "Acme Robots", ["Jane Doe", "John Roe"], "https://www.antler.co/portfolio"))

    def test_a_yc_page_gives_name_site_batch_and_founders(self):
        info = parse_yc_page(YC_TEXT, [])
        self.assertEqual(info["name"], "Torus Labs")
        self.assertEqual(info["website"], "https://www.torus-labs.example")
        self.assertEqual(info["batch"], "Summer 2026")
        self.assertEqual(info["founders"], ["Jane Doe", "John (Johnny) Roe"])            # not "Sam Smith": his role is not a founder's

    def test_resolve_a_yc_page(self):
        r = resolve(Entry(YC_URL), fetch=lambda url: page(url, YC_TEXT, "Torus Labs: AI agents | Y Combinator"))
        self.assertEqual((r.row.company_name, r.row.company_url, r.row.accelerator_profile_url), ("Torus Labs", "https://www.torus-labs.example", YC_URL))
        self.assertEqual((r.row.founder_names, r.raw_founders), (["Jane Doe", "John Roe"], ["Jane Doe", "John (Johnny) Roe"]))
        self.assertTrue(any("2 active founder(s)" in n for n in r.notes))

    def test_founders_in_the_input_file_win_over_the_page(self):
        r = resolve(Entry(YC_URL, founders=["Pat Quinn"]), fetch=lambda url: page(url, YC_TEXT))
        self.assertEqual(r.row.founder_names, ["Pat Quinn"])
        self.assertTrue(any("taken from the input file" in n for n in r.notes))

    def test_resolve_a_company_site_takes_the_name_from_the_list_never_from_the_page_title(self):
        title = "Argus: security for developers | Now with AI"
        r = resolve(Entry(SITE_URL, name="Hydra"), fetch=lambda url: page(url, "Acme builds robots.", title))
        self.assertEqual((r.row.company_name, r.row.company_url, r.row.accelerator_profile_url, r.row.founder_names), ("Hydra", SITE_URL, None, []))
        self.assertTrue(any("from the input list" in n and "discovery will find them" in n for n in r.notes))

    def test_without_a_name_in_the_list_the_domain_names_the_company_and_the_row_says_so(self):
        r = resolve(Entry(SITE_URL), fetch=lambda url: page(url, "x", "Garbled &amp; Title | Tagline"))
        self.assertEqual(r.row.company_name, "Acme-robots")
        self.assertTrue(any("from the domain (no name given)" in n for n in r.notes))
        self.assertEqual(name_from_domain("https://www.flizpay.de/"), "Flizpay")
        self.assertEqual(name_from_domain("https://x.co.uk/"), "X")
        self.assertEqual(name_from_slug("https://www.ycombinator.com/companies/torus-labs/"), "Torus Labs")

    def test_a_yc_pages_name_is_the_pages_own_not_its_title_and_the_list_name_wins(self):
        r = resolve(Entry(YC_URL), fetch=lambda url: page(url, YC_TEXT, "Something else entirely | Y Combinator"))
        self.assertEqual(r.row.company_name, "Torus Labs")
        r = resolve(Entry(YC_URL, name="Torus"), fetch=lambda url: page(url, YC_TEXT))
        self.assertEqual(r.row.company_name, "Torus")

    def test_an_antler_card_page_is_kept_as_the_accelerator_page(self):
        card = "https://www.antler.co/portfolio?0b933bfd_page=4"
        r = resolve(Entry(SITE_URL, name="Claro", accelerator_url=card), fetch=lambda url: page(url, "x"))
        self.assertEqual(r.row.accelerator_profile_url, card)

    def test_rows_round_trip_through_rows_json(self):
        r = resolve(Entry(YC_URL), fetch=lambda url: page(url, YC_TEXT))
        with tempfile.TemporaryDirectory() as d:
            save_rows([r], Path(d) / "rows.json")
            back = load_rows(Path(d) / "rows.json")[0]
        self.assertEqual((back.row, back.raw_founders, back.source_url), (r.row, r.raw_founders, r.source_url))


class FakeWorkflow:
    """Stands in for the real workflow: writes a run folder under the temp root and returns the Run."""

    def __init__(self, root: Path, cost=0.20, fail_on=(), fail_times=1, error_on=()):
        self.root, self.cost, self.fail_on, self.fail_times, self.error_on = root, cost, set(fail_on), fail_times, set(error_on)
        self.calls: list[str] = []
        self.failed: dict[str, int] = {}
        self.summaries = 0
        self.redirect_on: dict[str, str] = {}

    def resolve(self, entry):
        return Resolved(BatchRow(company_name=entry.url.split("//")[1].split(".")[0].title(), company_url=entry.url, founder_names=["Jane Doe"]),
                        ["Jane Doe"], entry.url)

    def run_company(self, resolved: Resolved, discover: bool) -> Run:
        key = resolved.row.company_url
        self.calls.append(key)
        if key in self.fail_on and self.failed.get(key, 0) < self.fail_times:
            self.failed[key] = self.failed.get(key, 0) + 1
            raise RuntimeError("network down")
        rid = f"20261009_1200{len(self.calls):02d}"
        from datetime import datetime
        claim = Claim(claim_id="c001", subject="Jane Doe", claim_type=ClaimType.IDENTITY, value="Founder", source_origin=SourceOrigin.COUNTERPARTY,
                      status=ClaimStatus.SOURCED, source_url="https://yc.example/x", quote="Jane Doe Founder")
        run = Run(run_id=rid, model="m", started_at=datetime(2026, 10, 9, 12, 0), finished_at=datetime(2026, 10, 9, 12, 1),
                  stop_reason="error" if key in self.error_on else "submitted", cost_usd=self.cost,
                  input=BatchRow(company_name=resolved.row.company_name, company_url=key, founder_names=[] if discover else ["Jane Doe"]),
                  claims=[claim])
        if key in self.redirect_on:
            run.events.append(f"redirect: {key.split('//')[1]} now redirects to {self.redirect_on[key]}")
            run.events.append("sitemap: 12 urls, 3 pages picked; pages kept: 3")
        (self.root / "runs" / rid).mkdir(parents=True)
        (self.root / "runs" / rid / "run.json").write_text(run.model_dump_json())
        return run

    def summarise(self, run: Run) -> dict:
        self.summaries += 1
        return {"label": "Summary generated from sourced claims", "sentences": [], "founder_headlines": [], "problems": [],
                "dropped": [], "cost_usd": 0.01, "calls": []}

    def deps(self) -> Deps:
        return Deps(resolve=self.resolve, run_company=self.run_company, summarise=self.summarise)


URLS = [f"https://co{i}.example" for i in range(5)]


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.log: list[str] = []

    def tearDown(self):
        self.tmp.cleanup()

    def batch(self, wf, cap=10.0, urls=URLS, **kw):
        return run_batch(urls, "demo", cap, wf.deps(), root=self.root, log=self.log.append, **kw)

    def test_every_row_runs_and_the_state_records_cost_and_run_ids(self):
        wf = FakeWorkflow(self.root)
        state = self.batch(wf)
        self.assertEqual([r["status"] for r in state["rows"]], ["done"] * 5)
        self.assertAlmostEqual(state["spent"], 5 * 0.21)                                    # model plus summary
        self.assertTrue(all(r["run_id"] for r in state["rows"]))
        self.assertEqual(load_state(batch_dir("demo", self.root) / "state.json")["spent"], state["spent"])
        self.assertEqual(len(json.loads((batch_dir("demo", self.root) / "rows.json").read_text())["rows"]), 5)

    def test_the_cost_cap_stops_cleanly_and_a_resume_with_more_budget_finishes(self):
        wf = FakeWorkflow(self.root)
        state = self.batch(wf, cap=0.60)                                                   # two rows: 0.42, then 0.42 + 0.31 > 0.60
        self.assertEqual([r["status"] for r in state["rows"]], ["done", "done", "pending", "pending", "pending"])
        self.assertIn("cost cap", state["stopped"])
        self.assertEqual(len(wf.calls), 2)
        state = self.batch(wf, cap=10.0)
        self.assertEqual([r["status"] for r in state["rows"]], ["done"] * 5)
        self.assertEqual(len(wf.calls), 5)                                                 # the first two were not run again
        self.assertEqual(state["stopped"], "")

    def test_a_failed_row_is_retried_once_at_once_and_the_batch_goes_on(self):
        wf = FakeWorkflow(self.root, fail_on={URLS[1]}, fail_times=1)
        state = self.batch(wf)
        self.assertEqual([r["status"] for r in state["rows"]], ["done"] * 5)
        self.assertEqual(state["rows"][1]["attempts"], 2)

    def test_a_row_that_keeps_failing_is_marked_failed_and_retry_failed_tries_again(self):
        wf = FakeWorkflow(self.root, fail_on={URLS[2]}, fail_times=99)
        state = self.batch(wf)
        self.assertEqual([r["status"] for r in state["rows"]], ["done", "done", "failed", "done", "done"])
        self.assertIn("network down", state["rows"][2]["error"])
        wf.fail_times = 0                                                                  # the network is back
        state = self.batch(wf)                                                             # a plain resume leaves it failed
        self.assertEqual(state["rows"][2]["status"], "failed")
        state = self.batch(wf, retry_failed=True)
        self.assertEqual(state["rows"][2]["status"], "done")

    def test_a_workflow_that_stops_with_an_error_counts_as_failed_and_its_cost_still_counts(self):
        wf = FakeWorkflow(self.root, error_on={URLS[0]})
        state = self.batch(wf, urls=URLS[:2])
        self.assertEqual(state["rows"][0]["status"], "failed")
        self.assertAlmostEqual(state["rows"][0]["cost"], 0.40)                              # both attempts were paid for
        self.assertAlmostEqual(state["spent"], 0.40 + 0.21)

    def test_a_crash_leaves_a_running_row_that_starts_over_on_resume(self):
        wf = FakeWorkflow(self.root)
        state = self.batch(wf, urls=URLS[:2])
        path = batch_dir("demo", self.root) / "state.json"
        state["rows"][1]["status"] = "running"
        path.write_text(json.dumps(state))
        calls_before = len(wf.calls)
        state = self.batch(wf, urls=URLS[:2])
        self.assertEqual((state["rows"][1]["status"], len(wf.calls) - calls_before), ("done", 1))

    def test_the_projection_uses_the_mean_cost_so_far(self):
        self.assertEqual(projected_cost({"rows": []}), MIN_ROW_COST)
        rows = [{"status": "done", "cost": 0.40, "summary_cost": 0.0}, {"status": "done", "cost": 0.20, "summary_cost": 0.0}, {"status": "pending", "cost": 0, "summary_cost": 0}]
        self.assertAlmostEqual(projected_cost({"rows": rows}), 0.45)

    def test_the_outputs_are_one_page_three_csvs_and_a_report(self):
        wf = FakeWorkflow(self.root)
        state = self.batch(wf, urls=URLS[:3])
        out = build_outputs(state, "demo", "demo version", root=self.root)
        self.assertTrue(out["page"].exists() and out["page"].name == "demo.html")
        for table in ("companies.csv", "people.csv", "claims.csv"):
            self.assertTrue((out["csv_dir"] / table).exists(), table)
        report = out["report"].read_text(encoding="utf-8")
        self.assertIn("3 of 3 rows done", report)
        self.assertIn("PII check", report)
        self.assertIn("0 hits", report)
        self.assertIn("name+organisation by position", report)
        self.assertIn("demo version", out["page"].read_text(encoding="utf-8"))

    def test_the_report_lists_redirects_sitemap_pages_independent_identity_and_funding_tiers(self):
        wf = FakeWorkflow(self.root)
        wf.redirect_on = {URLS[1]: "newname.dev"}
        state = self.batch(wf, urls=URLS[:3])
        report = build_outputs(state, "demo", "", root=self.root)["report"].read_text(encoding="utf-8")
        self.assertIn("Redirects flagged: 1", report)
        self.assertIn("co1.example now redirects to newname.dev", report)
        self.assertIn("(founders and facts from it need review)", report)
        self.assertIn("FLAG: co1.example now redirects to newname.dev", report)
        self.assertIn("Pages kept from the sitemap or common paths: 3", report)
        self.assertIn("independent source:", report)
        self.assertIn("Identity badge: confirmed by an independent source", report)
        self.assertIn("named as founder by the company site and a snippet", report)
        self.assertIn("Funding facts by tier: none", report)

    def test_discover_founders_is_passed_to_the_workflow(self):
        wf = FakeWorkflow(self.root)
        state = self.batch(wf, urls=URLS[:1], discover=True)
        run = json.loads((self.root / state["rows"][0]["run_dir"] / "run.json").read_text())
        self.assertEqual(run["input"]["founder_names"], [])


if __name__ == "__main__":
    unittest.main()
