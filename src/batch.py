"""
batch.py: run a whole batch from a list of URLs with one command.

    python -m src.batch urls.txt --name antler_demo --cost-cap 2.50 [--discover-founders] [--fresh] [--retry-failed]
        [--rows rows.json] [--provider ...] [--model ...] [--cost-limit 0.50] [--label "demo version"]

What it does:
  1. Intake (src/intake.py): each URL becomes a row (company name, site, accelerator page, founder names if the page
     lists them). The rows are saved as runs/_batches/<name>/rows.json; `--rows` reuses an edited file instead.
  2. One company at a time through the workflow (src/agent.py), then its summary. Each company keeps the per-run cost
     guard; the batch has a total cost cap. Before each company the batch checks that spend so far plus a projected
     cost (the larger of $0.30 and 1.5 times the mean cost so far) stays under the cap; if not it stops cleanly and the
     remaining rows stay pending.
  3. State is saved after every row (runs/_batches/<name>/state.json). Running the same command again resumes: done rows
     are skipped, a row left "running" by a crash starts over, and pages and searches already paid for come from the
     cache. A failed row is retried once at once; `--retry-failed` retries failed rows on a later run.
  4. At the end (also after a stop): one page (exports/<name>.html), the three CSVs (exports/<name>/), the email and
     phone check, the search yield and batch_report.txt (exports/<name>/).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from schema import ClaimStatus, ClaimType, Run
from src.agent import COST_LIMIT_USD, PROJECT_ROOT, new_run_dir, run_company, without_founder_names
from src.assemble import IDENTITY_INDEPENDENT, IDENTITY_LABELS, IDENTITY_SITE_AND_SNIPPET, assemble, identity_confirmed, identity_level, load_run, run_flags
from src.export import export_runs
from src.intake import Entry, Resolved, load_rows, read_entries, resolve, save_rows
from src.pii_check import check_exports, check_run
from src.summary import summarise_dossier
from src.view import build_page, company_view
from src.yield_report import Yield
from src.cache import PageCache

MIN_ROW_COST = 0.30           # projected cost of a row before any row has finished
PROJECTION_FACTOR = 1.5       # times the mean cost of finished rows
MAX_ATTEMPTS = 2              # a failed row is tried once more at once


@dataclass
class Deps:
    """What the runner calls out to; tests replace these so nothing touches the network."""
    resolve: Callable[[Entry], Resolved]
    run_company: Callable[[Resolved, bool], Run]                   # (resolved row, discover founders) -> a finished Run
    summarise: Callable[[Run], dict] = field(default=lambda run: summarise_dossier(assemble(run).dossier))


def default_deps(provider: str, model: Optional[str], cost_limit: float, fresh: bool) -> Deps:
    cache = PageCache(read=not fresh)

    def run_one(resolved: Resolved, discover: bool) -> Run:
        row, raw = (without_founder_names(resolved.row), []) if discover else (resolved.row, resolved.raw_founders)
        return run_company(row, raw, provider, model, cost_limit, fresh, new_run_dir())

    return Deps(resolve=lambda entry: resolve(entry, cache=cache), run_company=run_one)


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

def batch_dir(name: str, root: Path = PROJECT_ROOT) -> Path:
    return root / "runs" / "_batches" / name


def load_state(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)                                           # the state file is never left half written


def new_state(name: str, cost_cap: float, discover: bool, sources: list[str]) -> dict:
    return {"name": name, "cost_cap": cost_cap, "discover_founders": discover, "spent": 0.0, "stopped": "",
            "rows": [{"index": i, "source_url": u, "resolved": None, "status": "pending", "run_id": "", "run_dir": "",
                      "cost": 0.0, "summary_cost": 0.0, "attempts": 0, "error": "", "stop_reason": ""}
                     for i, u in enumerate(sources)]}


def projected_cost(state: dict) -> float:
    done = [r["cost"] + r["summary_cost"] for r in state["rows"] if r["status"] == "done"]
    return max(MIN_ROW_COST, PROJECTION_FACTOR * sum(done) / len(done)) if done else MIN_ROW_COST


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

def process_row(row: dict, state: dict, deps: Deps, entries: dict[str, Entry], state_path: Path,
                log: Callable[[str], None], root: Path = PROJECT_ROOT) -> None:
    """One attempt at one row: resolve if needed, run the workflow, write the summary. Updates the row in place."""
    row["attempts"] += 1
    row["status"], row["error"] = "running", ""
    save_state(state_path, state)
    try:
        if row["resolved"] is None:
            resolved = deps.resolve(entries.get(row["source_url"]) or Entry(row["source_url"]))
            row["resolved"] = resolved.to_json()
        resolved = Resolved.from_json(row["resolved"])
        run = deps.run_company(resolved, state["discover_founders"])
        row["run_id"], row["run_dir"], row["stop_reason"] = run.run_id, f"runs/{run.run_id}", run.stop_reason or ""
        row["cost"] = round(row["cost"] + run.cost_usd, 5)         # all attempts count toward the cap
        state["spent"] = round(sum(r["cost"] + r["summary_cost"] for r in state["rows"]), 5)
        if run.stop_reason == "error":
            raise RuntimeError("the workflow stopped with an error: " + next(
                (e for e in run.events if e.startswith("stopped")), "no detail"))
        try:                                                    # a summary that cannot be made does not fail the row
            summary = deps.summarise(run)
            (root / row["run_dir"] / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            row["summary_cost"] = round(row["summary_cost"] + float(summary.get("cost_usd", 0.0)), 5)
        except Exception as e:                                  # noqa: BLE001 - any failure is recorded, the batch goes on
            log(f"  summary failed: {type(e).__name__}: {e}")
        row["status"] = "done"
    except Exception as e:                                      # noqa: BLE001 - a row's failure never stops the batch
        row["status"], row["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    state["spent"] = round(sum(r["cost"] + r["summary_cost"] for r in state["rows"]), 5)
    save_state(state_path, state)


def run_batch(sources: list[str], name: str, cost_cap: float, deps: Deps, discover: bool = False,
              retry_failed: bool = False, resolved_rows: Optional[list[Resolved]] = None,
              entries: Optional[list[Entry]] = None, root: Path = PROJECT_ROOT,
              log: Callable[[str], None] = print) -> dict:
    """Run (or resume) a batch and return its state. `sources` are the row keys (the URLs); `entries` carry optional
    founder names; `resolved_rows` (from --rows) skip intake."""
    directory = batch_dir(name, root)
    state_path = directory / "state.json"
    state = load_state(state_path)
    if state is None:
        state = new_state(name, cost_cap, discover, sources)
        for r, resolved in zip(state["rows"], resolved_rows or []):
            r["resolved"] = resolved.to_json()
    else:
        state["cost_cap"] = cost_cap                              # a resume may raise or lower the cap
        for r in state["rows"]:
            if r["status"] == "running":                          # left over from a crash
                r["status"] = "pending"
            elif r["status"] == "failed" and retry_failed:
                r["status"], r["attempts"] = "pending", 0
    state["stopped"] = ""
    by_url = {e.url: e for e in (entries or [])}
    save_state(state_path, state)

    for row in state["rows"]:
        if row["status"] != "pending":
            continue
        remaining = state["cost_cap"] - state["spent"]
        needed = projected_cost(state)
        if needed > remaining:
            state["stopped"] = (f"cost cap: ${state['spent']:.3f} spent of ${state['cost_cap']:.2f}, the next row is "
                                f"projected at ${needed:.2f}")
            log(state["stopped"])
            break
        log(f"[{row['index'] + 1}/{len(state['rows'])}] {row['source_url']}")
        for _ in range(MAX_ATTEMPTS):
            process_row(row, state, deps, by_url, state_path, log, root)
            if row["status"] == "done":
                break
            log(f"  failed (attempt {row['attempts']}): {row['error']}")
            if row["attempts"] >= MAX_ATTEMPTS:
                break
        if row["status"] == "done":
            log(f"  done: {row['run_id']} cost ${row['cost'] + row['summary_cost']:.3f}")
    save_state(state_path, state)
    save_rows([Resolved.from_json(r["resolved"]) for r in state["rows"] if r["resolved"]], directory / "rows.json")
    return state


# --------------------------------------------------------------------------
# The end products
# --------------------------------------------------------------------------

def build_outputs(state: dict, name: str, label: str = "", root: Path = PROJECT_ROOT) -> dict[str, Path]:
    """The page, the three CSVs, the PII check and batch_report.txt for the rows that finished."""
    done = [r for r in state["rows"] if r["status"] == "done"]
    runs = [(load_run(root / r["run_dir"]), root / r["run_dir"]) for r in done]
    out_dir = root / "exports" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    page = root / "exports" / f"{name}.html"
    companies = []
    for run, run_dir in runs:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) if (run_dir / "summary.json").exists() else None
        companies.append(company_view(run, summary))
    companies.sort(key=lambda c: c["name"].lower())
    page.write_text(build_page(companies, label or name), encoding="utf-8")
    counts = export_runs([r for r, _ in runs], out_dir) if runs else {}
    pii = {run.run_id: check_run(run_dir) for run, run_dir in runs}
    if counts:
        pii["exported CSVs"] = check_exports(out_dir)
    y = Yield()
    for run, _ in runs:
        y.add(run)
    report = out_dir / "batch_report.txt"
    report.write_text(batch_report(state, runs, pii, y, counts), encoding="utf-8")
    return {"page": page, "csv_dir": out_dir, "report": report}


def site_pages(run: Run) -> int:
    """Pages kept from the company's sitemap or common paths, from the run's events."""
    return sum(int(m.group(1)) for e in run.events if (m := re.search(r"^sitemap: .*pages kept: (\d+)$", e)))


def batch_report(state: dict, runs: list, pii: dict, y: Yield, counts: dict) -> str:
    lines = [f"Batch {state['name']}: {sum(r['status'] == 'done' for r in state['rows'])} of {len(state['rows'])} rows done, "
             f"{sum(r['status'] == 'failed' for r in state['rows'])} failed, {sum(r['status'] == 'pending' for r in state['rows'])} pending",
             f"Cost: ${state['spent']:.3f} of the ${state['cost_cap']:.2f} cap (model, billed searches and summaries)"
             + (f"; stopped: {state['stopped']}" if state["stopped"] else ""), ""]
    lines.append(f"{'row':>3} {'status':8s} {'founders':>8s} {'indep. id':>9s} {'sourced':>7s} {'snippet':>7s} {'rejected':>8s} "
                 f"{'searches':>8s} {'billed':>6s} {'sitemap':>7s} {'cost':>7s}  note")
    by_id = {run.run_id: run for run, _ in runs}
    for r in state["rows"]:
        run = by_id.get(r["run_id"])
        if run is None:
            lines.append(f"{r['index'] + 1:3d} {r['status']:8s} {'':>8s} {'':>9s} {'':>7s} {'':>7s} {'':>8s} {'':>8s} {'':>6s} {'':>7s} "
                         f"{r['cost'] + r['summary_cost']:7.3f}  {r['error']}")
            continue
        names = run.input.founder_names
        independent = sum(identity_confirmed(run, f) for f in names)
        tier = lambda status: sum(c.status == status for c in run.claims)
        billed = sum(e.startswith("search fee") for e in run.events)
        flags = run_flags(run)
        lines.append(f"{r['index'] + 1:3d} {r['status']:8s} {len(names):8d} {independent:9d} {tier(ClaimStatus.SOURCED):7d} "
                     f"{tier(ClaimStatus.SNIPPET):7d} {tier(ClaimStatus.REJECTED):8d} {run.budget.searches_used:8d} {billed:6d} "
                     f"{site_pages(run):7d} {r['cost'] + r['summary_cost']:7.3f}  {r['stop_reason']}"
                     + "".join(f"  FLAG: {f}" for f in flags))
    all_runs = [run for run, _ in runs]
    tiers = Counter(f.confirmation.value for run in all_runs for f in assemble(run).dossier.facts.get(ClaimType.FUNDING, []))
    n_founders = sum(len(run.input.founder_names) for run in all_runs)
    levels = Counter(identity_level(run, f) for run in all_runs for f in run.input.founder_names)
    n_independent = levels[IDENTITY_INDEPENDENT]
    flagged = [(run, f) for run in all_runs for f in run_flags(run)]
    lines += ["", f"Redirects flagged: {len(flagged)}" + "".join(f"\n  {run.input.company_name}: {f} (founders and facts from it need review)" for run, f in flagged),
              f"Pages kept from the sitemap or common paths: {sum(site_pages(run) for run in all_runs)}",
              f"Founders with an identity or role claim from an independent source: {n_independent} of {n_founders}",
              "Identity badge: " + ", ".join(f"{IDENTITY_LABELS[k]} {levels[k]}" for k in (IDENTITY_INDEPENDENT, IDENTITY_SITE_AND_SNIPPET, "unconfirmed")),
              "Funding facts by tier: " + (", ".join(f"{t} {n}" for t, n in sorted(tiers.items())) or "none")]
    failures = [r for r in state["rows"] if r["status"] == "failed"]
    lines += ["", f"Failures ({len(failures)}):"] + [f"  row {r['index'] + 1}: {r['error']}" for r in failures]
    total_hits = sum(len(h) for h in pii.values())
    emails = sum(c.claim_type == ClaimType.EMAIL and c.status == ClaimStatus.SOURCED for run, _ in runs for c in run.claims)
    lines += ["", f"Founder work emails admitted (people.csv Email only): {emails}"]
    lines += ["", f"PII check (emails outside people.csv Email, phone numbers) over {sum(len(run.claims) for run, _ in runs)} claims, the summaries and the CSVs: "
                  f"{total_hits} hits" + "".join(f"\n  {rid}: {where} / {field_}: {kind}" for rid, hits in pii.items() for where, field_, kind in hits)]
    lines += ["", "Search yield (credits and kept facts per query type and per organisation-query position):", y.format()]
    if counts:
        lines += ["", f"Export: {counts.get('companies', 0)} companies, {counts.get('people', 0)} people, {counts.get('claims', 0)} claims"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "batch"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of companies from a list of URLs.")
    parser.add_argument("input", nargs="?", help="a text file with one URL per line, or a CSV with a url column")
    parser.add_argument("--name", required=True, help="the batch's name (its state lives in runs/_batches/<name>/)")
    parser.add_argument("--cost-cap", type=float, required=True, help="total cost cap for the batch in USD")
    parser.add_argument("--rows", help="rows.json (edited) to use instead of reading the input")
    parser.add_argument("--discover-founders", action="store_true", help="ignore founder names; the model finds the founders")
    parser.add_argument("--retry-failed", action="store_true", help="retry rows that failed in an earlier run")
    parser.add_argument("--fresh", action="store_true", help="bypass the page and search cache")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openrouter"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--cost-limit", type=float, default=COST_LIMIT_USD, help="per-company cost guard in USD")
    parser.add_argument("--label", default="", help="a label for the page's landing view")
    args = parser.parse_args()
    name = safe_name(args.name)

    state = load_state(batch_dir(name) / "state.json")
    deps = default_deps(args.provider, args.model, args.cost_limit, args.fresh)
    resolved_rows = entries = None
    if state is None:
        if args.rows:
            resolved_rows = load_rows(Path(args.rows))
            sources = [r.source_url for r in resolved_rows]
        elif args.input:
            entries = read_entries(Path(args.input))
            sources = [e.url for e in entries]
        else:
            raise SystemExit("give an input file (or --rows) for a new batch")
        print(f"New batch {name}: {len(sources)} rows, cost cap ${args.cost_cap:.2f}")
    else:
        sources = [r["source_url"] for r in state["rows"]]
        entries = read_entries(Path(args.input)) if args.input else None
        print(f"Resuming batch {name}: {sum(r['status'] == 'done' for r in state['rows'])} of {len(sources)} rows done")
    state = run_batch(sources, name, args.cost_cap, deps, args.discover_founders, args.retry_failed, resolved_rows, entries)
    outputs = build_outputs(state, name, args.label)
    print(f"Page: {outputs['page']}\nCSVs and report: {outputs['csv_dir']}/")


if __name__ == "__main__":
    main()
