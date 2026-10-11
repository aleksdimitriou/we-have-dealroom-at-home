"""
yield_report.py: what each kind of search bought, from stored runs.

For every search a run made, the query type (name+company, name+organisation, company funding/investors, company news,
database funding, company founders) and, for organisation searches, its position (1st, 2nd, 3rd for that founder) are
logged in the run's events as "search aNNN type=... [position=n]". Runs made before that logging are classified from the
query text. Per type and per position this reports:
    credits       searches run (one credit each; a search reused from the cache still counts here)
    pages         pages fetched that came from a result of that search
    results       results the searches returned (counted for every search, not only the first to return a URL)
    claims        sourced or snippet claims whose page or snippet came from a result of that search
    snippet       of those, the snippet-tier ones
    new           of those, claims whose value no claim from an anchor, a chosen link or OpenAlex already states
Bias to know: a URL returned by several searches is credited to the FIRST one (earlier queries look better), so the
numbers say what each type delivered first, not what it would deliver alone.

Usage:  python -m src.yield_report runs/<id> [runs/<id> ...]
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from schema import ActionKind, ClaimStatus, FoundVia, Run
from src.assemble import load_run, normalise_value, same_value
from src.verify import normalise_url

# "company news" is no longer searched; the type stays so stored runs that made the search still report it.
TYPES = ["name+company", "name+organisation", "company funding/investors", "company news", "database funding",
         "database funding by domain", "company founders"]
_EVENT = re.compile(r"^search (a\d+) type=(.+?)(?: position=(\d+))?$")


def classify(query: str, subject_is_founder: bool, company: str) -> str:
    """The query type from the query text (for runs without the logged events)."""
    if "[sites:" in query:
        first = (re.findall(r'"([^"]+)"', query) or [""])[0]
        site = query.split("[sites:", 1)[1].rstrip("] ").strip()
        by_domain = "." in first and " " not in first
        if not query.split("[sites:", 1)[0].rstrip().endswith(" funding"):      # the snippet budget: one site, no extra word
            if site == "linkedin.com":
                return "linkedin founder" if subject_is_founder else "linkedin company"
            return f"database {site} ({'domain' if by_domain else 'name'})"
        return "database funding by domain" if by_domain else "database funding"
    if query.endswith(" funding investors"):
        return "company funding/investors"
    if query.endswith(" news"):
        return "company news"
    if query.endswith(" founders") and not subject_is_founder:
        return "company founders"
    quoted = re.findall(r'"([^"]+)"', query)
    if len(quoted) >= 2:
        return "name+company" if quoted[1].lower() == company.lower() else "name+organisation"
    return "other"


def search_types(run: Run) -> dict[str, tuple[str, int]]:
    """action id -> (query type, organisation position or 0)."""
    logged = {}
    for e in run.events:
        m = _EVENT.match(e)
        if m:
            logged[m.group(1)] = (m.group(2), int(m.group(3) or 0))
    out, seen = {}, Counter()
    for a in run.actions:
        if a.kind != ActionKind.SEARCH:
            continue
        if a.action_id in logged:
            out[a.action_id] = logged[a.action_id]
            continue
        t = classify(a.query_or_url, a.for_subject is not None, run.input.company_name)
        position = 0
        if t == "name+organisation":
            seen[a.for_subject] += 1
            position = seen[a.for_subject]
        out[a.action_id] = (t, position)
    return out


class Yield:
    """Credits and kept facts per query type and per organisation-query position, over any number of runs."""

    def __init__(self):
        self.by_type: dict[str, Counter] = defaultdict(Counter)
        self.by_position: dict[int, Counter] = defaultdict(Counter)

    def _count(self, t: str, position: int, what: str, n: int = 1) -> None:
        self.by_type[t][what] += n
        if t == "name+organisation" and position:
            self.by_position[position][what] += n

    def add(self, run: Run) -> None:
        types = search_types(run)
        first: dict[str, tuple[str, int]] = {}
        for r in run.search_results:
            first.setdefault(normalise_url(r.url), types.get(r.action_id, ("other", 0)))
        for t, position in types.values():
            self._count(t, position, "credits")
        for r in run.search_results:                                  # every result a search returned, not only the first credit
            self._count(*types.get(r.action_id, ("other", 0)), "results")
        sources = {normalise_url(s.url): s for s in run.sources}
        for s in run.sources:
            if s.found_via == FoundVia.SEARCH and normalise_url(s.url) in first:
                self._count(*first[normalise_url(s.url)], "pages")

        def origin(c):
            s = sources.get(normalise_url(c.source_url or ""))
            if s is not None:
                return "search" if s.found_via == FoundVia.SEARCH else "other"
            return "snippet" if c.status == ClaimStatus.SNIPPET else "other"

        elsewhere: dict[tuple, list[str]] = defaultdict(list)
        for c in run.claims:
            if c.status == ClaimStatus.SOURCED and origin(c) == "other":
                elsewhere[(c.subject.lower(), c.claim_type)].append(normalise_value(c.value))
        for c in run.claims:
            if c.status not in (ClaimStatus.SOURCED, ClaimStatus.SNIPPET) or origin(c) == "other":
                continue
            key = first.get(normalise_url(c.source_url or ""))
            if key is None:
                continue
            self._count(*key, "claims")
            if c.status == ClaimStatus.SNIPPET:
                self._count(*key, "snippet")
            value = normalise_value(c.value)
            if not any(same_value(value, x) for x in elsewhere[(c.subject.lower(), c.claim_type)]):
                self._count(*key, "new")

    def format(self) -> str:
        lines = [f"{'query type':36s} {'credits':>7s} {'results':>7s} {'pages':>6s} {'claims':>7s} {'snippet':>7s} {'new':>5s} {'new/credit':>10s}"]
        known = [t for t in TYPES if t in self.by_type]
        later = sorted(t for t in self.by_type if t not in TYPES and t != "other")
        for t in known + later + (["other"] if "other" in self.by_type else []):
            c = self.by_type[t]
            lines.append(f"{t:36s} {c['credits']:7d} {c['results']:7d} {c['pages']:6d} {c['claims']:7d} {c['snippet']:7d} {c['new']:5d} "
                         f"{c['new'] / c['credits'] if c['credits'] else 0:10.2f}")
        lines.append("name+organisation by position")
        for p in sorted(self.by_position):
            c = self.by_position[p]
            lines.append(f"  {p}{'st' if p == 1 else 'nd' if p == 2 else 'rd' if p == 3 else 'th'} organisation {c['credits']:>21d} "
                         f"{c['results']:7d} {c['pages']:6d} {c['claims']:7d} {c['snippet']:7d} {c['new']:5d} {c['new'] / c['credits'] if c['credits'] else 0:10.2f}")
        lines.append("(a URL returned by several searches is credited to the first one)")
        return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m src.yield_report runs/<id> [runs/<id> ...]")
    y = Yield()
    for arg in sys.argv[1:]:
        y.add(load_run(Path(arg)))
    print(y.format())


if __name__ == "__main__":
    main()
