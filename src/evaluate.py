"""
evaluate.py: compare agent runs with the golden set. Measures correctness, not admissibility.

Method
------
1. Golden facts are the answer-key rows with found = Yes. Agent claims are the run's SOURCED claims.
2. Candidates are grouped by (company, subject) and limited to claim types present on both sides, in code.
   The golden key has no attribute column, so attribute is passed to the judge as context only.
3. One LLM judge call per (company, subject) group decides, for each agent claim, which golden fact of the
   same claim_type it states and whether the values mean the same thing:
       same | partial | contradicts | not_in_key.
   Claims with no golden fact of their type are not_in_key without a call. Golden facts with no agent claim of
   their type are misses without a call.
4. Judge spend is capped per evaluation (default $0.20) and logged; groups left over are reported as unjudged.
5. Each miss is classed reachable (the golden source URL was among the pages the run fetched: an extraction miss), surfaced by
   search (it appeared in search results but was not fetched) or unreachable (a navigation miss, e.g. LinkedIn).

Recall counts golden facts covered by a sourced claim: strict = same, lenient = same or partial.
Precision is the share of sourced claims that match a golden fact. The key holds only about 10 facts per
subject, so precision is a lower bound: a correct claim outside the key is "not_in_key", not wrong.
`contradicts` counts claims that disagree with a golden fact (namesakes, wrong values).

Usage (from the project root, venv active):
    python -m src.evaluate [--runs runs/<id> ...] [--cost-cap 0.20]
With no --runs, the latest finished run per golden company is used.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from schema import ClaimStatus, ClaimType, RejectionReason, Run
from src.assemble import latest_runs
from src.golden import GOLDEN_ANSWER_KEY, read_table
from src.llm import Usage, cost_usd, read_env_key
from src.verify import normalise_url

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JUDGE_MODEL = "claude-sonnet-5-5"
COST_CAP_USD = 0.20
VERDICTS = ("same", "partial", "contradicts", "not_in_key")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@dataclass
class GoldenFact:
    company: str
    subject: str                  # "company" or the founder's name
    claim_type: str
    value: str
    source_url: str
    origin: str
    linkedin_only: bool


@dataclass
class AgentClaim:
    subject: str
    claim_type: str
    attribute: Optional[str]
    value: str
    origin: str
    snippet: bool = False         # True for snippet-status claims (page not fetched); False for page-sourced


@dataclass
class Judgment:
    verdict: str                  # one of VERDICTS
    golden_index: Optional[int]   # index into the group's golden list
    reason: str = ""              # the judge's one-line reason


@dataclass
class GroupResult:
    company: str
    subject: str
    golden: list[GoldenFact]
    claims: list[AgentClaim]
    judgments: list[Optional[Judgment]]     # one per claim; None = unjudged (cost cap)


# A judge takes (golden facts, agent claims) of one group and returns one Judgment per claim, or None to skip.
Judge = Callable[[list[GoldenFact], list[AgentClaim]], Optional[list[Judgment]]]


def load_golden(path: Path = GOLDEN_ANSWER_KEY) -> list[GoldenFact]:
    """The answer key's found rows. Reads golden/golden_answer_key.csv by default, or the workbook (or the example
    file) if given its path. The key is private: it holds facts about real founders."""
    if not Path(path).exists():
        raise SystemExit(f"{path} not found. The answer key is private (golden/ is not in the repository); "
                         "data/golden_answer_key.example.csv shows the format.")
    rows = read_table(path, "Answer key")
    facts = []
    for r in rows[1:]:
        if r.get("A") and r.get("D") == "Yes":
            facts.append(GoldenFact(
                company=r["A"].strip(), subject=r.get("B", "").strip(), claim_type=r.get("C", "").strip(),
                value=r.get("E", "").strip(), source_url=r.get("F", "").strip(), origin=r.get("G", "").strip(),
                linkedin_only=r.get("H") == "Yes"))
    return facts


def same_subject(a: str, b: str) -> bool:
    return " ".join(a.lower().split()) == " ".join(b.lower().split())


# --------------------------------------------------------------------------
# Judge (LLM)
# --------------------------------------------------------------------------

JUDGE_SYSTEM = """You compare facts an agent found about a startup or its founder with an answer key.
For each agent claim, decide whether it states the same fact as one answer-key fact of the SAME claim type.
Verdicts:
- same: the claim and the key fact mean the same thing (wording, format, or detail level may differ).
- partial: the claim covers part of the key fact, or is vaguer, or covers one of several things the key fact lists.
- contradicts: the claim is about the same thing as a key fact but states an incompatible value.
- not_in_key: the claim is not about any key fact (it may be true; the key is incomplete).
Rules: judge only meaning, never source or origin. Match a claim to at most one key fact. A key fact may be
matched by several claims (one claim per investor, for example). Use golden_id null only with not_in_key.
Give a one-line reason for each verdict. Return one entry per claim, in order."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"judgments": {"type": "array", "items": {
        "type": "object",
        "properties": {"claim_id": {"type": "string"},
                       "verdict": {"type": "string", "enum": list(VERDICTS)},
                       "golden_id": {"type": ["string", "null"]},
                       "reason": {"type": "string"}},
        "required": ["claim_id", "verdict", "golden_id", "reason"], "additionalProperties": False}}},
    "required": ["judgments"], "additionalProperties": False,
}


class AnthropicJudge:
    """One call per group. Tracks spend and refuses (returns None) once the cap would be passed."""

    def __init__(self, cost_cap: float = COST_CAP_USD, model: str = JUDGE_MODEL):
        import anthropic
        self.client = anthropic.Anthropic(api_key=read_env_key("ANTHROPIC_API_KEY"))
        self.model, self.cost_cap = model, cost_cap
        self.spent = 0.0
        self.calls: list[dict] = []
        self._max_call_cost = 0.0

    def __call__(self, golden: list[GoldenFact], claims: list[AgentClaim]) -> Optional[list[Judgment]]:
        if self.spent + self._max_call_cost * 1.5 > self.cost_cap:    # next call would likely pass the cap
            return None
        g_lines = "\n".join(f"G{i}. [{g.claim_type}] {g.value}" for i, g in enumerate(golden))
        c_lines = "\n".join(
            f"C{i}. [{c.claim_type}{'/' + c.attribute if c.attribute else ''}] {c.value}" for i, c in enumerate(claims))
        prompt = f"ANSWER-KEY FACTS:\n{g_lines}\n\nAGENT CLAIMS:\n{c_lines}"
        response = self.client.messages.create(
            model=self.model, max_tokens=6000, system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}})
        u = response.usage
        usage = Usage(u.input_tokens or 0, u.cache_creation_input_tokens or 0, u.cache_read_input_tokens or 0,
                      u.output_tokens or 0)
        cost = cost_usd(response.model, usage)
        self.spent += cost
        self._max_call_cost = max(self._max_call_cost, cost)
        self.calls.append({"model": response.model, "input_tokens": usage.total_input,
                           "output_tokens": usage.output_tokens, "cost_usd": round(cost, 5)})
        text = next(b.text for b in response.content if b.type == "text")
        return parse_judgments(json.loads(text), len(golden), len(claims))


def parse_judgments(data: dict, n_golden: int, n_claims: int) -> list[Judgment]:
    """Turn the judge's JSON into one Judgment per claim; anything malformed becomes not_in_key."""
    by_claim = {j.get("claim_id"): j for j in data.get("judgments", [])}
    out = []
    for i in range(n_claims):
        j = by_claim.get(f"C{i}") or {}
        verdict = j.get("verdict") if j.get("verdict") in VERDICTS else "not_in_key"
        gid = j.get("golden_id")
        index = int(gid[1:]) if isinstance(gid, str) and gid[1:].isdigit() and int(gid[1:]) < n_golden else None
        out.append(Judgment(verdict if index is not None else "not_in_key", index, str(j.get("reason") or "")))
    return out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@dataclass
class Evaluation:
    groups: list[GroupResult] = field(default_factory=list)
    misses: list[dict] = field(default_factory=list)
    runs: dict[str, Run] = field(default_factory=dict)
    judge_cost: float = 0.0
    judge_calls: list[dict] = field(default_factory=list)


def evaluate(golden: list[GoldenFact], runs: dict[str, Run], judge: Judge) -> Evaluation:
    result = Evaluation(runs=runs)
    for company, run in runs.items():
        # Page-sourced claims, plus snippet-checked ones. Runs made before the snippet tier stored them as rejected
        # snippet_only leads; those count as snippet claims too, so old and new runs compare.
        sourced = [c for c in run.claims if c.claim_type != ClaimType.EMAIL       # an email never goes to the judge
                   and (c.status in (ClaimStatus.SOURCED, ClaimStatus.SNIPPET)
                        or (c.status == ClaimStatus.REJECTED and c.rejection_reason == RejectionReason.SNIPPET_ONLY))]
        subjects = {g.subject for g in golden if g.company == company} | {c.subject for c in sourced}
        for subject in sorted(subjects, key=str.lower):
            golden_s = [g for g in golden if g.company == company and same_subject(g.subject, subject)]
            claims_s = [AgentClaim(c.subject, c.claim_type.value, c.attribute.value if c.attribute else None,
                                   c.value, c.source_origin.value, snippet=c.status != ClaimStatus.SOURCED)
                        for c in sourced if same_subject(c.subject, subject)]
            if not golden_s and not claims_s:
                continue
            shared = {g.claim_type for g in golden_s} & {c.claim_type for c in claims_s}
            g_in = [g for g in golden_s if g.claim_type in shared]
            c_in = [c for c in claims_s if c.claim_type in shared]
            judged = judge(g_in, c_in) if c_in else []
            by_claim = {id(c): j for c, j in zip(c_in, judged)} if judged is not None else {}
            judgments = []
            for c in claims_s:
                if c.claim_type not in shared:
                    judgments.append(Judgment("not_in_key", None))
                elif judged is None:
                    judgments.append(None)                       # unjudged: cost cap reached
                else:
                    j = by_claim[id(c)]
                    if j.golden_index is not None and g_in[j.golden_index].claim_type != c.claim_type:
                        # A claim only matches golden facts of its own claim type; a cross-type match is ignored.
                        judgments.append(Judgment("not_in_key", None, "cross-type match ignored: " + j.reason))
                        continue
                    index = next(i for i, x in enumerate(golden_s) if x is g_in[j.golden_index]) if j.golden_index is not None else None
                    judgments.append(Judgment(j.verdict, index, j.reason))
            result.groups.append(GroupResult(company, subject, golden_s, claims_s, judgments))

    # misses and their reachability
    for group in result.groups:
        run = runs[group.company]
        fetched = {normalise_url(s.url) for s in run.sources}
        surfaced = {normalise_url(u) for u in run.search_result_urls}       # seen in search results
        for gi, g in enumerate(group.golden):
            verdicts = [j.verdict for j in group.judgments if j and j.golden_index == gi]
            if any(v in ("same", "partial") for v in verdicts):
                continue
            unjudged = any(j is None and c.claim_type == g.claim_type for c, j in zip(group.claims, group.judgments))
            reachable = normalise_url(g.source_url) in fetched if g.source_url else False
            result.misses.append({
                "company": group.company, "subject": g.subject, "claim_type": g.claim_type, "value": g.value,
                "source_url": g.source_url, "linkedin_only": g.linkedin_only,
                "contradicted": "contradicts" in verdicts, "unjudged": unjudged,
                "reachable": reachable,
                "surfaced": not reachable and bool(g.source_url) and normalise_url(g.source_url) in surfaced})
    return result


# --------------------------------------------------------------------------
# Numbers and report
# --------------------------------------------------------------------------

def recall_without_linkedin(groups: list[GroupResult]) -> Counter:
    """Recall counted only on golden facts the key does not mark linkedin_only."""
    c: Counter = Counter()
    for g in groups:
        for gi, fact in enumerate(g.golden):
            if fact.linkedin_only:
                continue
            matched = [(cl, j) for cl, j in zip(g.claims, g.judgments) if j and j.golden_index == gi]
            verdicts = [j.verdict for _, j in matched]
            c["golden"] += 1
            c["strict"] += "same" in verdicts
            c["strict_page"] += any(j.verdict == "same" and not cl.snippet for cl, j in matched)
            c["lenient"] += any(v in ("same", "partial") for v in verdicts)
    return c


def tally(groups: list[GroupResult], key: Callable[[GroupResult, str], bool] = lambda g, t: True) -> dict:
    """Counts over groups, split by claim_type. `key` filters (group, claim_type) pairs."""
    t: dict[str, Counter] = defaultdict(Counter)
    for g in groups:
        for gi, fact in enumerate(g.golden):
            if not key(g, fact.claim_type):
                continue
            matched = [(c, j) for c, j in zip(g.claims, g.judgments) if j and j.golden_index == gi]
            verdicts = [j.verdict for _, j in matched]
            t[fact.claim_type]["golden"] += 1
            t[fact.claim_type]["strict"] += "same" in verdicts
            t[fact.claim_type]["strict_page"] += any(j.verdict == "same" and not c.snippet for c, j in matched)
            t[fact.claim_type]["lenient"] += any(v in ("same", "partial") for v in verdicts)
        for c, j in zip(g.claims, g.judgments):
            if not key(g, c.claim_type):
                continue
            t[c.claim_type]["claims"] += 1
            t[c.claim_type]["claims_snippet"] += c.snippet
            if j is None:
                t[c.claim_type]["unjudged"] += 1
                continue
            t[c.claim_type]["claim_" + j.verdict] += 1
            if c.snippet:
                t[c.claim_type]["snippet_" + j.verdict] += 1
            elif j.verdict == "same":
                t[c.claim_type]["claim_same_page"] += 1
            if j.verdict in ("same", "partial") and j.golden_index is not None:
                t[c.claim_type]["origin_checked"] += 1
                t[c.claim_type]["origin_ok"] += c.origin == g.golden[j.golden_index].origin
    return t


def _pct(n: float, d: float) -> str:
    return f"{100 * n / d:.0f}%" if d else "-"


def _summary_row(label: str, c: Counter) -> str:
    claims = c["claims"] - c["unjudged"]
    return (f"{label:22} {c['golden']:>6} {c['strict']:>6} {_pct(c['strict'], c['golden']):>6} "
            f"{c['lenient']:>6} {_pct(c['lenient'], c['golden']):>6} | {c['claims']:>6} "
            f"{c['claim_same']:>5} {c['claim_partial']:>5} {c['claim_contradicts']:>6} {c['claim_not_in_key']:>6} "
            f"{_pct(c['claim_same'], claims):>6} {_pct(c['claim_same'] + c['claim_partial'], claims):>6}")


HEADER = (f"{'':22} {'golden':>6} {'same':>6} {'recall':>6} {'+part':>6} {'len.':>6} | {'claims':>6} "
          f"{'same':>5} {'part':>5} {'contra':>6} {'no-key':>6} {'prec.':>6} {'len.':>6}")


def total(t: dict) -> Counter:
    out: Counter = Counter()
    for c in t.values():
        out.update(c)
    return out


def format_report(ev: Evaluation) -> str:
    lines = ["EVALUATION: agent runs vs golden set (found facts only)",
             "recall = golden facts covered by a sourced claim (same | same+partial). precision = sourced claims matching a",
             "golden fact. The key holds ~10 facts per subject, so precision is a lower bound; 'no-key' claims may be true.",
             ""]
    companies = list(ev.runs)

    lines += ["OVERALL", HEADER, _summary_row("all", total(tally(ev.groups)))]
    for company in companies:
        lines.append(_summary_row(company, total(tally(ev.groups, lambda g, t, c=company: g.company == c))))
    lines += ["", "BY CLAIM TYPE (overall)", HEADER]
    overall = tally(ev.groups)
    for claim_type in sorted(overall):
        lines.append(_summary_row(claim_type, overall[claim_type]))

    for company in companies:
        lines += ["", f"--- {company} ---", HEADER]
        t = tally(ev.groups, lambda g, ct, c=company: g.company == c)
        for claim_type in sorted(t):
            lines.append(_summary_row(claim_type, t[claim_type]))

    o = total(overall)
    lines += ["", f"EXACT RECALL, ALL FACTS: page-sourced claims only {o['strict_page']} of {o['golden']} "
                  f"({_pct(o['strict_page'], o['golden'])}); page-sourced plus snippet-checked {o['strict']} "
                  f"({_pct(o['strict'], o['golden'])})",
              f"SNIPPET CLAIMS: {o['claims_snippet']} (same {o['snippet_same']}, partial {o['snippet_partial']}, "
              f"contradicts {o['snippet_contradicts']}, not in key {o['snippet_not_in_key']}); "
              f"page-sourced claims matching exactly: {o['claim_same_page']} of {o['claims'] - o['claims_snippet']}"]
    lines += ["", f"ORIGIN ACCURACY on matched claims: {o['origin_ok']} of {o['origin_checked']} "
                  f"({_pct(o['origin_ok'], o['origin_checked'])}) have the golden source_origin"]

    no_li = recall_without_linkedin(ev.groups)
    lines += [f"RECALL ON FACTS REACHABLE WITHOUT LINKEDIN (linkedin_only = No): exact, page-sourced only "
              f"{no_li['strict_page']} of {no_li['golden']} ({_pct(no_li['strict_page'], no_li['golden'])}); "
              f"page-sourced plus snippet-checked {no_li['strict']} ({_pct(no_li['strict'], no_li['golden'])}); "
              f"{no_li['lenient']} with partial ({_pct(no_li['lenient'], no_li['golden'])})"]

    misses = ev.misses
    reach = [m for m in misses if m["reachable"]]
    surfaced = [m for m in misses if m.get("surfaced")]
    unreach = [m for m in misses if not m["reachable"] and not m.get("surfaced")]
    lines += ["", f"MISSES: {len(misses)} golden facts not covered "
                  f"({len(reach)} reachable: golden source URL was fetched; {len(surfaced)} surfaced by search: "
                  f"in search results but not fetched; {len(unreach)} unreachable)"]
    for label, group in (("REACHABLE (extraction misses)", reach), ("SURFACED BY SEARCH (not fetched)", surfaced),
                         ("UNREACHABLE (navigation misses)", unreach)):
        lines.append(label)
        for m in group:
            tag = []
            if m["linkedin_only"]:
                tag.append("linkedin-only")
            if m["contradicted"]:
                tag.append("contradicted by a claim")
            if m["unjudged"]:
                tag.append("unjudged (cost cap)")
            lines.append(f"  [{m['company']} / {m['subject']} / {m['claim_type']}] {m['value'][:80]}"
                         f"  <{m['source_url'][:70]}>" + (f"  ({', '.join(tag)})" if tag else ""))

    contradictions = [(g, c, j) for g in ev.groups for c, j in zip(g.claims, g.judgments)
                      if j and j.verdict == "contradicts"]
    lines += ["", f"CONTRADICTIONS: {len(contradictions)} claims disagree with a golden fact"]
    for g, c, j in contradictions:
        lines.append(f"  [{g.company} / {c.subject} / {c.claim_type}] agent: {c.value[:70]} | key: {g.golden[j.golden_index].value[:70]}")

    lines += ["", "ACTIONS PER SUBJECT"]
    for company, run in ev.runs.items():
        per = Counter(a.for_subject or "company" for a in run.actions)
        lines.append(f"  {company}: " + ", ".join(f"{s} {n}" for s, n in per.items())
                     + f"; stopped: {run.stop_reason}; agent cost ${run.cost_usd:.4f}")
    unjudged = sum(1 for g in ev.groups for j in g.judgments if j is None)
    lines += ["", f"JUDGE: {len(ev.judge_calls)} calls, ${ev.judge_cost:.4f} (cap ${COST_CAP_USD:.2f}); "
                  f"{unjudged} claims unjudged"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent runs against the golden set.")
    parser.add_argument("--runs", nargs="*", help="run directories; default: the latest finished run per company")
    parser.add_argument("--golden", default=None, help="answer key: a CSV or the private .xlsx workbook "
                                                         "(default: golden/golden_answer_key.csv)")
    parser.add_argument("--cost-cap", type=float, default=COST_CAP_USD, help="judge spend cap in USD")
    args = parser.parse_args()

    if args.runs:
        dirs = {}
        for d in args.runs:
            run = Run.model_validate_json((Path(d) / "run.json").read_text())
            dirs[run.input.company_name] = Path(d)
    else:
        dirs = latest_runs(PROJECT_ROOT / "runs")
    golden = load_golden(Path(args.golden)) if args.golden else load_golden()
    known = {g.company for g in golden}
    runs = {name: Run.model_validate_json((d / "run.json").read_text()) for name, d in sorted(dirs.items())
            if name in known}

    judge = AnthropicJudge(cost_cap=args.cost_cap)
    ev = evaluate(golden, runs, judge)
    ev.judge_cost, ev.judge_calls = judge.spent, judge.calls

    report = format_report(ev)
    out_dir = PROJECT_ROOT / "runs" / "evaluations" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text(report)
    (out_dir / "judgments.json").write_text(json.dumps([
        {"company": g.company, "subject": c.subject, "claim_type": c.claim_type, "attribute": c.attribute,
         "agent_value": c.value, "snippet": c.snippet, "verdict": j.verdict if j else "unjudged", "reason": j.reason if j else "",
         "golden_value": g.golden[j.golden_index].value if j and j.golden_index is not None else None}
        for g in ev.groups for c, j in zip(g.claims, g.judgments)], indent=1, ensure_ascii=False))
    (out_dir / "judge_calls.json").write_text(json.dumps(
        {"calls": judge.calls, "total_cost_usd": round(judge.spent, 5), "run_dirs": {k: str(v) for k, v in dirs.items()}},
        indent=2))
    print(report, end="")
    print(f"Saved: {out_dir.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
