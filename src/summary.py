"""
summary.py: the optional summary, built last.

One model call per company whose only input is the company's sourced claims (ids, subjects, values, origins,
confirmation). It returns 3-4 sentences and one headline per founder, each citing the claim ids it uses. Code then
checks the result: every cited id must be a sourced claim, every sentence must cite at least one, and no evaluative
word may appear. If the check fails, the model gets one retry with the problems listed; any sentence that still fails
is dropped in code and recorded in the assembly notes (never a warning on the page). The summary is labelled "Summary
generated from sourced claims".

Usage (from the project root, venv active):
    python -m src.summary runs/<id> [runs/<id> ...]      writes summary.json into each run folder
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Callable, Optional

from schema import ClaimType, CompanyDossier, ConfirmationStatus, Fact
from src.keyfacts import pick_key_facts
from src.assemble import assemble, load_run
from src.llm import Usage, cost_usd, read_env_key

LABEL = "Summary generated from sourced claims"
SUMMARY_MODEL = "claude-sonnet-5-5"
EFFORT = "medium"
MAX_SENTENCES = 4
MAX_WORDS = 45                 # per sentence; a summary is for skimming

# Words that judge rather than report. Checked in code; the prompt also forbids them.
# Words that also occur in factual quotes ("leading product development") are left out to avoid false alarms.
EVALUATIVE_WORDS = ["strong", "impressive", "promising", "exceptional", "outstanding", "remarkable", "excellent",
                    "world-class", "innovative", "cutting-edge", "great", "successful", "talented",
                    "experienced", "seasoned", "visionary", "proven", "notable"]

SYSTEM = f"""You write a short, factual summary of a startup for a venture analyst, using ONLY the claims given.
Each claim has an id, a subject (a founder's name or "company"), a type, a value, the origin of its source
(founder = written by the founder, company = the company's own marketing, counterparty = an accelerator, investor,
employer or university speaking for itself, database, media) and a confirmation status.

Write 3-4 sentences, in this order:
1. The company profile, always first: what the company does, who it is for and how it sells, from the claims marked
   "profile": true (company descriptions and business model; the list is "profile_ids"). Cite those claims.
2. The key facts: the claims marked "key": true, all of them, in the order of the "key_order" list.
3. Who the founders are (prior ventures and their outcomes first, then the most relevant roles or education).
4. The company's stage and traction signals, then its backers.
If there are no profile claims, start with the key facts.
Rules:
- Use only facts in the claims. Copy figures, dates and titles as written. Do not add anything.
- Keep each sentence under {MAX_WORDS} words: pick the most important facts, leave minor ones out.
- Leave out filler: demo or pilot buttons, compliance work in progress, and individual job postings.
- Attribute what the company says about itself: "the company says ...". For what a founder wrote about themselves
  (origin founder), write "(own claim)" once per founder, after the first fact you take from them, and not again for
  that founder. Never write "according to his/her own profile". State counterparty facts with their source
  ("YC lists ...", "an investor's page lists ...").
- If claims disagree (status sources_disagree), say the sources differ; do not pick one.
- No evaluative words: never use {", ".join(EVALUATIVE_WORDS)}, or any similar judgement.
- Every sentence lists the ids of the claims it uses. Skip a topic if there are no claims for it.
Also write one headline per founder who has claims: prior ventures with their outcomes first, then the most
notable education or role, as a short plain line (not a sentence), with the ids it uses."""

SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {"type": "array", "items": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "claim_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["text", "claim_ids"], "additionalProperties": False}},
        "founder_headlines": {"type": "array", "items": {
            "type": "object",
            "properties": {"founder": {"type": "string"}, "text": {"type": "string"},
                           "claim_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["founder", "text", "claim_ids"], "additionalProperties": False}},
    },
    "required": ["sentences", "founder_headlines"], "additionalProperties": False,
}


def sourced_facts(dossier: CompanyDossier) -> list[Fact]:
    """Every page-sourced fact in the main fields. Leads are not facts, and snippet-status facts are left out too:
    the summary's only input is sourced claims."""
    facts = [f for fs in dossier.facts.values() for f in fs]
    facts += [f for founder in dossier.founders for fs in founder.facts.values() for f in fs]
    return [f for f in facts if f.confirmation != ConfirmationStatus.SNIPPET]


def key_claim_ids(dossier: CompanyDossier) -> list[str]:
    """Claim ids of the key facts, in key-fact order (first claim of each fact; page-sourced facts only)."""
    return [k.fact.claims[0].claim_id for k in pick_key_facts(dossier) if k.fact.confirmation != ConfirmationStatus.SNIPPET]


PROFILE_TYPES = (ClaimType.COMPANY_DESCRIPTION, ClaimType.BUSINESS_MODEL)


def profile_claim_ids(dossier: CompanyDossier) -> list[str]:
    """Ids of the sourced claims that say what the company does and how it sells (the first sentence's material)."""
    return [c.claim_id for f in sourced_facts(dossier) if f.claim_type in PROFILE_TYPES for c in f.claims]


def summary_input(dossier: CompanyDossier) -> list[dict]:
    """The only input the summary model sees: one line per sourced claim, with the claims of the key facts and the
    company-profile claims marked."""
    keys = key_claim_ids(dossier)
    profile = set(profile_claim_ids(dossier))
    rows = []
    for fact in sourced_facts(dossier):
        for c in fact.claims:
            rows.append({"id": c.claim_id, "subject": c.subject, "type": c.claim_type.value,
                         "attribute": c.attribute.value if c.attribute else None, "value": c.value,
                         "origin": c.source_origin.value, "confirmation": fact.confirmation.value,
                         "signal": c.signal.value if c.signal else None, "key": c.claim_id in keys,
                         "profile": c.claim_id in profile})
    return rows


def piece_problems(kind: str, piece: dict, known_ids: set[str]) -> list[str]:
    """Problems with one sentence or headline: cites no claim, cites an id that is not a sourced claim, too long,
    'own profile', evaluative words."""
    problems = []
    ids = piece.get("claim_ids", [])
    if not ids:
        problems.append(f"{kind} cites no claim")
    unknown = [i for i in ids if i not in known_ids]
    if unknown:
        problems.append(f"{kind} cites ids that are not sourced claims: {', '.join(unknown)}")
    if kind == "sentence" and len(piece.get("text", "").split()) > MAX_WORDS:
        problems.append(f"sentence has {len(piece['text'].split())} words (max {MAX_WORDS})")
    if re.search(r"\bown (profile|bio|page)\b", piece.get("text", ""), re.I):
        problems.append(f"{kind} says 'own profile' (write '(own claim)' once per founder instead)")
    words = [w for w in EVALUATIVE_WORDS if re.search(rf"\b{re.escape(w)}\b", piece.get("text", ""), re.I)]
    if words:
        problems.append(f"{kind} uses evaluative words: {', '.join(words)}")
    return problems


def check_summary(result: dict, known_ids: set[str], profile_ids: Optional[set[str]] = None) -> list[str]:
    """Problems with a model's summary, found in code. Empty list = it passes. When the company has profile claims
    (description, business model) the first sentence must cite one of them: it is the company profile."""
    problems = []
    sentences = result.get("sentences", [])
    if profile_ids and sentences and not set(sentences[0].get("claim_ids", [])) & profile_ids:
        problems.append("the first sentence must be the company profile: cite a profile claim (company description or business model)")
    if not 1 <= len(sentences) <= MAX_SENTENCES:
        problems.append(f"{len(sentences)} sentences (want 3-{MAX_SENTENCES})")
    pieces = [("sentence", s) for s in sentences] + [("headline", h) for h in result.get("founder_headlines", [])]
    for kind, piece in pieces:
        problems += [f"{p}: {piece.get('text', '')[:60]!r}" if "cites no claim" in p else p
                     for p in piece_problems(kind, piece, known_ids)]
    return problems


def drop_failing(result: dict, known_ids: set[str]) -> tuple[dict, list[dict]]:
    """What is left of a summary after the retry: every sentence or headline that still fails a check is dropped, and
    sentences past MAX_SENTENCES are cut. Returns (the clean summary, a record of each drop with its reasons)."""
    dropped, sentences = [], []
    for s in result.get("sentences", []):
        problems = piece_problems("sentence", s, known_ids)
        if problems:
            dropped.append({"kind": "sentence", "text": s.get("text", ""), "problems": problems})
        elif len(sentences) >= MAX_SENTENCES:
            dropped.append({"kind": "sentence", "text": s.get("text", ""), "problems": [f"more than {MAX_SENTENCES} sentences"]})
        else:
            sentences.append(s)
    headlines = []
    for h in result.get("founder_headlines", []):
        problems = piece_problems("headline", h, known_ids)
        if problems:
            dropped.append({"kind": "headline", "text": h.get("text", ""), "problems": problems})
        else:
            headlines.append(h)
    return {**result, "sentences": sentences, "founder_headlines": headlines}, dropped


# A model call takes (system, user message) and returns (parsed JSON, model name, usage).
ModelCall = Callable[[str, str], tuple[dict, str, Usage]]


def anthropic_call(system: str, user: str) -> tuple[dict, str, Usage]:
    import anthropic
    client = anthropic.Anthropic(api_key=read_env_key("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=SUMMARY_MODEL, max_tokens=4000, system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}})
    u = response.usage
    usage = Usage(u.input_tokens or 0, u.cache_creation_input_tokens or 0, u.cache_read_input_tokens or 0,
                  u.output_tokens or 0)
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), response.model, usage


def summarise_dossier(dossier: CompanyDossier, call: ModelCall = anthropic_call) -> dict:
    """Ask for a summary, check it, retry once if the check fails, then drop any sentence that still fails (the drops are
    saved in "dropped" and appear in the assembly notes). Returns what is saved to summary.json."""
    claims = summary_input(dossier)
    known = {c["id"] for c in claims}
    if not claims:
        return {"label": LABEL, "sentences": [], "founder_headlines": [], "problems": ["no sourced claims"],
                "cost_usd": 0.0, "calls": []}
    profile = profile_claim_ids(dossier)
    user = ("key_order: " + json.dumps(key_claim_ids(dossier)) + "\nprofile_ids: " + json.dumps(profile)
            + "\nCLAIMS (JSON, one per line):\n"
            + "\n".join(json.dumps(c, ensure_ascii=False) for c in claims))
    calls, result, problems = [], {}, []
    for attempt in range(2):
        message = user if attempt == 0 else (
            user + "\n\nYour previous answer had these problems; fix them:\n- " + "\n- ".join(problems)
            + "\nPrevious answer:\n" + json.dumps(result, ensure_ascii=False))
        result, model, usage = call(SYSTEM, message)
        calls.append({"model": model, "input_tokens": usage.total_input, "output_tokens": usage.output_tokens,
                      "cost_usd": round(cost_usd(model, usage), 5)})
        problems = check_summary(result, known, set(profile))
        if not problems:
            break
    dropped = []
    if problems:                                  # still failing after the retry: drop what fails, never warn on the page
        result, dropped = drop_failing(result, known)
    return {"label": LABEL, "sentences": result.get("sentences", []),
            "founder_headlines": result.get("founder_headlines", []), "problems": [], "dropped": dropped,
            "cost_usd": round(sum(c["cost_usd"] for c in calls), 5), "calls": calls}


def load_summary(run_dir: Path) -> Optional[dict]:
    path = run_dir / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m src.summary runs/<id> [runs/<id> ...]")
    total = 0.0
    for arg in sys.argv[1:]:
        run_dir = Path(arg)
        run = load_run(run_dir)
        summary = summarise_dossier(assemble(run).dossier)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        total += summary["cost_usd"]
        print(f"{run.input.company_name}: ${summary['cost_usd']:.4f} in {len(summary['calls'])} call(s); "
              f"problems: {summary['problems'] or 'none'}")
    print(f"Total: ${total:.4f}")


if __name__ == "__main__":
    main()
