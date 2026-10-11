"""
pii_check.py: finds email addresses and phone numbers in a run's claims (value and quote, every status) and in its
summary (sentences and headlines), and in the exported CSVs. The product records professional background only.
One exception: a founder's work email in a sourced email claim (verify.py admitted it from the company's own
site), and in people.csv only in the Email column. Phone numbers are flagged everywhere.
Counts and locations are reported; the matched text is never printed.

Usage (from the project root, venv active):
    python -m src.pii_check runs/<id> [runs/<id> ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import csv

from schema import ClaimStatus, ClaimType
from src.assemble import load_run
from src.summary import load_summary

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# A phone number: an optional +, then digits with spaces, dots, dashes or brackets between them, at least 9 digits.
PHONE = re.compile(r"(?<![\w.])\+?\(?\d[\d\s().-]{7,}\d(?![\w])")


def find_pii(text: str) -> list[str]:
    """Kinds found in the text: 'email', 'phone'."""
    found = []
    if EMAIL.search(text):
        found.append("email")
    if any(sum(ch.isdigit() for ch in m.group(0)) >= 9 and not re.fullmatch(r"\d{4}\s*[-–]\s*\d{4}", m.group(0).strip())
           for m in PHONE.finditer(text)):
        found.append("phone")
    return found


def check_run(run_dir: Path) -> list[tuple[str, str, str]]:
    """(where, field, kind) for every hit in the claims and the summary of one run folder."""
    run, hits = load_run(run_dir), []
    for c in run.claims:
        admitted_email = c.claim_type == ClaimType.EMAIL and c.status == ClaimStatus.SOURCED
        for field, text in (("value", c.value), ("quote", c.quote or "")):
            hits += [(c.claim_id, field, kind) for kind in find_pii(text) if not (admitted_email and kind == "email")]
    summary = load_summary(run_dir) or {}
    for i, s in enumerate(summary.get("sentences", [])):
        hits += [(f"summary sentence {i + 1}", "text", kind) for kind in find_pii(s.get("text", ""))]
    for h in summary.get("founder_headlines", []):
        hits += [("summary headline", "text", kind) for kind in find_pii(h.get("text", ""))]
    return hits


def check_exports(csv_dir: Path) -> list[tuple[str, str, str]]:
    """(file and row, column, kind) for every hit in the exported CSVs; an email is allowed only in people.csv Email."""
    hits = []
    for path in sorted(csv_dir.glob("*.csv")):
        with path.open(encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f), start=2):
                for column, text in row.items():
                    hits += [(f"{path.name} row {i}", column, kind) for kind in find_pii(text or "")
                             if not (kind == "email" and path.name == "people.csv" and column == "Email")]
    return hits


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m src.pii_check runs/<id> [runs/<id> ...]")
    total = 0
    for arg in sys.argv[1:]:
        run = load_run(Path(arg))
        n_claims = len(run.claims)
        hits = check_run(Path(arg))
        total += len(hits)
        print(f"{run.input.company_name}: {n_claims} claims checked, {len(hits)} hits"
              + "".join(f"\n  {where} / {field}: {kind}" for where, field, kind in hits))
    print(f"Total hits: {total}")


if __name__ == "__main__":
    main()
