"""
golden.py: read the golden set with the standard library only.

Two forms of the same data, both private (golden/ is gitignored: the answer key holds facts about real founders):
    golden/golden_set.xlsx      the workbook, with a `notes` column and a 'why I picked it' column
    golden/golden_*.csv         the CSV copy without those two columns (what evaluate.py and the agent CLI read)
data/golden_*.example.csv shows the format with invented rows.

An .xlsx file is a zip of XML files; read_sheet() reads one sheet's cell values by column letter. read_table()
reads either form and returns rows the same way ({column letter: text}, empty cells absent), so evaluate.py and the
agent CLI work from either. `python -m src.golden export-csv` rewrites the CSV copy in golden/ from the workbook.
"""

from __future__ import annotations

import csv
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from schema import BatchRow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = PROJECT_ROOT / "golden" / "golden_set.xlsx"          # private workbook, gitignored
GOLDEN_DIR = PROJECT_ROOT / "golden"
GOLDEN_INPUT = GOLDEN_DIR / "golden_input.csv"                      # the Input tab without 'why I picked it'
GOLDEN_ANSWER_KEY = GOLDEN_DIR / "golden_answer_key.csv"            # the Answer key tab without `notes`
EXAMPLE_INPUT = PROJECT_ROOT / "data" / "golden_input.example.csv"  # invented rows, same columns (committed)
EXAMPLE_ANSWER_KEY = PROJECT_ROOT / "data" / "golden_answer_key.example.csv"

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
_M = "{%s}" % _NS["m"]


def read_sheet(path: Path, sheet_name: str) -> list[dict[str, str]]:
    """Rows of one sheet as {column letter: text}. Empty rows come back as {}."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(_M + "t")))
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        for sheet in workbook.find("m:sheets", _NS):
            if sheet.get("name") != sheet_name:
                continue
            target = rels[sheet.get("{%s}id" % _NS["r"])].lstrip("/").removeprefix("xl/")
            rows = []
            for row in ET.fromstring(z.read("xl/" + target)).iter(_M + "row"):
                cells = {}
                for c in row.findall("m:c", _NS):
                    column = re.match(r"[A-Z]+", c.get("r")).group()
                    if c.get("t") == "inlineStr":
                        cells[column] = "".join(t.text or "" for t in c.iter(_M + "t"))
                    elif (v := c.find("m:v", _NS)) is not None:
                        cells[column] = shared[int(v.text)] if c.get("t") == "s" else v.text
                rows.append(cells)
            return rows
    raise KeyError(f"sheet {sheet_name!r} not found in {path}")


def column_letter(index: int) -> str:
    """0 -> 'A', 25 -> 'Z', 26 -> 'AA'."""
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def read_table(path: Path, sheet_name: str) -> list[dict[str, str]]:
    """Rows as {column letter: text} from the workbook (named sheet) or from a CSV file (sheet_name ignored)."""
    if Path(path).suffix.lower() != ".csv":
        return read_sheet(Path(path), sheet_name)
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [{column_letter(i): v for i, v in enumerate(row) if v != ""} for row in csv.reader(f)]


def export_csv_copy(workbook: Path = GOLDEN_PATH, out_dir: Path = GOLDEN_DIR) -> dict[str, int]:
    """Write the CSV copy from the private workbook. The Answer key keeps columns A-H (everything except
    `notes`); the Input tab keeps A-E (everything except 'why I picked it'). Only rows with a company name."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for sheet, last_column, target in (("Answer key", "H", out_dir / "golden_answer_key.csv"),
                                       ("Input", "E", out_dir / "golden_input.csv")):
        rows = read_sheet(workbook, sheet)
        letters = [column_letter(i) for i in range(ord(last_column) - 64)]
        header = [rows[0].get(letter, "") for letter in letters]
        assert not any(h.strip().lower().startswith(("notes", "why i picked")) for h in header), header
        body = [[row.get(letter, "") for letter in letters] for row in rows[1:] if row.get("A")]
        with target.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(body)
        counts[target.name] = len(body)
    return counts


def clean_founder_name(raw: str) -> str:
    """'Jonas (Joni) Weber' -> 'Jonas Weber'. Parenthetical notes are aliases or remarks, not the name."""
    return " ".join(re.sub(r"\([^)]*\)", " ", raw).split())


def first_url(raw: str) -> str | None:
    """'https://www.antler.co/portfolio (filters: ...)' -> 'https://www.antler.co/portfolio'."""
    match = re.search(r"https?://\S+", raw or "")
    return match.group().rstrip(").,;") if match else None


def read_input_rows(path: Path = GOLDEN_INPUT) -> list[tuple[BatchRow, list[str]]]:
    """Each Input row as (BatchRow, raw founder strings). The raw strings keep aliases, e.g. '(Joni)',
    which the agent is told about but which are not part of the founder's name."""
    rows = read_table(path, "Input")
    out = []
    for cells in rows[1:]:                       # row 1 is the header
        if not cells.get("A"):
            continue
        raw_founders = [f.strip() for f in cells.get("D", "").split(";") if f.strip()]
        out.append((BatchRow(
            company_name=cells["A"].strip(),
            company_url=first_url(cells.get("B", "")) or cells.get("B", "").strip(),
            accelerator_profile_url=first_url(cells.get("C", "")),
            founder_names=[clean_founder_name(f) for f in raw_founders],
        ), raw_founders))
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["export-csv"]:
        print(export_csv_copy())
    else:
        raise SystemExit("usage: python -m src.golden export-csv")
