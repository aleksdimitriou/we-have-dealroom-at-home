"""
Tests for the golden-set readers in src/golden.py and the committed example files (data/*.example.csv, invented rows).
The real answer key and input (golden/*.csv) and the workbook (golden/golden_set.xlsx) are private and not in the
repository: the tests that need them are skipped when they are absent.
Run from the project root:  python -m unittest discover tests
"""

import csv
import tempfile
import unittest
from pathlib import Path

from src.evaluate import load_golden
from src.golden import (
    EXAMPLE_ANSWER_KEY, EXAMPLE_INPUT, GOLDEN_ANSWER_KEY, GOLDEN_INPUT, GOLDEN_PATH, column_letter, export_csv_copy,
    read_input_rows, read_table,
)


def rows_of(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


class ReaderTests(unittest.TestCase):
    def test_column_letter(self):
        self.assertEqual([column_letter(i) for i in (0, 7, 25, 26, 27)], ["A", "H", "Z", "AA", "AB"])

    def test_csv_rows_use_column_letters_and_skip_empty_cells(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.csv"
            path.write_text('a,b,c\n1,,3\n"x, y",,\n', encoding="utf-8")
            self.assertEqual(read_table(path, "ignored"), [{"A": "a", "B": "b", "C": "c"}, {"A": "1", "C": "3"}, {"A": "x, y"}])

    def test_a_missing_answer_key_says_it_is_private_and_where_the_format_is(self):
        with self.assertRaises(SystemExit) as raised:
            load_golden(Path("/nonexistent/golden_answer_key.csv"))
        self.assertIn("example", str(raised.exception))


class ExampleFileTests(unittest.TestCase):
    """The committed examples: the same columns as the private files, invented rows only."""

    def test_the_answer_key_example_has_the_columns_and_three_invented_rows(self):
        rows = rows_of(EXAMPLE_ANSWER_KEY)
        self.assertEqual(len(rows[0]), 8)
        self.assertEqual(rows[0][2:4], ["claim_type", "found"])
        self.assertEqual(len(rows) - 1, 3)
        self.assertTrue(all(len(r) == 8 for r in rows))
        text = EXAMPLE_ANSWER_KEY.read_text(encoding="utf-8")
        self.assertIn("Jane Doe", text)
        self.assertTrue(all("example" in r[5] for r in rows[1:]))                       # every source is an example domain

    def test_the_example_key_loads_like_the_real_one(self):
        facts = load_golden(EXAMPLE_ANSWER_KEY)
        self.assertEqual(len(facts), 3)
        self.assertEqual({f.company for f in facts}, {"Example Robotics"})
        self.assertEqual({f.company for f in facts}, {r.company_name for r, _ in read_input_rows(EXAMPLE_INPUT)} - {"Example Foods"})

    def test_the_input_example_has_the_columns_and_no_why_i_picked_it(self):
        rows = rows_of(EXAMPLE_INPUT)
        self.assertEqual(len(rows[0]), 5)
        self.assertFalse(any("why" in h.lower() for h in rows[0]))
        self.assertEqual(len(rows) - 1, 2)


@unittest.skipUnless(GOLDEN_ANSWER_KEY.exists() and GOLDEN_INPUT.exists(), "private golden CSVs not present")
class PrivateCopyTests(unittest.TestCase):
    """golden/*.csv: shape only, so a bad export is noticed."""

    def test_answer_key_has_every_column_except_notes_and_all_rows(self):
        rows = rows_of(GOLDEN_ANSWER_KEY)
        self.assertEqual(len(rows[0]), 8)
        self.assertFalse(any("note" in h.lower() for h in rows[0]))
        self.assertEqual(rows[0], rows_of(EXAMPLE_ANSWER_KEY)[0])                       # the example shows the real format
        self.assertEqual(len(rows) - 1, 99)
        self.assertTrue(all(len(r) == 8 for r in rows))

    def test_input_has_no_why_i_picked_it_column(self):
        rows = rows_of(GOLDEN_INPUT)
        self.assertEqual(rows[0], rows_of(EXAMPLE_INPUT)[0])
        self.assertEqual(len(rows) - 1, 4)

    def test_evaluate_reads_the_private_copy(self):
        facts = load_golden()
        self.assertEqual(len(facts), 92)                              # the found rows
        self.assertEqual({f.company for f in facts}, {r.company_name for r, _ in read_input_rows()})

    def test_agent_input_rows_come_from_the_private_copy(self):
        rows = read_input_rows()
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row.company_url.startswith("http") and row.founder_names for row, _ in rows))


@unittest.skipUnless(GOLDEN_PATH.exists() and GOLDEN_ANSWER_KEY.exists(), "private workbook or CSVs not present")
class WorkbookParityTests(unittest.TestCase):
    def test_workbook_and_csv_give_identical_golden_facts(self):
        self.assertEqual(load_golden(GOLDEN_PATH), load_golden(GOLDEN_ANSWER_KEY))

    def test_input_rows_are_identical_from_both(self):
        self.assertEqual(read_input_rows(GOLDEN_PATH), read_input_rows(GOLDEN_INPUT))

    def test_export_reproduces_the_csv_files_and_drops_notes(self):
        with tempfile.TemporaryDirectory() as d:
            counts = export_csv_copy(GOLDEN_PATH, Path(d))
            self.assertEqual(counts, {"golden_answer_key.csv": 99, "golden_input.csv": 4})
            self.assertEqual((Path(d) / "golden_answer_key.csv").read_text(encoding="utf-8"),
                             GOLDEN_ANSWER_KEY.read_text(encoding="utf-8"))
            self.assertEqual((Path(d) / "golden_input.csv").read_text(encoding="utf-8"),
                             GOLDEN_INPUT.read_text(encoding="utf-8"))
            # the notes text of every row is nowhere in the export
            from src.golden import read_sheet
            exported = (Path(d) / "golden_answer_key.csv").read_text(encoding="utf-8")
            notes = [r["I"] for r in read_sheet(GOLDEN_PATH, "Answer key")[1:] if len(r.get("I", "")) > 60]
            self.assertTrue(notes)
            self.assertFalse(any(n[:60] in exported for n in notes))


if __name__ == "__main__":
    unittest.main()
