"""
Tests for src/pii_check.py. Run from the project root:  python -m unittest discover tests
"""

import unittest

from src.pii_check import find_pii


class PiiTests(unittest.TestCase):
    def test_emails(self):
        self.assertEqual(find_pii("Write to jane.doe+x@example-robotics.com for a demo"), ["email"])
        self.assertEqual(find_pii("No address here, @handle is not one"), [])

    def test_phone_numbers(self):
        for text in ("Call +1 (415) 555-0132", "tel 030 1234 5678", "+49 30 12345678", "415.555.0132 ext 4"):
            self.assertEqual(find_pii(text), ["phone"], text)

    def test_ordinary_numbers_are_not_phones(self):
        for text in ("raised $12,000,000 in 2024", "Founded 2020 - 2025", "6 yrs experience", "Seed round of 750000",
                     "10997 Berlin", "SOC 2 Type 1 & 2", "(May 2015 - June 2021)"):
            self.assertEqual(find_pii(text), [], text)


if __name__ == "__main__":
    unittest.main()
