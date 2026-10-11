"""
Tests for src/orgnames.py. Run from the project root:  python -m unittest discover tests
"""

import unittest

from src.orgnames import organisation_variants, same_organisation


class OrgNameTests(unittest.TestCase):
    def test_variants_of_the_same_organisation(self):
        for a, b in [("UC Berkeley", "University of California, Berkeley"), ("UT Austin", "The University of Texas at Austin"),
                     ("MIT", "Massachusetts Institute of Technology"), ("Stanford", "Stanford University"),
                     ("Orion", "Orion Labs"), ("Los Alamos National Lab", "Los Alamos National Laboratory"),
                     ("ETH Zurich", "ETH Zürich")]:
            self.assertTrue(same_organisation(a, b), (a, b))

    def test_different_organisations(self):
        for a, b in [("University of Toronto", "University of Texas"), ("Carleton University", "University of Toronto"),
                     ("University", "University of Texas"), ("Berkeley Lights", "UC Berkeley"), ("", "Acme")]:
            self.assertFalse(same_organisation(a, b), (a, b))

    def test_text_variants(self):
        self.assertIn("University of California, Berkeley", organisation_variants("UC Berkeley"))
        self.assertIn("UT Austin", organisation_variants("The University of Texas at Austin"))
        self.assertEqual(organisation_variants("Acme Corp"), ["Acme Corp"])


if __name__ == "__main__":
    unittest.main()
