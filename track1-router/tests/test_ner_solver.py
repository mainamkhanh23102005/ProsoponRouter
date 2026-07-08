from __future__ import annotations

import unittest

from src.solvers import ner_solver


def spacy_model_available() -> bool:
    return ner_solver.load_spacy_model() is not None


class NerSolverTest(unittest.TestCase):
    def assert_entities(self, text: str, expected: str) -> None:
        answer, confidence = ner_solver.solve({"text": text})
        self.assertIsInstance(answer, str)
        self.assertEqual(answer, expected)
        self.assertGreaterEqual(confidence, 0.90)

    def assert_declines(self, text: str) -> None:
        answer, confidence = ner_solver.solve({"text": text})
        self.assertIsNone(answer)
        self.assertEqual(confidence, 0.0)

    def test_numeric_date_regex(self) -> None:
        self.assert_entities("The launch date is 2026-07-11.", "DATE: 2026-07-11")

    def test_slash_date_regex(self) -> None:
        self.assert_entities("Schedule it for 07/08/2026.", "DATE: 07/08/2026")

    def test_email_money_percent_are_not_official_entities(self) -> None:
        self.assert_entities("Email dan@example.com, pay $25.50, and apply 25%.", "NONE")

    def test_zero_official_entities_returns_empty_list(self) -> None:
        self.assert_entities("The dashboard loaded after three seconds.", "NONE")

    @unittest.skipUnless(spacy_model_available(), "en_core_web_sm is not installed")
    def test_person_and_org(self) -> None:
        self.assert_entities(
            "Satya Nadella leads Microsoft.",
            "PERSON: Satya Nadella; ORG: Microsoft",
        )

    @unittest.skipUnless(spacy_model_available(), "en_core_web_sm is not installed")
    def test_location(self) -> None:
        self.assert_entities("The meeting is in Hanoi.", "LOCATION: Hanoi")

    @unittest.skipUnless(spacy_model_available(), "en_core_web_sm is not installed")
    def test_mixed_official_entities_preserve_order(self) -> None:
        self.assert_entities(
            "Alice Johnson met Google in London on 2026-08-01.",
            "PERSON: Alice Johnson; ORG: Google; LOCATION: London; DATE: 2026-08-01",
        )

    @unittest.skipUnless(spacy_model_available(), "en_core_web_sm is not installed")
    def test_multiple_locations(self) -> None:
        self.assert_entities(
            "The team moved from Paris to Berlin.",
            "LOCATION: Paris; LOCATION: Berlin",
        )


if __name__ == "__main__":
    unittest.main()
