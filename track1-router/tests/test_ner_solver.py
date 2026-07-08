from __future__ import annotations

import unittest

from src.solvers import ner_solver


def spacy_model_available() -> bool:
    return ner_solver.load_spacy_model() is not None


class NerSolverTest(unittest.TestCase):
    def assert_entities(self, text: str, expected: list[dict[str, str]]) -> None:
        answer, confidence = ner_solver.solve({"text": text})
        self.assertEqual(answer, expected)
        self.assertGreaterEqual(confidence, 0.90)

    def assert_declines(self, text: str) -> None:
        answer, confidence = ner_solver.solve({"text": text})
        self.assertIsNone(answer)
        self.assertEqual(confidence, 0.0)

    def test_numeric_date_regex(self) -> None:
        self.assert_entities("The launch date is 2026-07-11.", [{"text": "2026-07-11", "label": "DATE"}])

    def test_slash_date_regex(self) -> None:
        self.assert_entities("Schedule it for 07/08/2026.", [{"text": "07/08/2026", "label": "DATE"}])

    def test_email_money_percent_are_not_official_entities(self) -> None:
        self.assert_declines("Email dan@example.com, pay $25.50, and apply 25%.")

    @unittest.skipUnless(spacy_model_available(), "en_core_web_sm is not installed")
    def test_person_and_org(self) -> None:
        self.assert_entities(
            "Satya Nadella leads Microsoft.",
            [
                {"text": "Satya Nadella", "label": "PERSON"},
                {"text": "Microsoft", "label": "ORG"},
            ],
        )

    @unittest.skipUnless(spacy_model_available(), "en_core_web_sm is not installed")
    def test_location(self) -> None:
        self.assert_entities("The meeting is in Hanoi.", [{"text": "Hanoi", "label": "LOCATION"}])

    @unittest.skipUnless(spacy_model_available(), "en_core_web_sm is not installed")
    def test_mixed_official_entities_preserve_order(self) -> None:
        self.assert_entities(
            "Alice Johnson met Google in London on 2026-08-01.",
            [
                {"text": "Alice Johnson", "label": "PERSON"},
                {"text": "Google", "label": "ORG"},
                {"text": "London", "label": "LOCATION"},
                {"text": "2026-08-01", "label": "DATE"},
            ],
        )

    @unittest.skipUnless(spacy_model_available(), "en_core_web_sm is not installed")
    def test_multiple_locations(self) -> None:
        self.assert_entities(
            "The team moved from Paris to Berlin.",
            [
                {"text": "Paris", "label": "LOCATION"},
                {"text": "Berlin", "label": "LOCATION"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
