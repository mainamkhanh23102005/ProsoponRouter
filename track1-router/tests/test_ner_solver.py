from __future__ import annotations

import unittest

from src.solvers import ner_solver


class NerSolverTest(unittest.TestCase):
    def assert_entities(self, text: str, expected: list[dict[str, str]]) -> None:
        answer, confidence = ner_solver.solve({"text": text})
        self.assertEqual(answer, expected)
        self.assertGreaterEqual(confidence, 0.90)

    def assert_declines(self, text: str) -> None:
        answer, confidence = ner_solver.solve({"text": text})
        self.assertIsNone(answer)
        self.assertEqual(confidence, 0.0)

    def test_adjacent_entities_are_bounded(self) -> None:
        self.assert_entities(
            "Meeting on 2026-07-11 at 3pm, email dan@example.com, cost $25.50 (25% discount).",
            [
                {"text": "2026-07-11", "label": "DATE"},
                {"text": "dan@example.com", "label": "EMAIL"},
                {"text": "$25.50", "label": "MONEY"},
                {"text": "25%", "label": "PERCENT"},
            ],
        )

    def test_invalid_email_declines(self) -> None:
        self.assert_declines("user@@doubled..com")

    def test_numeric_date_matches(self) -> None:
        self.assert_entities("07/08/2026", [{"text": "07/08/2026", "label": "DATE"}])

    def test_natural_language_date_declines(self) -> None:
        self.assert_declines("the 4th of July")

    def test_no_entities_declines(self) -> None:
        self.assert_declines("The weather is nice today.")

    def test_money_words_known_gap_declines(self) -> None:
        self.assert_declines("It costs 25 dollars and 50 cents")

    def test_percent_not_phone_number(self) -> None:
        self.assert_entities("Call 100% at 555-0142", [{"text": "100%", "label": "PERCENT"}])

    def test_multiple_emails_preserve_order(self) -> None:
        self.assert_entities(
            "cc jane@x.com and john@y.com",
            [
                {"text": "jane@x.com", "label": "EMAIL"},
                {"text": "john@y.com", "label": "EMAIL"},
            ],
        )


if __name__ == "__main__":
    unittest.main()

