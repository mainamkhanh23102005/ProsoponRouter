from __future__ import annotations

import unittest

from src.solvers import sentiment_solver


class SentimentSolverTest(unittest.TestCase):
    def assert_label(self, text: str, expected: str) -> None:
        answer, confidence = sentiment_solver.solve({"text": text})
        self.assertEqual(answer, expected)
        self.assertGreaterEqual(confidence, 0.92)

    def assert_declines(self, text: str) -> None:
        answer, confidence = sentiment_solver.solve({"text": text})
        self.assertIsNone(answer)
        self.assertEqual(confidence, 0.0)

    def test_negation_flips_bad(self) -> None:
        self.assert_label("not bad at all", "positive")

    def test_negation_flips_good(self) -> None:
        self.assert_label("not good", "negative")

    def test_sarcastic_best_worst_declines(self) -> None:
        self.assert_declines("This is the best worst thing ever")

    def test_mild_ambivalence_declines_or_neutral(self) -> None:
        answer, confidence = sentiment_solver.solve({"text": "It's fine, I guess."})
        self.assertIn(answer, {None, "neutral"})
        if answer is None:
            self.assertEqual(confidence, 0.0)
        else:
            self.assertGreaterEqual(confidence, 0.92)

    def test_sarcasm_marker_declines(self) -> None:
        self.assert_declines("GREAT. Just great. Another bug.")

    def test_empty_declines(self) -> None:
        self.assert_declines("")

    def test_gibberish_declines(self) -> None:
        self.assert_declines("asdkjfh qwoiuey")

    def test_neither_good_nor_bad_is_neutral(self) -> None:
        self.assert_label("The movie was neither good nor bad", "neutral")

    def test_long_neutral_factual_text(self) -> None:
        self.assert_label("The meeting is at 3pm on Tuesday in room 204.", "neutral")

    def test_mixed_real_magnitude_declines(self) -> None:
        self.assert_declines("Great acting but terrible pacing and a boring ending")


if __name__ == "__main__":
    unittest.main()

