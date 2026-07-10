from __future__ import annotations

import unittest

from src import config


class ConfigTest(unittest.TestCase):
    def test_sentiment_fallback_is_honest_empty_answer(self) -> None:
        self.assertEqual(config.FALLBACK_ANSWERS["sentiment"], "")

    def test_low_budget_model_categories_have_reasoning_headroom(self) -> None:
        self.assertGreaterEqual(config.POLICY["factual knowledge"].max_tokens, 120)
        self.assertGreaterEqual(config.POLICY["sentiment"].max_tokens, 80)
        self.assertGreaterEqual(config.POLICY["logical reasoning"].max_tokens, 110)
        self.assertGreaterEqual(config.POLICY["unknown"].max_tokens, 90)
        self.assertEqual(config.POLICY["factual knowledge"].stop, ())


if __name__ == "__main__":
    unittest.main()
