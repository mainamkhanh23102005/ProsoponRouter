from __future__ import annotations

import unittest

from src import config


class ConfigTest(unittest.TestCase):
    def test_sentiment_fallback_is_honest_empty_answer(self) -> None:
        self.assertEqual(config.FALLBACK_ANSWERS["sentiment"], "")

    def test_low_budget_model_categories_have_reasoning_headroom(self) -> None:
        self.assertGreaterEqual(config.POLICY["factual knowledge"].max_tokens, 64)
        self.assertGreaterEqual(config.POLICY["sentiment"].max_tokens, 80)
        self.assertGreaterEqual(config.POLICY["logical reasoning"].max_tokens, 150)


if __name__ == "__main__":
    unittest.main()
