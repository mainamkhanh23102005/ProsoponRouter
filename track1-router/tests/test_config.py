from __future__ import annotations

import unittest

from src import config


class ConfigTest(unittest.TestCase):
    def test_sentiment_fallback_is_honest_empty_answer(self) -> None:
        self.assertEqual(config.FALLBACK_ANSWERS["sentiment"], "")


if __name__ == "__main__":
    unittest.main()
