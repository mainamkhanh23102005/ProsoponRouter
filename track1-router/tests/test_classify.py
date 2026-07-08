from __future__ import annotations

import unittest

from src.classify import classify


class ClassifyTest(unittest.TestCase):
    def test_explicit_category_wins(self) -> None:
        self.assertEqual(classify({"category": "sentiment", "text": "What is 2 + 2?"}), "sentiment")

    def test_positive_word_alone_is_not_sentiment(self) -> None:
        self.assertEqual(classify({"text": "The word positive appears in this sentence."}), "unknown")

    def test_email_or_date_alone_is_not_ner(self) -> None:
        self.assertEqual(classify({"text": "Email dan@example.com on 2026-07-11."}), "unknown")

    def test_sentiment_request_is_classified(self) -> None:
        self.assertEqual(classify({"text": "Classify the sentiment of this review: not bad."}), "sentiment")

    def test_ner_request_is_classified(self) -> None:
        self.assertEqual(classify({"text": "Extract named entities from: Satya Nadella visited Paris."}), "ner")

    def test_math_request_is_classified(self) -> None:
        self.assertEqual(classify({"text": "Calculate 7 + 5 for the total."}), "math")


if __name__ == "__main__":
    unittest.main()

