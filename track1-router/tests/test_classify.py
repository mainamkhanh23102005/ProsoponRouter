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

    def test_natural_language_math_request_is_not_stolen_by_factual_rule(self) -> None:
        self.assertEqual(classify({"text": "What is half of 18 minus 3?"}), "math")

    def test_schedule_times_question_is_not_math(self) -> None:
        self.assertNotEqual(classify({"prompt": "What times are available on July 11?"}), "math")

    def test_mod_concept_question_is_not_math(self) -> None:
        prompt = (
            "What does the 'mod' function do in Python, and why might you use it "
            "in a program that has 5 inputs?"
        )
        self.assertNotEqual(classify({"prompt": prompt}), "math")

    def test_mod_word_math_request_is_classified(self) -> None:
        self.assertEqual(classify({"text": "Calculate 45 mod 6."}), "math")

    def test_factual_request_is_classified(self) -> None:
        self.assertEqual(
            classify({"prompt": "Answer this factual knowledge question: What is the capital of Vietnam?"}),
            "factual knowledge",
        )

    def test_natural_factual_question_is_classified(self) -> None:
        self.assertEqual(
            classify({"prompt": "What is the capital of Australia, and what body of water is it near?"}),
            "factual knowledge",
        )

    def test_constraint_puzzle_is_logical_reasoning(self) -> None:
        prompt = (
            "Five friends sit in a row of seats. Ivy does not sit in seat 1, "
            "and the tea drinker sits immediately after Ivy. Who sits in seat 1?"
        )
        self.assertEqual(classify({"prompt": prompt}), "logical reasoning")

    def test_civics_difference_question_is_not_logical_reasoning(self) -> None:
        prompt = (
            "What is the difference between a senator and a representative, "
            "and how are they assigned to committees?"
        )
        self.assertEqual(classify({"prompt": prompt}), "factual knowledge")

    def test_pet_ownership_constraint_puzzle_is_logical_reasoning(self) -> None:
        prompt = (
            "Three people each own a different pet. Sam does not own the bird. "
            "Jo owns the dog. Who owns the bird?"
        )
        self.assertEqual(classify({"prompt": prompt}), "logical reasoning")

    def test_hard_logic_seating_puzzle_is_logical_reasoning(self) -> None:
        prompt = (
            "Five friends - Ivy, Jude, Kai, Lena, and Moss - sit in a row of 5 seats numbered 1 to 5. "
            "Ivy does not sit in seat 1. Moss does not sit in seat 5. "
            "The person who drinks tea sits immediately after Ivy. Kai does not drink water, "
            "and Kai does not sit immediately before Lena. Who sits in seat 1?"
        )
        self.assertEqual(classify({"prompt": prompt}), "logical reasoning")


if __name__ == "__main__":
    unittest.main()
