from __future__ import annotations

import unittest

from src.classify import classify


class ClassifyTest(unittest.TestCase):
    def test_explicit_category_wins(self) -> None:
        self.assertEqual(classify({"category": "sentiment", "text": "What is 2 + 2?"}), "sentiment")

    def test_official_category_names_are_normalized(self) -> None:
        cases = {
            "factual_knowledge": "factual knowledge",
            "mathematical_reasoning": "math",
            "sentiment_classification": "sentiment",
            "text_summarization": "summarization",
            "named_entity_recognition": "ner",
        }
        for official, internal in cases.items():
            with self.subTest(category=official):
                self.assertEqual(classify({"category": official, "prompt": "ambiguous"}), internal)

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

    def test_multistep_word_problem_is_classified_as_math(self) -> None:
        prompt = (
            "A tank starts with 480 liters. It drains at 8 liters per minute for 15 minutes, "
            "then is refilled at 12 liters per minute for 20 minutes, then drains again at "
            "5 liters per minute for 10 minutes. How many liters are in the tank now?"
        )
        self.assertEqual(classify({"prompt": prompt}), "math")

    def test_natural_language_math_request_is_not_stolen_by_factual_rule(self) -> None:
        self.assertEqual(classify({"text": "What is half of 18 minus 3?"}), "math")

    def test_inventory_percentage_word_problem_is_math(self) -> None:
        self.assertEqual(
            classify({"prompt": "A warehouse starts with 2,400 units, sells 37% of stock, "
                    "restocks 800 units, and sells 640 units. How many units remain?"}),
            "math",
        )

    def test_recipe_scaling_and_cost_word_problem_is_math(self) -> None:
        self.assertEqual(
            classify({"prompt": "A recipe requires 3/4 cup for 12 cookies. How much is needed "
                    "for 30 cookies if it costs $2.40 per cup?"}),
            "math",
        )

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

    def test_broad_factual_question_forms_are_classified(self) -> None:
        prompts = (
            "Which scientist proposed the three laws of motion?",
            "Who designed the analytical engine?",
            "When was the first programmable computer completed?",
            "Where does the Danube River begin?",
            "What material conducts electricity with zero resistance at low temperatures?",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(classify({"prompt": prompt}), "factual knowledge")

    def test_broad_summarization_imperatives_are_classified(self) -> None:
        prompts = (
            "Provide a short summary of the passage: Bees pollinate many crops.",
            "Give a concise overview of this article: The city expanded its rail network.",
            "Condense the following paragraph into one sentence: Rust prevents many memory errors.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(classify({"prompt": prompt}), "summarization")

    def test_broad_code_generation_imperatives_are_classified(self) -> None:
        prompts = (
            "Generate code that reverses a linked list.",
            "Produce a Python function named median_value for a list of numbers.",
            "Create code that groups records by department.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(classify({"prompt": prompt}), "code generation")

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
