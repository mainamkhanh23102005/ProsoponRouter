from __future__ import annotations

import unittest

from src.solvers import math_solver


class MathSolverTest(unittest.TestCase):
    def assert_answer(self, text: str, expected: str) -> None:
        answer, confidence = math_solver.solve({"text": text})
        self.assertEqual(answer, expected)
        self.assertGreaterEqual(confidence, 0.95)

    def assert_declines(self, text: str) -> None:
        answer, confidence = math_solver.solve({"text": text})
        self.assertIsNone(answer)
        self.assertEqual(confidence, 0.0)

    def test_word_half_minus(self) -> None:
        self.assert_answer("What is half of 18 minus 3?", "6")

    def test_add_then_multiply(self) -> None:
        self.assert_answer("Add 5 and 7, then multiply by 2", "24")

    def test_divided_by_chain(self) -> None:
        self.assert_answer("What's 100 divided by 4, divided by 5?", "5")

    def test_equality_question_declines(self) -> None:
        self.assert_declines("Is 2 + 2 equal to 5?")

    def test_double_negative(self) -> None:
        self.assert_answer("3 - -2", "5")

    def test_caret_exponent(self) -> None:
        self.assert_answer("2^10", "1024")

    def test_python_exponent(self) -> None:
        self.assert_answer("2**10", "1024")

    def test_modulo_word(self) -> None:
        self.assert_answer("What is 7 mod 3?", "1")

    def test_empty_or_no_numbers_declines(self) -> None:
        self.assert_declines("Hello there")

    def test_unrelated_numbers_declines(self) -> None:
        self.assert_declines("I have 3 apples and my friend has 5 apples. Today is the 3rd.")

    def test_negative_result(self) -> None:
        self.assert_answer("3 - 10", "-7")

    def test_decimal_result(self) -> None:
        self.assert_answer("10 / 4", "2.5")

    def test_dash_separated_date_declines(self) -> None:
        self.assert_declines("7-11-2026")

    def test_slash_separated_date_declines(self) -> None:
        self.assert_declines("7/11/2026")

    def test_trailing_sentence_period_after_parenthesized_expression(self) -> None:
        self.assert_answer("Calculate 7 + (6 * 5).", "37")

    def test_percent_symbol_modulo(self) -> None:
        self.assert_answer("Calculate 45 % 6.", "3")

    def test_tank_rate_word_problem(self) -> None:
        self.assert_answer(
            "A tank starts with 480 liters. It drains at 8 liters per minute for 15 minutes, "
            "then is refilled at 12 liters per minute for 20 minutes, then drains again at "
            "5 liters per minute for 10 minutes. How many liters are in the tank now?",
            "550",
        )

    def test_fraction_inside_multistep_word_problem_declines(self) -> None:
        self.assert_declines(
            "Paul has 52 marbles. His friend gave him 28 marbles. Then, he lost "
            "1/4 of his marbles. How many marbles does Paul have left?"
        )

    def test_inventory_percent_restock_sequence(self) -> None:
        self.assert_answer(
            "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. "
            "In Q2 it restocks 800 units. In Q3 it sells 640 units. "
            "How many units remain at the end of Q3?",
            "1672",
        )

    def test_recipe_scaling_with_cost_returns_both_requested_values(self) -> None:
        self.assert_answer(
            "A recipe requires 3/4 cup of sugar for 12 cookies. How much sugar is needed "
            "for 30 cookies? If sugar costs $2.40 per cup, what is the total cost of sugar "
            "for 30 cookies?",
            "1.875 cups; $4.50",
        )


if __name__ == "__main__":
    unittest.main()
