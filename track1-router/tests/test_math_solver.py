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


if __name__ == "__main__":
    unittest.main()
