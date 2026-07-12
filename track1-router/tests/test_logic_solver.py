from __future__ import annotations

import unittest

from src.solvers import logic_solver


class LogicSolverTest(unittest.TestCase):
    def test_transitive_all_syllogism(self) -> None:
        answer, confidence = logic_solver.solve(
            {"prompt": "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?"}
        )
        self.assertEqual(answer, "Yes")
        self.assertGreaterEqual(confidence, 0.95)

    def test_finite_seating_problem_with_unique_answer(self) -> None:
        answer, confidence = logic_solver.solve(
            {
                "prompt": (
                    "Ava, Ben, and Cara sit in a row of 3 seats numbered 1 to 3. "
                    "Ava sits immediately before Ben. Cara does not sit in seat 1. "
                    "Who sits in seat 1?"
                )
            }
        )
        self.assertEqual(answer, "Ava")
        self.assertGreaterEqual(confidence, 0.95)

    def test_fully_parsed_underdetermined_seating_problem_reports_ambiguity(self) -> None:
        answer, confidence = logic_solver.solve(
            {
                "prompt": (
                    "Five friends - Ivy, Jude, Kai, Lena, and Moss - sit in a row of 5 seats numbered 1 to 5. "
                    "Ivy does not sit in seat 1. Moss does not sit in seat 5. "
                    "The person who drinks tea sits immediately after Ivy. "
                    "Kai does not drink water, and Kai does not sit immediately before Lena. "
                    "Who sits in seat 1?"
                )
            }
        )
        self.assertEqual(answer, "Cannot be determined")
        self.assertGreaterEqual(confidence, 0.95)

    def test_unrecognized_seating_constraint_declines(self) -> None:
        answer, confidence = logic_solver.solve(
            {
                "prompt": (
                    "Ava, Ben, and Cara sit in a row of 3 seats numbered 1 to 3. "
                    "Ava sits somewhere to the left of Ben. Who sits in seat 1?"
                )
            }
        )
        self.assertIsNone(answer)
        self.assertEqual(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
