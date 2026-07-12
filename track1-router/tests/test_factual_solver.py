from __future__ import annotations

import unittest

from src.solvers import factual_solver


class FactualSolverTest(unittest.TestCase):
    def test_common_capital_still_escalates_without_broad_knowledge_base(self) -> None:
        answer, confidence = factual_solver.solve({"prompt": "What is the capital of Vietnam?"})
        self.assertIsNone(answer)
        self.assertEqual(confidence, 0.0)

    def test_untrusted_fact_declines(self) -> None:
        answer, confidence = factual_solver.solve({"prompt": "Who won the 2026 world championship?"})
        self.assertIsNone(answer)
        self.assertEqual(confidence, 0.0)
