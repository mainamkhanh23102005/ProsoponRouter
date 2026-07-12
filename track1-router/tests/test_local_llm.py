from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src import cascade
from src.fireworks_client import FireworksResult


class LocalLlmCascadeTest(unittest.TestCase):
    def test_unverifiable_local_summary_does_not_short_circuit_fireworks(self) -> None:
        fake_client = Mock()
        fake_client.complete.return_value = FireworksResult(answer="Remote summary")
        with (
            patch.object(cascade.local_llm, "can_attempt", return_value=True),
            patch.object(cascade.local_llm, "complete", return_value=FireworksResult(answer="Short summary")),
            patch.object(cascade, "CLIENT", fake_client),
        ):
            answer, meta = cascade.route_task({"prompt": "Summarize this text."}, "summarization")

        self.assertEqual(answer, "Remote summary")
        self.assertEqual(meta["path"], "fireworks")
        fake_client.complete.assert_called_once()

    def test_local_code_with_task_owned_tests_short_circuits_fireworks(self) -> None:
        fake_client = Mock()
        with (
            patch.object(cascade.local_llm, "can_attempt", return_value=True),
            patch.object(
                cascade.local_llm,
                "complete",
                return_value=FireworksResult(answer="def add(a, b):\n    return a + b"),
            ),
            patch.object(cascade, "CLIENT", fake_client),
        ):
            answer, meta = cascade.route_task(
                {"prompt": "Write add.", "tests": [{"call": "add(2, 3)", "expected": 5}]},
                "code generation",
            )

        self.assertEqual(answer, "def add(a, b):\n    return a + b")
        self.assertEqual(meta["path"], "local_llm")
        fake_client.complete.assert_not_called()

    def test_invalid_local_answer_falls_through_to_fireworks(self) -> None:
        fake_client = Mock()
        fake_client.complete.return_value = FireworksResult(answer="positive: remote answer")
        with (
            patch.object(cascade.local_llm, "can_attempt", return_value=True),
            patch.object(cascade.local_llm, "complete", return_value=FireworksResult(answer="maybe")),
            patch.object(cascade, "CLIENT", fake_client),
        ):
            answer, meta = cascade.route_task({"prompt": "Sentiment?"}, "sentiment")

        self.assertEqual(answer, "positive: remote answer")
        self.assertEqual(meta["path"], "fireworks")
        fake_client.complete.assert_called_once()

    def test_renamed_local_code_answer_falls_through_to_fireworks(self) -> None:
        fake_client = Mock()
        fake_client.complete.return_value = FireworksResult(
            answer="def larger(a, b):\n    return a if a > b else b"
        )
        with (
            patch.object(cascade.local_llm, "can_attempt", return_value=True),
            patch.object(
                cascade.local_llm,
                "complete",
                return_value=FireworksResult(
                    answer="def find_larger(a, b):\n    return a if a > b else b"
                ),
            ),
            patch.object(cascade, "CLIENT", fake_client),
        ):
            answer, meta = cascade.route_task(
                {"prompt": "Write a Python function that returns the larger of two numbers."},
                "code generation",
            )

        self.assertEqual(answer, "def larger(a, b):\n    return a if a > b else b")
        self.assertEqual(meta["path"], "fireworks")
        fake_client.complete.assert_called_once()

    def test_sentiment_model_answer_accepts_dash_separator(self) -> None:
        ok, answer, error = cascade.validate_model_answer(
            {"prompt": "Classify sentiment."},
            "sentiment",
            "Positive - clear approval",
        )

        self.assertTrue(ok)
        self.assertEqual(answer, "positive: clear approval")
        self.assertIsNone(error)

    def test_meta_answer_is_rejected(self) -> None:
        ok, answer, error = cascade.validate_model_answer(
            {"prompt": "Summarize this text."},
            "summarization",
            "Thinking Process:\n1. Analyze the request.\nFinal summary.",
        )

        self.assertFalse(ok)
        self.assertIn("analysis instead of final answer", error or "")

    def test_numbered_analysis_answer_is_rejected(self) -> None:
        ok, answer, error = cascade.validate_model_answer(
            {"prompt": "Summarize this text."},
            "summarization",
            "1. **Analyze the Request:** The user wants a summary.\n\nFinal summary.",
        )

        self.assertFalse(ok)
        self.assertIn("analysis instead of final answer", error or "")

    def test_logic_reasoning_trace_for_bloops_case_is_rejected(self) -> None:
        prompt = "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?"
        local_answer = (
            "The user is asking a question about logical deduction. This is a classic "
            "syllogism. Premise 1: all bloops are razzies. Premise 2: all razzies are "
            "lazzies. Therefore"
        )

        ok, answer, error = cascade.validate_model_answer(
            {"prompt": prompt},
            "logical reasoning",
            local_answer,
        )

        self.assertFalse(ok)
        self.assertIn("analysis instead of final answer", error or "")

    def test_visible_math_work_is_reduced_to_final_answer(self) -> None:
        ok, answer, error = cascade.validate_model_answer(
            {"prompt": "A word problem."},
            "math",
            "480 - 120 = 360\n360 + 240 = 600\n600 - 50 = 550\nAnswer: 550",
        )
        self.assertTrue(ok)
        self.assertEqual(answer, "550")
        self.assertIsNone(error)

    def test_visible_logic_work_is_reduced_to_final_answer(self) -> None:
        ok, answer, error = cascade.validate_model_answer(
            {"prompt": "A logic problem."},
            "logical reasoning",
            "A before B. C cannot be first.\nAnswer: A",
        )
        self.assertTrue(ok)
        self.assertEqual(answer, "A")
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
