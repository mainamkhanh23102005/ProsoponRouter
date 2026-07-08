from __future__ import annotations

import unittest

from src import cascade
from src.fireworks_client import FireworksResult


class FakeClient:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.retry_feedback: list[str | None] = []

    def complete(self, task, category, retry_feedback=None):
        self.retry_feedback.append(retry_feedback)
        return FireworksResult(answer=self.answers.pop(0), prompt_tokens=0, completion_tokens=0)


class CodeRetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_client = cascade.CLIENT

    def tearDown(self) -> None:
        cascade.CLIENT = self.previous_client

    def test_correct_code_answer_is_accepted(self) -> None:
        fake = FakeClient(["def add(a, b):\n    return a + b"])
        cascade.CLIENT = fake
        answer, meta = cascade.route_task(
            {"tests": [{"call": "add(2, 3)", "expected": 5}]},
            "code generation",
        )
        self.assertEqual(answer, "def add(a, b):\n    return a + b")
        self.assertEqual(meta["path"], "fireworks")
        self.assertEqual(meta["attempts"], 1)

    def test_syntax_error_triggers_retry_with_failure_info(self) -> None:
        fake = FakeClient(["def bad(:\n    pass", "def add(a, b):\n    return a + b"])
        cascade.CLIENT = fake
        answer, meta = cascade.route_task(
            {"tests": [{"call": "add(2, 3)", "expected": 5}]},
            "code generation",
        )
        self.assertEqual(answer, "def add(a, b):\n    return a + b")
        self.assertEqual(meta["path"], "fireworks_retry")
        self.assertIn("SyntaxError", fake.retry_feedback[1] or "")

    def test_failed_test_case_triggers_retry_with_failure_info(self) -> None:
        fake = FakeClient(["def add(a, b):\n    return a - b", "def add(a, b):\n    return a + b"])
        cascade.CLIENT = fake
        answer, meta = cascade.route_task(
            {"tests": [{"call": "add(2, 3)", "expected": 5}]},
            "code generation",
        )
        self.assertEqual(answer, "def add(a, b):\n    return a + b")
        self.assertEqual(meta["path"], "fireworks_retry")
        self.assertIn("expected 5, got -1", fake.retry_feedback[1] or "")


if __name__ == "__main__":
    unittest.main()

