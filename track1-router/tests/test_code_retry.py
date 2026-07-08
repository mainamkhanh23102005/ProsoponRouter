from __future__ import annotations

import unittest

from src import cascade
from src.fireworks_client import FireworksResult


class FakeClient:
    def __init__(self, answers: list[str], errors: list[str | None] | None = None) -> None:
        self.answers = answers
        self.errors = errors or [None] * len(answers)
        self.retry_feedback: list[str | None] = []

    def complete(self, task, category, retry_feedback=None):
        self.retry_feedback.append(retry_feedback)
        return FireworksResult(
            answer=self.answers.pop(0),
            prompt_tokens=0,
            completion_tokens=0,
            error=self.errors.pop(0),
        )


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

    def test_self_check_failure_triggers_retry_with_assertion_info(self) -> None:
        fake = FakeClient(
            [
                "def add(a, b):\n    return a - b\n\n# SELF_CHECK:\nassert add(2, 3) == 5",
                "def add(a, b):\n    return a + b\n\n# SELF_CHECK:\nassert add(2, 3) == 5",
            ]
        )
        cascade.CLIENT = fake
        answer, meta = cascade.route_task(
            {"prompt": "Write a Python function add(a, b)."},
            "code generation",
        )
        self.assertEqual(answer, "def add(a, b):\n    return a + b")
        self.assertEqual(meta["path"], "fireworks_retry")
        self.assertIn("AssertionError", fake.retry_feedback[1] or "")

    def test_empty_content_failure_does_not_trigger_code_semantic_retry(self) -> None:
        fake = FakeClient([""], ["empty content, message keys were: ['role']"])
        cascade.CLIENT = fake
        answer, meta = cascade.route_task(
            {"prompt": "Write a Python function add(a, b)."},
            "code generation",
        )
        self.assertEqual(answer, "")
        self.assertEqual(meta["path"], "fallback")
        self.assertEqual(meta["attempts"], 1)
        self.assertFalse(meta["retried"])
        self.assertEqual(fake.retry_feedback, [None])


if __name__ == "__main__":
    unittest.main()
