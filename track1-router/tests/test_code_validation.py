from __future__ import annotations

import unittest

from src.code_validation import extract_code, validate_code_answer


class CodeValidationTest(unittest.TestCase):
    def test_extract_code_strips_markdown_fence(self) -> None:
        answer = "```python\ndef add(a, b):\n    return a + b\n```"
        self.assertEqual(extract_code(answer), "def add(a, b):\n    return a + b")

    def test_valid_code_passes_call_tests(self) -> None:
        result = validate_code_answer(
            {"tests": [{"call": "add(2, 3)", "expected": 5}]},
            "def add(a, b):\n    return a + b",
        )
        self.assertTrue(result.ok)

    def test_syntax_error_fails_compile_check(self) -> None:
        result = validate_code_answer({}, "def bad(:\n    pass")
        self.assertFalse(result.ok)
        self.assertIn("SyntaxError", result.error or "")

    def test_failing_call_test_reports_actual_failure(self) -> None:
        result = validate_code_answer(
            {"tests": [{"call": "add(2, 3)", "expected": 5}]},
            "def add(a, b):\n    return a - b",
        )
        self.assertFalse(result.ok)
        self.assertIn("expected 5, got -1", result.error or "")


if __name__ == "__main__":
    unittest.main()

