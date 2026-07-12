from __future__ import annotations

import unittest

from src.code_validation import (
    expected_function_names,
    extract_code,
    parse_code_answer,
    validate_code_answer,
)


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

    def test_signature_match_from_prompt_passes(self) -> None:
        result = validate_code_answer(
            {"prompt": "Write a Python function `is_palindrome(s)` that returns True."},
            "def is_palindrome(s):\n    return s == s[::-1]",
        )
        self.assertTrue(result.ok)

    def test_renamed_function_from_prompt_is_rejected(self) -> None:
        result = validate_code_answer(
            {"prompt": "Write a Python function that returns the larger of two numbers."},
            "def find_larger(a, b):\n    return a if a > b else b",
        )
        self.assertFalse(result.ok)
        self.assertIn("function signature mismatch", result.error or "")
        self.assertIn("expected larger", result.error or "")

    def test_signature_match_from_explicit_call_test_passes(self) -> None:
        result = validate_code_answer(
            {"tests": [{"call": "larger(2, 3)", "expected": 3}]},
            "def larger(a, b):\n    return a if a > b else b",
        )
        self.assertTrue(result.ok)

    def test_renamed_function_from_explicit_call_test_is_rejected_before_execution(self) -> None:
        result = validate_code_answer(
            {"tests": [{"call": "larger(2, 3)", "expected": 3}]},
            "def find_larger(a, b):\n    return a if a > b else b",
        )
        self.assertFalse(result.ok)
        self.assertIn("expected larger", result.error or "")
        self.assertIn("generated find_larger", result.error or "")

    def test_expected_function_names_from_stub_and_backticked_signature(self) -> None:
        self.assertEqual(
            expected_function_names({"prompt": "Find the bug: def add(a, b): return a - b"}),
            {"add"},
        )
        self.assertEqual(
            expected_function_names({"prompt": "Write a Python function `is_palindrome(s)`."}),
            {"is_palindrome"},
        )

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

    def test_valid_self_check_passes_without_explicit_tests(self) -> None:
        result = validate_code_answer(
            {},
            "def add(a, b):\n    return a + b\n\n# SELF_CHECK:\nassert add(2, 3) == 5\nassert add(-1, 1) == 0",
        )
        self.assertTrue(result.ok)

    def test_failing_self_check_reports_assertion_failure(self) -> None:
        result = validate_code_answer(
            {},
            "def add(a, b):\n    return a - b\n\n# SELF_CHECK:\nassert add(2, 3) == 5",
        )
        self.assertFalse(result.ok)
        self.assertIn("AssertionError", result.error or "")

    def test_explicit_tests_take_priority_over_self_checks(self) -> None:
        result = validate_code_answer(
            {"tests": [{"call": "add(2, 3)", "expected": 5}]},
            "def add(a, b):\n    return a - b\n\n# SELF_CHECK:\nassert add(2, 3) == -1",
        )
        self.assertFalse(result.ok)
        self.assertIn("expected 5, got -1", result.error or "")

    def test_missing_self_check_block_falls_back_to_compile_only(self) -> None:
        result = validate_code_answer({}, "def add(a, b):\n    return a - b")
        self.assertTrue(result.ok)

    def test_local_acceptance_requires_task_owned_semantic_tests(self) -> None:
        result = validate_code_answer(
            {"prompt": "Write a Python function `add(a, b)`."},
            "def add(a, b):\n    return a - b\n\n# SELF_CHECK:\nassert add(2, 3) == -1",
            require_task_tests=True,
        )
        self.assertFalse(result.ok)
        self.assertIn("task-owned semantic tests", result.error or "")

    def test_malformed_self_check_block_falls_back_to_compile_only(self) -> None:
        parsed = parse_code_answer("def add(a, b):\n    return a - b\n\n# SELF_CHECK:\nadd(2, 3) == 5")
        self.assertEqual(parsed.self_checks, ())
        self.assertTrue(validate_code_answer({}, "def add(a, b):\n    return a - b\n\n# SELF_CHECK:\nadd(2, 3) == 5").ok)


if __name__ == "__main__":
    unittest.main()
