from __future__ import annotations

import unittest

from src.fireworks_client import build_retry_prompt


class RetryPromptTest(unittest.TestCase):
    def test_retry_prompt_is_strict_and_includes_failure_feedback(self) -> None:
        prompt = build_retry_prompt(
            {"prompt": "Write a function."},
            "code generation",
            "SyntaxError: invalid syntax",
        )
        self.assertIn("Reply with ONLY Python code followed by # SELF_CHECK: assert statements. Nothing else.", prompt)
        self.assertIn("SyntaxError: invalid syntax", prompt)
        self.assertIn("Write a function.", prompt)


if __name__ == "__main__":
    unittest.main()
