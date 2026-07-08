from __future__ import annotations

import unittest

from src import config
from src.fireworks_client import build_prompt


class PromptTemplateTest(unittest.TestCase):
    def test_prompts_are_compressed_and_format_specific(self) -> None:
        task = {"prompt": "Example task text."}
        expectations = {
            "math": "Only final value.",
            "ner": "LABEL: text",
            "sentiment": "one-sentence justification",
            "summarization": "Output only summary.",
            "factual knowledge": "no explanation",
            "code debugging": "corrected Python code only",
            "code generation": "Python code only",
            "logical reasoning": "yes/no/unknown only",
        }
        for category, expected in expectations.items():
            with self.subTest(category=category):
                prompt = build_prompt(task, category)
                self.assertIn(expected, prompt)
                self.assertLessEqual(len(prompt.split()), 24)

    def test_policy_token_budgets_are_category_specific(self) -> None:
        self.assertLess(config.POLICY["factual knowledge"].max_tokens, config.POLICY["summarization"].max_tokens)
        self.assertLess(config.POLICY["summarization"].max_tokens, config.POLICY["code generation"].max_tokens)
        self.assertTrue(config.POLICY["code debugging"].retry_on_invalid)
        self.assertTrue(config.POLICY["code generation"].retry_on_invalid)
        self.assertFalse(config.POLICY["sentiment"].retry_on_invalid)


if __name__ == "__main__":
    unittest.main()

