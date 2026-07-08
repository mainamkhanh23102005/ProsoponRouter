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
            "summarization": "Output only the summary",
            "factual knowledge": "no explanation",
            "code debugging": "# SELF_CHECK:",
            "code generation": "# SELF_CHECK:",
            "logical reasoning": "final answer",
        }
        for category, expected in expectations.items():
            with self.subTest(category=category):
                prompt = build_prompt(task, category)
                self.assertIn(expected, prompt)
                self.assertLessEqual(len(prompt.split()), 24)

    def test_logical_reasoning_prompt_does_not_force_yes_no_unknown(self) -> None:
        prompt = build_prompt({"prompt": "Who sits in seat 1?"}, "logical reasoning")
        self.assertNotIn("yes/no/unknown", prompt)
        self.assertIn("final answer", prompt)

    def test_summarization_prompt_does_not_hardcode_sentence_count(self) -> None:
        prompt = build_prompt({"prompt": "Summarize in one sentence: Example text."}, "summarization")
        self.assertNotIn("<=2 sentences", prompt)
        self.assertNotIn("two sentences", prompt.lower())
        self.assertIn("length/format instruction", prompt)

    def test_policy_token_budgets_are_category_specific(self) -> None:
        self.assertLess(config.POLICY["factual knowledge"].max_tokens, config.POLICY["summarization"].max_tokens)
        self.assertLess(config.POLICY["summarization"].max_tokens, config.POLICY["code generation"].max_tokens)
        self.assertTrue(config.POLICY["code debugging"].retry_on_invalid)
        self.assertTrue(config.POLICY["code generation"].retry_on_invalid)
        self.assertFalse(config.POLICY["sentiment"].retry_on_invalid)


if __name__ == "__main__":
    unittest.main()
