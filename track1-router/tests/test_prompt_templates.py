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
            "factual knowledge": "one compact sentence",
            "code debugging": "# SELF_CHECK:",
            "code generation": "# SELF_CHECK:",
            "logical reasoning": "final answer",
        }
        for category, expected in expectations.items():
            with self.subTest(category=category):
                prompt = build_prompt(task, category)
                self.assertIn(expected, prompt)
                if category != "factual knowledge":
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

    def test_factual_prompt_requests_compact_non_markdown_answer(self) -> None:
        prompt = build_prompt({"prompt": "What is the difference between TCP and UDP?"}, "factual knowledge")
        self.assertIn(
            "Answer all parts in one compact sentence using semicolons; "
            "include only the requested facts, no background, no markdown.",
            prompt,
        )
        self.assertNotIn("**", prompt)
        self.assertNotIn("- ", prompt)

    def test_policy_token_budgets_are_category_specific(self) -> None:
        self.assertGreaterEqual(config.POLICY["factual knowledge"].max_tokens, 160)
        self.assertLess(config.POLICY["summarization"].max_tokens, config.POLICY["code generation"].max_tokens)
        self.assertTrue(config.POLICY["code debugging"].retry_on_invalid)
        self.assertTrue(config.POLICY["code generation"].retry_on_invalid)
        self.assertFalse(config.POLICY["sentiment"].retry_on_invalid)


if __name__ == "__main__":
    unittest.main()
