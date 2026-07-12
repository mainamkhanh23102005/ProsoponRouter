from __future__ import annotations

import unittest

from src import config
from src.fireworks_client import build_prompt


class PromptTemplateTest(unittest.TestCase):
    def test_prompts_are_compressed_and_format_specific(self) -> None:
        task = {"prompt": "Example task text."}
        expectations = {
            "math": "Last line: Answer:",
            "ner": "ORGANIZATION: name",
            "sentiment": "one-sentence justification",
            "summarization": "Output only the summary",
            "factual knowledge": "Answer every part",
            "code debugging": "# SELF_CHECK:",
            "code generation": "# SELF_CHECK:",
            "logical reasoning": "Last line: Answer:",
        }
        for category, expected in expectations.items():
            with self.subTest(category=category):
                prompt = build_prompt(task, category)
                self.assertIn(expected, prompt)
                if category not in {"factual knowledge", "ner"}:
                    self.assertLessEqual(len(prompt.split()), 24)

    def test_logical_reasoning_prompt_does_not_force_yes_no_unknown(self) -> None:
        prompt = build_prompt({"prompt": "Who sits in seat 1?"}, "logical reasoning")
        self.assertNotIn("yes/no/unknown", prompt)
        self.assertIn("Last line: Answer:", prompt)

    def test_math_prompt_preserves_multi_part_answers(self) -> None:
        prompt = build_prompt({"prompt": "Find the amount and total cost."}, "math")
        self.assertIn("include every requested result", prompt)

    def test_summarization_prompt_does_not_hardcode_sentence_count(self) -> None:
        prompt = build_prompt({"prompt": "Summarize in one sentence: Example text."}, "summarization")
        self.assertNotIn("<=2 sentences", prompt)
        self.assertNotIn("two sentences", prompt.lower())
        self.assertIn("length/format instruction", prompt)

    def test_factual_prompt_requests_compact_non_markdown_answer(self) -> None:
        prompt = build_prompt({"prompt": "What is the difference between TCP and UDP?"}, "factual knowledge")
        self.assertIn("Answer every part", prompt)
        self.assertIn("key properties", prompt)
        self.assertNotIn("**", prompt)
        self.assertNotIn("- ", prompt)

    def test_ner_prompt_does_not_invite_literal_label_placeholder(self) -> None:
        prompt = build_prompt({"prompt": "Extract all named entities: Alice visited Paris."}, "ner")
        self.assertNotIn("LABEL:", prompt)
        self.assertIn("ORGANIZATION: name", prompt)
        self.assertIn("compound organization names", prompt)
        self.assertIn("ETH Zurich", prompt)
        self.assertIn("enumerate every named organization, person, location, and date", prompt)
        self.assertIn("omit none", prompt)

    def test_policy_token_budgets_are_category_specific(self) -> None:
        self.assertGreaterEqual(config.POLICY["factual knowledge"].max_tokens, 120)
        self.assertLess(config.POLICY["summarization"].max_tokens, config.POLICY["code generation"].max_tokens)
        self.assertTrue(config.POLICY["code debugging"].retry_on_invalid)
        self.assertTrue(config.POLICY["code generation"].retry_on_invalid)
        self.assertFalse(config.POLICY["sentiment"].retry_on_invalid)


if __name__ == "__main__":
    unittest.main()
