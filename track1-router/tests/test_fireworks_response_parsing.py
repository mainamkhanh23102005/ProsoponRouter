from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import Mock, patch


class FakeResponse:
    def __init__(self, message: dict[str, object]) -> None:
        self.message = message

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "choices": [{"message": self.message}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }


class FireworksResponseParsingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_env = {
            "ALLOWED_MODELS": os.environ.get("ALLOWED_MODELS"),
            "CHEAPEST_MODEL": os.environ.get("CHEAPEST_MODEL"),
            "DRY_RUN": os.environ.get("DRY_RUN"),
            "FIREWORKS_API_KEY": os.environ.get("FIREWORKS_API_KEY"),
        }
        os.environ["ALLOWED_MODELS"] = "minimax-m3"
        os.environ.pop("CHEAPEST_MODEL", None)
        os.environ["DRY_RUN"] = "0"
        os.environ["FIREWORKS_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        for key, value in self.previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.reload_client()

    def reload_client(self):
        import src.config as config
        import src.fireworks_client as fireworks_client

        importlib.reload(config)
        return importlib.reload(fireworks_client)

    def complete_with_messages(self, messages: list[dict[str, object]]):
        fireworks_client = self.reload_client()
        client = fireworks_client.FireworksClient()
        fake_post = Mock(side_effect=[FakeResponse(message) for message in messages])
        fake_requests = SimpleNamespace(post=fake_post)
        with patch.dict(sys.modules, {"requests": fake_requests}):
            stderr = StringIO()
            with redirect_stderr(stderr):
                result = client.complete({"prompt": "What is the capital of Australia?"}, "factual knowledge")
        return result, stderr.getvalue(), fake_post.call_count

    def complete_with_message(self, message: dict[str, object]):
        return self.complete_with_messages([message, message])

    def test_missing_content_key_is_normal_failure_not_keyerror(self) -> None:
        result, stderr, call_count = self.complete_with_message({"role": "assistant"})

        self.assertEqual(result.answer, "")
        self.assertIn("empty content, message keys were: ['role']", result.error or "")
        self.assertIn("empty content, message keys were: ['role']", stderr)
        self.assertEqual(call_count, 2)

    def test_none_content_is_normal_failure_not_attribute_error(self) -> None:
        result, stderr, call_count = self.complete_with_message({"content": None})

        self.assertEqual(result.answer, "")
        self.assertIn("empty content, message keys were: ['content']", result.error or "")
        self.assertIn("empty content, message keys were: ['content']", stderr)
        self.assertEqual(call_count, 2)

    def test_reasoning_content_is_used_when_content_missing(self) -> None:
        result, stderr, call_count = self.complete_with_message({"role": "assistant", "reasoning_content": "Canberra"})

        self.assertEqual(result.answer, "Canberra")
        self.assertIsNone(result.error)
        self.assertEqual(stderr, "")
        self.assertEqual(call_count, 1)

    def test_empty_content_retries_once_and_returns_second_content(self) -> None:
        result, stderr, call_count = self.complete_with_messages(
            [{"role": "assistant"}, {"content": "Canberra"}]
        )

        self.assertEqual(result.answer, "Canberra")
        self.assertIsNone(result.error)
        self.assertIn("empty content, message keys were: ['role']", stderr)
        self.assertEqual(result.total_tokens, 14)
        self.assertEqual(call_count, 2)

    def test_empty_content_twice_returns_empty_answer_after_two_attempts(self) -> None:
        result, stderr, call_count = self.complete_with_messages(
            [{"role": "assistant"}, {"content": None}]
        )

        self.assertEqual(result.answer, "")
        self.assertIn("empty content, message keys were: ['content']", result.error or "")
        self.assertEqual(stderr.count("empty content, message keys were:"), 2)
        self.assertEqual(result.total_tokens, 14)
        self.assertEqual(call_count, 2)


if __name__ == "__main__":
    unittest.main()
