from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src import cascade
from src.fireworks_client import FireworksResult


class LocalLlmCascadeTest(unittest.TestCase):
    def test_valid_local_answer_short_circuits_fireworks(self) -> None:
        fake_client = Mock()
        with (
            patch.object(cascade.local_llm, "can_attempt", return_value=True),
            patch.object(cascade.local_llm, "complete", return_value=FireworksResult(answer="Short summary")),
            patch.object(cascade, "CLIENT", fake_client),
        ):
            answer, meta = cascade.route_task({"prompt": "Summarize this text."}, "summarization")

        self.assertEqual(answer, "Short summary")
        self.assertEqual(meta["path"], "local_llm")
        self.assertEqual(meta["tokens"], 0)
        fake_client.complete.assert_not_called()

    def test_invalid_local_answer_falls_through_to_fireworks(self) -> None:
        fake_client = Mock()
        fake_client.complete.return_value = FireworksResult(answer="positive: remote answer")
        with (
            patch.object(cascade.local_llm, "can_attempt", return_value=True),
            patch.object(cascade.local_llm, "complete", return_value=FireworksResult(answer="maybe")),
            patch.object(cascade, "CLIENT", fake_client),
        ):
            answer, meta = cascade.route_task({"prompt": "Sentiment?"}, "sentiment")

        self.assertEqual(answer, "positive: remote answer")
        self.assertEqual(meta["path"], "fireworks")
        fake_client.complete.assert_called_once()

    def test_sentiment_model_answer_accepts_dash_separator(self) -> None:
        ok, answer, error = cascade.validate_model_answer(
            {"prompt": "Classify sentiment."},
            "sentiment",
            "Positive - clear approval",
        )

        self.assertTrue(ok)
        self.assertEqual(answer, "positive: clear approval")
        self.assertIsNone(error)

    def test_meta_answer_is_rejected(self) -> None:
        ok, answer, error = cascade.validate_model_answer(
            {"prompt": "Summarize this text."},
            "summarization",
            "Thinking Process:\n1. Analyze the request.\nFinal summary.",
        )

        self.assertFalse(ok)
        self.assertIn("analysis instead of final answer", error or "")


if __name__ == "__main__":
    unittest.main()
