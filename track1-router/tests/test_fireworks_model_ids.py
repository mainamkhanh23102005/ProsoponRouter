from __future__ import annotations

import unittest

from src import config
from src.fireworks_client import build_payload, normalize_model_id


class FireworksModelIdTest(unittest.TestCase):
    def test_bare_model_name_is_normalized_to_full_fireworks_path(self) -> None:
        self.assertEqual(
            normalize_model_id("minimax-m3"),
            "accounts/fireworks/models/minimax-m3",
        )

    def test_full_accounts_path_is_left_unchanged(self) -> None:
        full = "accounts/fireworks/models/minimax-m3"
        self.assertEqual(normalize_model_id(full), full)

    def test_payload_uses_normalized_policy_model_without_mutating_policy(self) -> None:
        policy = config.CategoryPolicy(False, "minimax-m3", 24)
        payload = build_payload(policy, "Answer only.\nQ")
        self.assertEqual(payload["model"], "accounts/fireworks/models/minimax-m3")
        self.assertEqual(policy.model, "minimax-m3")

    def test_payload_normalization_applies_to_every_policy_category(self) -> None:
        for category in config.POLICY:
            with self.subTest(category=category):
                policy = config.CategoryPolicy(
                    config.POLICY[category].free_enabled,
                    "minimax-m3",
                    config.POLICY[category].max_tokens,
                    config.POLICY[category].min_confidence,
                    config.POLICY[category].retry_on_invalid,
                    config.POLICY[category].stop,
                )
                payload = build_payload(policy, "Prompt")
                self.assertEqual(payload["model"], "accounts/fireworks/models/minimax-m3")

    def test_payload_disables_minimax_thinking(self) -> None:
        for category in config.POLICY:
            with self.subTest(category=category):
                payload = build_payload(config.POLICY[category], "Prompt")
                if category == "sentiment":
                    self.assertEqual(payload["reasoning_effort"], "none")
                    self.assertNotIn("thinking", payload)
                else:
                    self.assertEqual(payload["thinking"], {"type": "disabled"})
                    self.assertNotIn("reasoning_effort", payload)


if __name__ == "__main__":
    unittest.main()
