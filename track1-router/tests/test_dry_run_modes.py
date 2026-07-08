from __future__ import annotations

import importlib
import os
import unittest


class DryRunModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = {
            "DRY_RUN": os.environ.get("DRY_RUN"),
            "DRY_RUN_MODE": os.environ.get("DRY_RUN_MODE"),
        }
        os.environ["DRY_RUN"] = "1"

    def tearDown(self) -> None:
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.reload_modules()

    def reload_modules(self):
        import src.config as config
        import src.fireworks_client as fireworks_client
        import src.cascade as cascade

        importlib.reload(config)
        importlib.reload(fireworks_client)
        return importlib.reload(cascade)

    def route_with_mode(self, mode: str):
        os.environ["DRY_RUN_MODE"] = mode
        cascade = self.reload_modules()
        return cascade.route_task({"prompt": "Summarize this: AMD builds AI software."}, "summarization")

    def test_success_mode_passes_through(self) -> None:
        answer, meta = self.route_with_mode("success")
        self.assertEqual(answer, "AMD hackathon teams build token-efficient AI apps.")
        self.assertEqual(meta["path"], "fireworks")

    def test_invalid_mode_falls_back(self) -> None:
        answer, meta = self.route_with_mode("invalid")
        self.assertEqual(answer, "")
        self.assertEqual(meta["path"], "fallback")
        self.assertEqual(meta["attempts"], 1)
        self.assertFalse(meta["retried"])

    def test_error_mode_falls_back(self) -> None:
        answer, meta = self.route_with_mode("error")
        self.assertEqual(answer, "")
        self.assertEqual(meta["path"], "fallback")
        self.assertEqual(meta["error"], "dry-run simulated API error")
        self.assertEqual(meta["attempts"], 1)

    def test_invalid_mode_retries_for_code_categories(self) -> None:
        os.environ["DRY_RUN_MODE"] = "invalid"
        cascade = self.reload_modules()
        answer, meta = cascade.route_task({"prompt": "Write a Python function add(a, b)."}, "code generation")
        self.assertEqual(answer, "")
        self.assertEqual(meta["path"], "fallback")
        self.assertTrue(meta["retried"])
        self.assertEqual(meta["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
