from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.fireworks_client import sanitize_error


class LoggingTest(unittest.TestCase):
    def test_startup_and_per_task_error_are_logged_without_key_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "tasks.json"
            output_path = root / "results.json"
            input_path.write_text(
                json.dumps([{"task_id": "summary-1", "prompt": "Summarize this: AMD builds AI."}]),
                encoding="utf-8",
            )

            previous = {
                "TASK_INPUT_PATH": os.environ.get("TASK_INPUT_PATH"),
                "RESULTS_OUTPUT_PATH": os.environ.get("RESULTS_OUTPUT_PATH"),
                "DRY_RUN": os.environ.get("DRY_RUN"),
                "DRY_RUN_MODE": os.environ.get("DRY_RUN_MODE"),
                "FIREWORKS_API_KEY": os.environ.get("FIREWORKS_API_KEY"),
            }
            os.environ["TASK_INPUT_PATH"] = str(input_path)
            os.environ["RESULTS_OUTPUT_PATH"] = str(output_path)
            os.environ["DRY_RUN"] = "1"
            os.environ["DRY_RUN_MODE"] = "error"
            os.environ["FIREWORKS_API_KEY"] = "secret-test-key"
            try:
                import src.config as config
                import src.main as main

                importlib.reload(config)
                importlib.reload(main)
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(main.main(), 0)
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

            log = stderr.getvalue()
            self.assertIn("startup api_key_env=FIREWORKS_API_KEY api_key_set=True", log)
            self.assertIn("fireworks_base_url=", log)
            self.assertIn("allowed_models=", log)
            self.assertIn("error=dry-run simulated API error", log)
            self.assertNotIn("secret-test-key", log)

    def test_fireworks_error_sanitizer_redacts_secret_values(self) -> None:
        previous_key = os.environ.get("FIREWORKS_API_KEY")
        os.environ["FIREWORKS_API_KEY"] = "fw-secret-123"
        try:
            error = (
                "request failed Authorization: Bearer fw-secret-123 "
                "api_key=fw-secret-123 raw=fw-secret-123"
            )
            sanitized = sanitize_error(error)
        finally:
            if previous_key is None:
                os.environ.pop("FIREWORKS_API_KEY", None)
            else:
                os.environ["FIREWORKS_API_KEY"] = previous_key

        self.assertNotIn("fw-secret-123", sanitized)
        self.assertIn("[REDACTED_API_KEY]", sanitized)


if __name__ == "__main__":
    unittest.main()

