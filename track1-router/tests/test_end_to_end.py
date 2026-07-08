from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path


class EndToEndTest(unittest.TestCase):
    def test_main_writes_one_answer_per_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "tasks.json"
            output_path = root / "results.json"
            tasks = [
                {"id": "m", "category": "math", "question": "What is 2 + 3?"},
                {"id": "s", "category": "sentiment", "text": "Excellent and useful."},
                {"id": "n", "category": "ner", "text": "Email dan@example.com on 2026-07-11."},
                {"id": "u", "category": "unknown", "text": ""},
            ]
            input_path.write_text(json.dumps(tasks), encoding="utf-8")

            previous_env = {
                "TASK_INPUT_PATH": os.environ.get("TASK_INPUT_PATH"),
                "RESULTS_OUTPUT_PATH": os.environ.get("RESULTS_OUTPUT_PATH"),
                "DRY_RUN": os.environ.get("DRY_RUN"),
            }
            os.environ["TASK_INPUT_PATH"] = str(input_path)
            os.environ["RESULTS_OUTPUT_PATH"] = str(output_path)
            os.environ["DRY_RUN"] = "1"
            try:
                import src.config as config
                import src.main as main

                importlib.reload(config)
                importlib.reload(main)
                self.assertEqual(main.main(), 0)
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

            results = json.loads(output_path.read_text(encoding="utf-8"))
            answer_by_id = {item["id"]: item["answer"] for item in results}
            self.assertEqual(set(answer_by_id), {"m", "s", "n", "u"})
            self.assertEqual(answer_by_id["m"], "5")
            self.assertEqual(answer_by_id["s"], "positive")
            self.assertIsInstance(answer_by_id["n"], list)


if __name__ == "__main__":
    unittest.main()
