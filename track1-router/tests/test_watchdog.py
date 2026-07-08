from __future__ import annotations

import importlib
import os
import time
import unittest


class WatchdogTest(unittest.TestCase):
    def test_watchdog_active_by_default_near_runtime_budget(self) -> None:
        previous_limit = os.environ.get("LATENCY_LIMIT_SECONDS")
        previous_reserve = os.environ.get("LATENCY_RESERVE_SECONDS")
        os.environ.pop("LATENCY_LIMIT_SECONDS", None)
        os.environ.pop("LATENCY_RESERVE_SECONDS", None)
        try:
            import src.config as config
            import src.main as main

            importlib.reload(config)
            importlib.reload(main)
            self.assertEqual(config.LATENCY_LIMIT_SECONDS, 540)
            self.assertTrue(main.watchdog_expired(time.monotonic() - 540))
            self.assertFalse(main.watchdog_expired(time.monotonic() - 100))
        finally:
            if previous_limit is None:
                os.environ.pop("LATENCY_LIMIT_SECONDS", None)
            else:
                os.environ["LATENCY_LIMIT_SECONDS"] = previous_limit
            if previous_reserve is None:
                os.environ.pop("LATENCY_RESERVE_SECONDS", None)
            else:
                os.environ["LATENCY_RESERVE_SECONDS"] = previous_reserve

    def test_watchdog_expires_inside_reserve_window(self) -> None:
        previous_limit = os.environ.get("LATENCY_LIMIT_SECONDS")
        previous_reserve = os.environ.get("LATENCY_RESERVE_SECONDS")
        os.environ["LATENCY_LIMIT_SECONDS"] = "10"
        os.environ["LATENCY_RESERVE_SECONDS"] = "2"
        try:
            import src.config as config
            import src.main as main

            importlib.reload(config)
            importlib.reload(main)
            self.assertTrue(main.watchdog_expired(time.monotonic() - 8.5))
            self.assertFalse(main.watchdog_expired(time.monotonic() - 5))
        finally:
            if previous_limit is None:
                os.environ.pop("LATENCY_LIMIT_SECONDS", None)
            else:
                os.environ["LATENCY_LIMIT_SECONDS"] = previous_limit
            if previous_reserve is None:
                os.environ.pop("LATENCY_RESERVE_SECONDS", None)
            else:
                os.environ["LATENCY_RESERVE_SECONDS"] = previous_reserve


if __name__ == "__main__":
    unittest.main()
