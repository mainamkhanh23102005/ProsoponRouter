from __future__ import annotations

import unittest

from scripts.evaluate_external_zero_token import (
    Metric,
    gsm8k_final_answer,
    parse_ner_answer,
    sentiment_label,
)


class ExternalBenchmarkTest(unittest.TestCase):
    def test_extracts_gsm8k_final_answer(self) -> None:
        self.assertEqual(gsm8k_final_answer("work\n#### 1,234.50"), "1234.5")

    def test_extracts_sentiment_label(self) -> None:
        self.assertEqual(sentiment_label("positive: strong approval"), "1")
        self.assertEqual(sentiment_label("negative: disappointment"), "0")
        self.assertIsNone(sentiment_label("neutral: factual"))

    def test_metric_separates_coverage_from_accepted_precision(self) -> None:
        metric = Metric()
        metric.record(None, "5")
        metric.record("5", "5")
        metric.record("4", "5")
        self.assertEqual(metric.total, 3)
        self.assertEqual(metric.accepted, 2)
        self.assertEqual(metric.correct, 1)
        self.assertAlmostEqual(metric.coverage, 2 / 3)
        self.assertAlmostEqual(metric.accepted_precision, 0.5)

    def test_parses_router_ner_format_as_ordered_entities(self) -> None:
        self.assertEqual(
            parse_ner_answer("PERSON: Maria Chen; ORG: OpenAI; LOCATION: Geneva"),
            (("PERSON", "Maria Chen"), ("ORG", "OpenAI"), ("LOCATION", "Geneva")),
        )
        self.assertEqual(parse_ner_answer("NONE"), ())


if __name__ == "__main__":
    unittest.main()
