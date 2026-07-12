from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src import cascade
from src.fireworks_client import FireworksResult


class LocalLlmCascadeTest(unittest.TestCase):
    def test_short_summary_uses_keyless_extractive_path(self) -> None:
        answer, meta = cascade.route_task(
            {"prompt": "Summarize this text: AMD announced a hackathon. Teams build AI apps."},
            "summarization",
        )
        self.assertEqual(answer, "AMD announced a hackathon.")
        self.assertEqual(meta["path"], "deterministic")

    def test_valid_local_summary_short_circuits_fireworks(self) -> None:
        fake_client = Mock()
        fake_client.complete.return_value = FireworksResult(answer="Remote summary")
        with (
            patch.object(cascade.local_llm, "can_attempt", return_value=True),
            patch.object(cascade.local_llm, "complete", return_value=FireworksResult(answer="Short summary")),
            patch.object(cascade, "CLIENT", fake_client),
        ):
            answer, meta = cascade.route_task({"prompt": "Summarize this text."}, "summarization")

        self.assertEqual(answer, "Short summary")
        self.assertEqual(meta["path"], "local_llm")
        fake_client.complete.assert_not_called()

    def test_valid_local_factual_answer_short_circuits_fireworks(self) -> None:
        fake_client = Mock()
        with (
            patch.object(cascade.local_llm, "can_attempt", return_value=True),
            patch.object(cascade.local_llm, "complete", return_value=FireworksResult(answer="Hanoi")),
            patch.object(cascade, "CLIENT", fake_client),
        ):
            answer, meta = cascade.route_task({"prompt": "What is the capital of Vietnam?"}, "factual knowledge")

        self.assertEqual(answer, "Hanoi")
        self.assertEqual(meta["path"], "local_llm")
        fake_client.complete.assert_not_called()

    def test_local_factual_answer_is_extracted_from_analysis_scaffold(self) -> None:
        ok, answer, error = cascade.validate_model_answer(
            {"prompt": "What is the capital of Vietnam?"},
            "factual knowledge",
            "The user wants a fact.\nThe question is factual.\nCapital of Vietnam is Hanoi.\n\nConstraint Checklist:\n1. Done",
            local_candidate=True,
        )
        self.assertTrue(ok)
        self.assertEqual(answer, "Capital of Vietnam is Hanoi.")
        self.assertIsNone(error)

    def test_factual_normalizer_prefers_drafted_answer_over_meta_sentence(self) -> None:
        raw = (
            "Since this is a single factual question, I will provide a direct answer.\n\n"
            "Plan: Identify the planet.\n\n"
            "Drafting the answer: Mars is the planet known as the Red Planet. "
            "Its surface contains iron oxide."
        )
        ok, answer, error = cascade.validate_model_answer(
            {"prompt": "Which planet is called the Red Planet?"},
            "factual knowledge",
            raw,
            local_candidate=True,
        )
        self.assertTrue(ok)
        self.assertEqual(answer, "Mars is the planet known as the Red Planet. Its surface contains iron oxide.")
        self.assertIsNone(error)

    def test_structural_factual_filter_recovers_claim_before_truncated_scaffold(self) -> None:
        raw = (
            "Thinking Process:\nThe protocol is DNS.\n"
            "Drafting the answer:\n* Sentence 1 (Direct Answer): State the protocol.\n* Sentence"
        )
        ok, answer, error = cascade.validate_model_answer(
            {"prompt": "Which protocol resolves host names?"},
            "factual knowledge",
            raw,
            local_candidate=True,
        )
        self.assertTrue(ok)
        self.assertEqual(answer, "The protocol is DNS.")
        self.assertIsNone(error)

    def test_structural_factual_filter_keeps_claim_and_drops_process_commentary(self) -> None:
        raw = (
            "Who developed the first successful polio vaccine?\n\n"
            "1. Identify the core question: Who developed the first successful polio vaccine?\n"
            "2. Recall relevant historical facts: The development of the polio vaccine is primarily attributed to Jonas Salk.\n"
            "3. Formulate the answer concisely: State the developer and significance.\n\n"
            "Self-Correction/Refinement: Ensure the answer is direct and adheres to the constraint.\n"
            "Drafting the response."
        )
        ok, answer, error = cascade.validate_model_answer(
            {"prompt": "Who developed the first successful polio vaccine?"},
            "factual knowledge",
            raw,
            local_candidate=True,
        )
        self.assertTrue(ok)
        self.assertEqual(answer, "The development of the polio vaccine is primarily attributed to Jonas Salk.")
        self.assertIsNone(error)

    def test_structural_factual_filter_rejects_process_only_output(self) -> None:
        raw = (
            "1. Identify the core question.\n"
            "2. Formulate a concise answer.\n"
            "Self-Correction: Ensure the answer follows every constraint.\n"
            "Drafting the response."
        )
        ok, _, error = cascade.validate_model_answer(
            {"prompt": "Who invented the device?"},
            "factual knowledge",
            raw,
            local_candidate=True,
        )
        self.assertFalse(ok)
        self.assertIn("analysis instead of final answer", error or "")

    def test_structural_filter_prefers_wwii_dates_over_first_person_instructions(self) -> None:
        raw = (
            "I must avoid markdown.\n1. Identify the end date.\n"
            "The Second World War officially ended in September 1945. "
            "Japan formally surrendered on August 15, 1945. Germany surrendered in May 1945."
        )
        self.assertEqual(
            cascade.extract_factual_answer(raw),
            "The Second World War officially ended in September 1945. "
            "Japan formally surrendered on August 15, 1945. Germany surrendered in May 1945.",
        )

    def test_structural_filter_prefers_absorption_claim_over_imperatives(self) -> None:
        raw = (
            "Constraint Check: Answer every part concisely. For comparisons include relationship, key properties or mechanism, and uses. No markdown.\n"
            "4. Determine the Answer: Plants absorb carbon dioxide (CO2).\n"
            "5. Draft the Answer:\n* Start with the direct answer.\n"
            "* Explain the role in photosynthesis.\n* Mention the chemical process briefly.\n"
            "7. Final Output Generation."
        )
        self.assertEqual(cascade.extract_factual_answer(raw), "Plants absorb carbon dioxide (CO2).")

    def test_structural_filter_drops_optional_context_scaffold(self) -> None:
        raw = (
            "It empties into the Mediterranean Sea.\n"
            "Sentence 2 (Optional/Elaboration): Provide context (e.g., the specific body of water).\n"
            "(Check length"
        )
        self.assertEqual(cascade.extract_factual_answer(raw), "It empties into the Mediterranean Sea.")

    def test_checklist_question_and_response_are_one_meta_unit(self) -> None:
        raw = (
            "Capital of Vietnam is Hanoi.\n"
            "For comparisons include mechanisms and uses? N/A (Factual question).\n"
            "Factual question answered? Yes."
        )
        self.assertEqual(cascade.extract_factual_answer(raw), "Capital of Vietnam is Hanoi.")

    def test_short_fragments_require_known_answer_shape(self) -> None:
        self.assertIsNone(cascade.extract_factual_answer("5/5\nState capital directly."))
        self.assertEqual(cascade.extract_factual_answer("Hanoi"), "Hanoi")
        self.assertEqual(cascade.extract_factual_answer("42"), "42")

    def test_general_state_imperatives_are_meta(self) -> None:
        raw = "State the capital directly.\nState a short answer.\nVietnam's capital is Hanoi."
        self.assertEqual(cascade.extract_factual_answer(raw), "Vietnam's capital is Hanoi.")

    def test_factual_ranking_beats_later_low_quality_fragments(self) -> None:
        raw = (
            "Capital of Vietnam is Hanoi.\n"
            "N/A (Factual question).\n5/5\nState the capital directly."
        )
        self.assertEqual(cascade.extract_factual_answer(raw), "Capital of Vietnam is Hanoi.")

    def test_vietnam_raw_trace_keeps_fact_over_checklist_scores_and_plan(self) -> None:
        raw = (
            "The user wants to know the capital of Vietnam.\n"
            "I need to provide a concise answer in at most three sentences.\n"
            "The question is a factual knowledge question.\n\n"
            "Capital of Vietnam is Hanoi.\n\n"
            "Constraint Checklist & Confidence Score:\n"
            "1. Answer every part in at most three concise sentences? Yes.\n"
            "2. For comparisons include relationship, key properties or mechanism, and uses? N/A (Factual question).\n"
            "3. No markdown? Yes.\n"
            "4. Factual knowledge question answered? Yes.\n\n"
            "Confidence Score: 5/5\n\n"
            "Plan: State the capital directly."
        )
        self.assertEqual(cascade.extract_factual_answer(raw), "Capital of Vietnam is Hanoi.")

    def test_local_code_without_task_tests_uses_code_not_model_self_checks(self) -> None:
        ok, answer, error = cascade.validate_model_answer(
            {"prompt": "Find the bug: def add(a, b): return a - b"},
            "code debugging",
            "def add(a, b): return a + b\n# SELF_CHECK:\nassert add(-1, 1) == -2",
            local_candidate=True,
        )
        self.assertTrue(ok)
        self.assertEqual(answer, "def add(a, b): return a + b")
        self.assertIsNone(error)

    def test_local_code_with_task_owned_tests_short_circuits_fireworks(self) -> None:
        fake_client = Mock()
        with (
            patch.object(cascade.local_llm, "can_attempt", return_value=True),
            patch.object(
                cascade.local_llm,
                "complete",
                return_value=FireworksResult(answer="def add(a, b):\n    return a + b"),
            ),
            patch.object(cascade, "CLIENT", fake_client),
        ):
            answer, meta = cascade.route_task(
                {"prompt": "Write add.", "tests": [{"call": "add(2, 3)", "expected": 5}]},
                "code generation",
            )

        self.assertEqual(answer, "def add(a, b):\n    return a + b")
        self.assertEqual(meta["path"], "local_llm")
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

    def test_renamed_local_code_answer_falls_through_to_fireworks(self) -> None:
        fake_client = Mock()
        fake_client.complete.return_value = FireworksResult(
            answer="def larger(a, b):\n    return a if a > b else b"
        )
        with (
            patch.object(cascade.local_llm, "can_attempt", return_value=True),
            patch.object(
                cascade.local_llm,
                "complete",
                return_value=FireworksResult(
                    answer="def find_larger(a, b):\n    return a if a > b else b"
                ),
            ),
            patch.object(cascade, "CLIENT", fake_client),
        ):
            answer, meta = cascade.route_task(
                {"prompt": "Write a Python function that returns the larger of two numbers."},
                "code generation",
            )

        self.assertEqual(answer, "def larger(a, b):\n    return a if a > b else b")
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

    def test_numbered_analysis_answer_is_rejected(self) -> None:
        ok, answer, error = cascade.validate_model_answer(
            {"prompt": "Summarize this text."},
            "summarization",
            "1. **Analyze the Request:** The user wants a summary.\n\nFinal summary.",
        )

        self.assertFalse(ok)
        self.assertIn("analysis instead of final answer", error or "")

    def test_logic_reasoning_trace_for_bloops_case_is_rejected(self) -> None:
        prompt = "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?"
        local_answer = (
            "The user is asking a question about logical deduction. This is a classic "
            "syllogism. Premise 1: all bloops are razzies. Premise 2: all razzies are "
            "lazzies. Therefore"
        )

        ok, answer, error = cascade.validate_model_answer(
            {"prompt": prompt},
            "logical reasoning",
            local_answer,
        )

        self.assertFalse(ok)
        self.assertIn("analysis instead of final answer", error or "")

    def test_visible_math_work_is_reduced_to_final_answer(self) -> None:
        ok, answer, error = cascade.validate_model_answer(
            {"prompt": "A word problem."},
            "math",
            "480 - 120 = 360\n360 + 240 = 600\n600 - 50 = 550\nAnswer: 550",
        )
        self.assertTrue(ok)
        self.assertEqual(answer, "550")
        self.assertIsNone(error)

    def test_visible_logic_work_is_reduced_to_final_answer(self) -> None:
        ok, answer, error = cascade.validate_model_answer(
            {"prompt": "A logic problem."},
            "logical reasoning",
            "A before B. C cannot be first.\nAnswer: A",
        )
        self.assertTrue(ok)
        self.assertEqual(answer, "A")
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
