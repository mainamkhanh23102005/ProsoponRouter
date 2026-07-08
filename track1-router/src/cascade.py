"""Routing cascade from free solvers to Fireworks fallback."""

from __future__ import annotations

from typing import Any

from src import config
from src.fireworks_client import FireworksClient
from src.solvers import get_solver
from src.validate import validate_answer


CLIENT = FireworksClient()


def route_task(task: dict[str, Any], category: str) -> tuple[Any, dict[str, Any]]:
    policy = config.POLICY.get(category, config.POLICY["unknown"])

    if policy.free_enabled:
        solver = get_solver(category)
        if solver is not None:
            try:
                answer, confidence = solver.solve(task)
                if (
                    answer is not None
                    and confidence >= policy.min_confidence
                    and validate_answer(category, answer)
                ):
                    return answer, {"path": "deterministic", "confidence": confidence, "tokens": 0}
            except Exception as exc:  # noqa: BLE001 - solvers must never break the cascade.
                deterministic_error = str(exc)
            else:
                deterministic_error = None
        else:
            deterministic_error = "no_solver"
    else:
        deterministic_error = "free_disabled"

    result = CLIENT.complete(task, category)
    if result.answer is not None and validate_answer(category, result.answer):
        return result.answer, {
            "path": "fireworks",
            "tokens": result.total_tokens,
            "attempts": 1,
            "error": deterministic_error,
        }

    total_tokens = result.total_tokens
    if result.answer is not None and policy.retry_on_invalid:
        retry_feedback = f"expected {category} answer format; got {result.answer!r}"
        retry_result = CLIENT.complete(task, category, retry_feedback=retry_feedback)
        total_tokens += retry_result.total_tokens
        if retry_result.answer is not None and validate_answer(category, retry_result.answer):
            return retry_result.answer, {
                "path": "fireworks_retry",
                "tokens": total_tokens,
                "attempts": 2,
                "retried": True,
                "error": deterministic_error,
            }
        retry_error = retry_result.error or result.error or deterministic_error
    else:
        retry_error = result.error or deterministic_error

    fallback = config.FALLBACK_ANSWERS.get(category, config.FALLBACK_ANSWERS["unknown"])
    return fallback, {
        "path": "fallback",
        "tokens": total_tokens,
        "attempts": 2 if result.answer is not None and policy.retry_on_invalid else 1,
        "retried": bool(result.answer is not None and policy.retry_on_invalid),
        "error": retry_error,
    }
