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
            "error": deterministic_error,
        }

    fallback = config.FALLBACK_ANSWERS.get(category, config.FALLBACK_ANSWERS["unknown"])
    return fallback, {
        "path": "fallback",
        "tokens": result.total_tokens,
        "error": result.error or deterministic_error,
    }

