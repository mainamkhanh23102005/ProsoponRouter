"""Routing cascade from free solvers to Fireworks fallback."""

from __future__ import annotations

import re
from typing import Any

from src import config
from src.code_validation import validate_code_answer
from src.fireworks_client import FireworksClient
from src import local_llm
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

    local_error = None
    if local_llm.can_attempt(task, category):
        local_result = local_llm.complete(task, category)
        local_ok, local_answer, local_failure_feedback = validate_model_answer(task, category, local_result.answer)
        if local_ok:
            return local_answer, {
                "path": "local_llm",
                "tokens": 0,
                "local_tokens": local_result.total_tokens,
                "attempts": 1,
                "error": deterministic_error,
            }
        local_error = local_failure_feedback or local_result.error

    result = CLIENT.complete(task, category)
    answer_ok, accepted_answer, failure_feedback = validate_model_answer(task, category, result.answer)
    if answer_ok:
        return accepted_answer, {
            "path": "fireworks",
            "tokens": result.total_tokens,
            "attempts": 1,
            "error": local_error or deterministic_error,
        }

    total_tokens = result.total_tokens
    if result.answer is not None and policy.retry_on_invalid and not is_empty_content_error(result.error):
        retry_feedback = result.error or failure_feedback or f"expected {category} answer format; got {result.answer!r}"
        retry_result = CLIENT.complete(task, category, retry_feedback=retry_feedback)
        total_tokens += retry_result.total_tokens
        retry_ok, retry_answer, retry_failure_feedback = validate_model_answer(task, category, retry_result.answer)
        if retry_ok:
            return retry_answer, {
                "path": "fireworks_retry",
                "tokens": total_tokens,
                "attempts": 2,
                "retried": True,
                "error": deterministic_error,
            }
        retry_error = retry_failure_feedback or retry_result.error or result.error or deterministic_error
    else:
        retry_error = result.error or deterministic_error

    fallback = config.FALLBACK_ANSWERS.get(category, config.FALLBACK_ANSWERS["unknown"])
    return fallback, {
        "path": "fallback",
        "tokens": total_tokens,
        "attempts": 2 if result.answer is not None and policy.retry_on_invalid and not is_empty_content_error(result.error) else 1,
        "retried": bool(result.answer is not None and policy.retry_on_invalid and not is_empty_content_error(result.error)),
        "error": retry_error,
    }


def is_empty_content_error(error: str | None) -> bool:
    return bool(error and error.startswith("empty content, message keys were:"))


def validate_model_answer(task: dict[str, Any], category: str, answer: Any) -> tuple[bool, Any, str | None]:
    if answer is None:
        return False, answer, None
    answer = normalize_model_answer(category, answer)
    if is_meta_answer(category, answer):
        return False, answer, f"model returned analysis instead of final answer: {str(answer)[:80]!r}"
    if not validate_answer(category, answer):
        return False, answer, f"expected {category} answer format; got {answer!r}"
    if category in {"code debugging", "code generation"}:
        validation = validate_code_answer(task, str(answer))
        if not validation.ok:
            return False, answer, validation.error
        return True, validation.code, None
    return True, answer, None


def normalize_model_answer(category: str, answer: Any) -> Any:
    if category != "sentiment" or not isinstance(answer, str):
        return answer
    match = re.match(r"^\s*(positive|negative|neutral)\b\s*(?::|-)?\s*(.*)$", answer, re.IGNORECASE | re.S)
    if not match:
        return answer
    label = match.group(1).lower()
    reason = " ".join(match.group(2).split()) or "model label"
    return f"{label}: {reason}"


def is_meta_answer(category: str, answer: Any) -> bool:
    if not isinstance(answer, str):
        return False
    if category in {"code debugging", "code generation"}:
        return False
    lowered = answer.lstrip().lower()
    if lowered.startswith((
        "thinking process",
        "analysis:",
        "reasoning:",
        "the user wants",
        "the user asked",
        "i need to",
        "i should",
        "let me",
    )):
        return True
    if has_reasoning_scaffold(lowered):
        return True
    if has_truncated_reasoning_shape(lowered):
        return True
    return is_too_long_for_final_answer(category, answer)


def has_reasoning_scaffold(lowered: str) -> bool:
    patterns = (
        r"\bthe user (is asking|wants|asked)\b",
        r"\bthis is a classic syllogism\b",
        r"\bpremise\s*\d+\b",
        r"\bnumbered reasoning steps?\b",
        r"\bconstraint\s*\d+\b",
        r"\b(?:first|second|third),\s+i\b",
        r"\bi need to\b",
        r"\bi should\b",
        r"\blet me\b",
        r"^\d+[.)]\s+\*{0,2}(analyze|identify|list|determine)\b",
        r"\n\s*\d+[.)]\s+\*{0,2}(analyze|identify|list|determine)\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def has_truncated_reasoning_shape(lowered: str) -> bool:
    stripped = lowered.rstrip()
    if not stripped:
        return False
    if re.search(r"\b(therefore|because|and|or|so|then|but|with|from|to|the|a|an)\s*$", stripped):
        return True
    if re.search(r"[:;,]\s*$", stripped):
        return True
    if re.search(r"\n\s*[-*]\s+\S.{0,80}$", stripped) and not re.search(r"[.!?)]\s*$", stripped):
        return True
    return False


def is_too_long_for_final_answer(category: str, answer: str) -> bool:
    words = answer.split()
    lines = [line for line in answer.splitlines() if line.strip()]
    if category == "logical reasoning":
        return len(words) > 24 or len(lines) > 2
    if category == "math":
        return len(words) > 12 or len(lines) > 1
    if category == "sentiment":
        return len(words) > 40 or len(lines) > 3
    if category == "factual knowledge":
        return len(words) > 80 or len(lines) > 4
    return False
