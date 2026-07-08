"""Per-category answer validators."""

from __future__ import annotations

from typing import Any


SENTIMENT_LABELS = {"positive", "negative", "neutral"}


def validate_answer(category: str, answer: Any) -> bool:
    if answer is None:
        return False
    if category == "sentiment":
        if not isinstance(answer, str):
            return False
        lowered = answer.strip().lower()
        return any(lowered.startswith(f"{label}: ") for label in SENTIMENT_LABELS)
    if category == "ner":
        return isinstance(answer, (list, dict, str))
    if category == "math":
        return isinstance(answer, (int, float, str)) and str(answer).strip() != ""
    return str(answer).strip() != ""
