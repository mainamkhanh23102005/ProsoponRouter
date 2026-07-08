"""Conservative sentiment solver."""

from __future__ import annotations

import re
from typing import Any

from src.task_utils import task_text


POSITIVE = {"good", "great", "excellent", "fast", "clear", "useful", "love", "best", "positive"}
NEGATIVE = {"bad", "terrible", "slow", "broken", "hate", "worst", "poor", "negative", "awful"}


def solve(task: dict[str, Any]) -> tuple[str | None, float]:
    text = task_text(task)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        score = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
        if score >= 0.35:
            return "positive", 0.96
        if score <= -0.35:
            return "negative", 0.96
        if abs(score) <= 0.10:
            return "neutral", 0.93
        return None, 0.0
    except Exception:
        return fallback_lexicon(text)


def fallback_lexicon(text: str) -> tuple[str | None, float]:
    words = set(re.findall(r"[a-zA-Z']+", text.lower()))
    pos = len(words & POSITIVE)
    neg = len(words & NEGATIVE)
    if pos >= 2 and neg == 0:
        return "positive", 0.93
    if neg >= 2 and pos == 0:
        return "negative", 0.93
    if pos == 0 and neg == 0:
        return "neutral", 0.90
    return None, 0.0

