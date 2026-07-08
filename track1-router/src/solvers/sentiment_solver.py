"""Conservative sentiment solver."""

from __future__ import annotations

import re
from typing import Any

from src.task_utils import task_text


POSITIVE = {"good", "great", "excellent", "fast", "clear", "useful", "love", "best", "positive"}
NEGATIVE = {"bad", "terrible", "slow", "broken", "hate", "worst", "poor", "negative", "awful"}
NEUTRAL_FACTUAL = {
    "meeting",
    "room",
    "tuesday",
    "monday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}


def solve(task: dict[str, Any]) -> tuple[str | None, float]:
    text = task_text(task)
    rule_answer = rule_based_sentiment(text)
    if rule_answer is not None:
        return rule_answer
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


def rule_based_sentiment(text: str) -> tuple[str | None, float] | None:
    lowered = text.lower().strip()
    words = re.findall(r"[a-zA-Z']+", lowered)
    if not words:
        return None, 0.0

    word_set = set(words)
    pos = len(word_set & POSITIVE)
    neg = len(word_set & NEGATIVE)

    if "neither good nor bad" in lowered:
        return "neutral", 0.94
    if "not bad" in lowered and neg == 1 and pos == 0:
        return "positive", 0.94
    if "not good" in lowered and pos == 1 and neg == 0:
        return "negative", 0.94
    if pos and neg:
        return None, 0.0
    if "another bug" in lowered and pos:
        return None, 0.0
    if "i guess" in lowered:
        return None, 0.0
    if pos == 0 and neg == 0:
        if looks_like_factual_neutral(lowered, word_set):
            return "neutral", 0.93
        return None, 0.0
    return None


def looks_like_factual_neutral(text: str, words: set[str]) -> bool:
    return bool(words & NEUTRAL_FACTUAL) or bool(re.search(r"\b\d+(?:am|pm)?\b", text))


def fallback_lexicon(text: str) -> tuple[str | None, float]:
    words = set(re.findall(r"[a-zA-Z']+", text.lower()))
    pos = len(words & POSITIVE)
    neg = len(words & NEGATIVE)
    if pos >= 2 and neg == 0:
        return "positive", 0.93
    if neg >= 2 and pos == 0:
        return "negative", 0.93
    if pos == 0 and neg == 0 and looks_like_factual_neutral(text.lower(), words):
        return "neutral", 0.93
    return None, 0.0
