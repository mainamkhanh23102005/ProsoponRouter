"""Rule-based category classifier."""

from __future__ import annotations

import re
from typing import Any

from src.config import VALID_CATEGORIES
from src.task_utils import task_text


CATEGORY_ALIASES = {
    "named entity recognition": "ner",
    "entity extraction": "ner",
    "entities": "ner",
    "natural language inference": "logical reasoning",
    "logic": "logical reasoning",
    "reasoning": "logical reasoning",
    "debug": "code debugging",
    "debugging": "code debugging",
    "code": "code generation",
    "generation": "code generation",
    "summary": "summarization",
    "summarize": "summarization",
    "factual": "factual knowledge",
    "knowledge": "factual knowledge",
}


def normalize_category(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower().replace("_", " ").replace("-", " ")
    raw = re.sub(r"\s+", " ", raw)
    if raw in VALID_CATEGORIES:
        return raw
    return CATEGORY_ALIASES.get(raw)


def classify(task: dict[str, Any]) -> str:
    for key in ("category", "type", "task_type", "label"):
        category = normalize_category(task.get(key))
        if category:
            return category

    text = task_text(task).lower()
    if re.search(r"\b(summarize|summary|tl;dr)\b", text):
        return "summarization"
    if re.search(r"\b(classify|label|determine|analyze)\b.{0,40}\bsentiment\b|\bsentiment\s+of\b", text):
        return "sentiment"
    if re.search(
        r"\b(extract|find|identify|list)\b.{0,50}\b(named entities|entities|person|people|organizations?|locations?|dates?)\b|\bner\b",
        text,
    ):
        return "ner"
    if re.search(r"\b(debug|find the bug|fix the bug|traceback|syntaxerror)\b", text):
        return "code debugging"
    if re.search(r"\b(write|implement|create)\b.{0,40}\b(function|class|program|script)\b", text):
        return "code generation"
    if re.search(
        r"\b(if all|therefore|must be|can be true|logic|each owns?|does not|do not|"
        r"immediately after|immediately before|next to|exactly one)\b",
        text,
    ):
        return "logical reasoning"
    if re.search(r"\b(calculate|compute|evaluate|what is|what's)\b.{0,40}\d+\s*[-+*/%^]\s*\d+", text):
        return "math"
    if re.search(
        r"\bhalf of\s*-?\d|"
        r"\badd\s+-?\d+(?:\.\d+)?.{0,20}\b(?:and|to)\b.{0,20}-?\d|"
        r"-?\d+(?:\.\d+)?.{0,12}\b(?:divided by|mod|minus|plus|times)\b.{0,12}-?\d|"
        r"\bmultiply by\s*-?\d",
        text,
    ):
        return "math"
    if re.search(
        r"\b(what is|what are|who is|where is|when did|why does|how does|explain|define|"
        r"difference between|name one|name a|name the)\b",
        text,
    ):
        return "factual knowledge"
    return "unknown"
