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
    if re.search(r"\b(sentiment|positive|negative|neutral)\b", text):
        return "sentiment"
    if re.search(r"\b(summarize|summary|tl;dr)\b", text):
        return "summarization"
    if re.search(r"\b(entity|entities|ner|email|phone|date)\b", text):
        return "ner"
    if re.search(r"\b(debug|bug|traceback|exception|syntaxerror)\b", text):
        return "code debugging"
    if re.search(r"\b(write|implement|function|class|program)\b", text):
        return "code generation"
    if re.search(r"\b(if all|therefore|must be|can be true|logic)\b", text):
        return "logical reasoning"
    if re.search(r"\d+\s*[-+*/^]\s*\d+", text):
        return "math"
    return "unknown"

