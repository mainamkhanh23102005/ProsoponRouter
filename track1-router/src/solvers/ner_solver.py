"""Regex-first NER solver for high-precision entities."""

from __future__ import annotations

import re
from typing import Any

from src.task_utils import task_text


PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "DATE": re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    "MONEY": re.compile(r"(?<!\w)(?:\$|USD\s*)\d+(?:\.\d{2})?\b", re.IGNORECASE),
    "PERCENT": re.compile(r"\b\d+(?:\.\d+)?%"),
}


def solve(task: dict[str, Any]) -> tuple[list[dict[str, str]] | None, float]:
    text = task_text(task)
    entities: list[dict[str, str]] = []
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append({"text": match.group(0), "label": label})
    if not entities:
        return None, 0.0
    entities.sort(key=lambda item: text.find(item["text"]))
    return entities, 0.92

