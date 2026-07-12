"""Conservative extractive summarization for short, sentence-based prompts."""

from __future__ import annotations

import re
from typing import Any

from src.task_utils import task_text


def solve(task: dict[str, Any]) -> tuple[str | None, float]:
    text = task_text(task).strip()
    if not re.match(r"summarize\b", text, re.I) or ":" not in text:
        return None, 0.0
    instruction, source = text.split(":", 1)
    count_match = re.search(r"exactly\s+(one|two|three)\s+sentences?", instruction, re.I)
    requested = {"one": 1, "two": 2, "three": 3}.get(
        count_match.group(1).lower() if count_match else "", 1
    )
    source = source.strip().strip("'\"")
    sentences = re.findall(r"[^.!?]+[.!?]", source)
    if not sentences:
        return None, 0.0
    summary = " ".join(sentence.strip() for sentence in sentences[:requested])
    return summary, 0.91
