"""Task normalization helpers."""

from __future__ import annotations

import json
from typing import Any


TEXT_KEYS = ("text", "question", "prompt", "input", "content", "task")
ID_KEYS = ("id", "task_id", "uid", "uuid")


def task_id(task: dict[str, Any], index: int) -> str:
    for key in ID_KEYS:
        value = task.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"task-{index}"


def task_text(task: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        if key not in task:
            continue
        value = task[key]
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return json.dumps(task, ensure_ascii=False, sort_keys=True)

