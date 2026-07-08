"""Container entrypoint: read tasks, route answers, write results."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from src import config
from src.cascade import route_task
from src.classify import classify
from src.task_utils import task_id


def load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        raw_tasks = data
    elif isinstance(data, dict) and isinstance(data.get("tasks"), list):
        raw_tasks = data["tasks"]
    else:
        raise ValueError("task input must be a list or an object with a tasks list")
    return [item if isinstance(item, dict) else {"input": item} for item in raw_tasks]


def format_results(answer_by_id: dict[str, Any]) -> Any:
    return [{"task_id": key, "answer": value} for key, value in answer_by_id.items()]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def run(input_path: Path, output_path: Path) -> int:
    answers: dict[str, Any] = {}
    tasks = load_tasks(input_path)
    started_at = time.monotonic()

    for index, task in enumerate(tasks):
        current_id = task_id(task, index)
        try:
            category = classify(task)
            if watchdog_expired(started_at):
                answer = config.FALLBACK_ANSWERS.get(category, config.FALLBACK_ANSWERS["unknown"])
                meta = {"path": "watchdog_fallback", "tokens": 0}
            else:
                answer, meta = route_task(task, category)
        except Exception as exc:  # noqa: BLE001 - every task must get a result.
            category = "unknown"
            answer = config.FALLBACK_ANSWERS["unknown"]
            meta = {"path": "fallback", "tokens": 0, "error": str(exc)}
        answers[current_id] = answer
        print(
            f"task_id={current_id} category={category} path={meta.get('path')} tokens={meta.get('tokens', 0)}",
            file=sys.stderr,
        )

    atomic_write_json(output_path, format_results(answers))
    return 0


def watchdog_expired(started_at: float) -> bool:
    if config.LATENCY_LIMIT_SECONDS <= 0:
        return False
    elapsed = time.monotonic() - started_at
    usable_seconds = max(0.0, config.LATENCY_LIMIT_SECONDS - config.LATENCY_RESERVE_SECONDS)
    return elapsed >= usable_seconds


def main() -> int:
    try:
        return run(Path(config.TASK_INPUT_PATH), Path(config.RESULTS_OUTPUT_PATH))
    except Exception as exc:  # noqa: BLE001 - malformed global input still exits cleanly.
        print(f"fatal={exc}", file=sys.stderr)
        atomic_write_json(Path(config.RESULTS_OUTPUT_PATH), format_results({}))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
