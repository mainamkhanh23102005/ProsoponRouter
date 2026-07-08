"""Starter local scorer for deterministic paths."""

from __future__ import annotations

import json
from pathlib import Path

from src.cascade import route_task
from src.classify import classify


def main() -> None:
    dataset_path = Path(__file__).parent / "data" / "dev_tasks.json"
    if not dataset_path.exists():
        raise SystemExit("Run python -m eval.make_dataset first")
    tasks = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, int]] = {}
    for task in tasks:
        category = classify(task)
        answer, meta = route_task(task, category)
        row = rows.setdefault(category, {"total": 0, "correct": 0, "tokens": 0})
        row["total"] += 1
        row["tokens"] += int(meta.get("tokens", 0) or 0)
        if answer == task.get("answer"):
            row["correct"] += 1

    print("category,total,correct,accuracy,tokens")
    for category, row in sorted(rows.items()):
        accuracy = row["correct"] / row["total"] if row["total"] else 0
        print(f"{category},{row['total']},{row['correct']},{accuracy:.3f},{row['tokens']}")


if __name__ == "__main__":
    main()

