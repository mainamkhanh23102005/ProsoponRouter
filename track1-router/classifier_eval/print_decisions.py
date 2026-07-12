from __future__ import annotations

import json
from pathlib import Path

from src.classify import classify


for label, filename in (
    ("OLD_BATCH", "old_missed_prompts.json"),
    ("NEW_BATCH", "fresh_24_tasks.json"),
):
    tasks = json.loads((Path(__file__).parent / filename).read_text(encoding="utf-8"))
    print(label)
    for task in tasks:
        print(f"{task['task_id']}: {classify(task)}")
