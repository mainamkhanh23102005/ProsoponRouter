"""Create a tiny starter dev dataset."""

from __future__ import annotations

import json
from pathlib import Path


DATA = [
    {"id": "math-dev-1", "category": "math", "question": "What is 10 / 2 + 7?", "answer": "12"},
    {"id": "sent-dev-1", "category": "sentiment", "text": "The tool is excellent and useful.", "answer": "positive"},
    {"id": "ner-dev-1", "category": "ner", "text": "Email a@b.com by 2026-07-11.", "answer": [{"text": "a@b.com", "label": "EMAIL"}, {"text": "2026-07-11", "label": "DATE"}]},
]


def main() -> None:
    output = Path(__file__).parent / "data" / "dev_tasks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(DATA, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

