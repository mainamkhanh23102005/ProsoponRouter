"""Measure zero-token solver coverage and accepted-answer accuracy by category."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from src.solvers import get_solver


MIN_ACCEPTED_ACCURACY = 0.90


def equivalent(category: str, actual: object, expected: object) -> bool:
    left = str(actual).strip().lower()
    right = str(expected).strip().lower()
    if category == "sentiment":
        return left.startswith(right + ":")
    return left == right


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "evaluation/local_challenge.json")
    cases = json.loads(path.read_text(encoding="utf-8"))
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "accepted": 0, "correct": 0})
    for case in cases:
        category = case["category"]
        stats[category]["total"] += 1
        solver = get_solver(category)
        answer, confidence = solver.solve(case) if solver else (None, 0.0)
        if answer is None:
            continue
        stats[category]["accepted"] += 1
        if equivalent(category, answer, case["expected"]):
            stats[category]["correct"] += 1

    report = {}
    failed = False
    for category, values in sorted(stats.items()):
        accepted = values["accepted"]
        accuracy = values["correct"] / accepted if accepted else 0.0
        coverage = accepted / values["total"] if values["total"] else 0.0
        report[category] = {**values, "accepted_accuracy": accuracy, "coverage": coverage}
        if accepted and accuracy < MIN_ACCEPTED_ACCURACY:
            failed = True
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
