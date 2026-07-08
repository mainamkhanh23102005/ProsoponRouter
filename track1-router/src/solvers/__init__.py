"""Solver registry."""

from __future__ import annotations

from typing import Any, Protocol


class Solver(Protocol):
    def solve(self, task: dict[str, Any]) -> tuple[Any | None, float]:
        ...


def get_solver(category: str) -> Solver | None:
    if category == "math":
        from src.solvers import math_solver

        return math_solver
    if category == "sentiment":
        from src.solvers import sentiment_solver

        return sentiment_solver
    if category == "ner":
        from src.solvers import ner_solver

        return ner_solver
    if category == "summarization":
        from src.solvers import summarize_solver

        return summarize_solver
    if category == "factual knowledge":
        from src.solvers import factual_solver

        return factual_solver
    if category == "code debugging":
        from src.solvers import code_solver

        return code_solver
    if category == "logical reasoning":
        from src.solvers import logic_solver

        return logic_solver
    return None

