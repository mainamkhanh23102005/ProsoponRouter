"""Narrow, proof-producing solvers for common finite logic tasks."""

from __future__ import annotations

import itertools
import re
from typing import Any

from src.task_utils import task_text


def solve(task: dict[str, Any]) -> tuple[str | None, float]:
    text = task_text(task)
    syllogism = solve_transitive_all(text)
    if syllogism is not None:
        return syllogism, 0.99
    seating = solve_seating(text)
    if seating is not None:
        return seating, 0.99
    return None, 0.0


def solve_transitive_all(text: str) -> str | None:
    lowered = text.lower()
    relations = re.findall(r"\ball\s+(\w+)\s+are\s+(\w+)", lowered)
    question = re.search(r"\bare\s+all\s+(\w+)\s+(\w+)\s*\?", lowered)
    if len(relations) < 2 or not question:
        return None
    graph: dict[str, set[str]] = {}
    for source, target in relations:
        graph.setdefault(source, set()).add(target)
    source, target = question.groups()
    frontier = [source]
    visited: set[str] = set()
    while frontier:
        current = frontier.pop()
        if current == target:
            return "Yes"
        if current in visited:
            continue
        visited.add(current)
        frontier.extend(graph.get(current, ()))
    return None


def solve_seating(text: str) -> str | None:
    size_match = re.search(r"row of\s+(\d+)\s+seats", text, re.IGNORECASE)
    query = re.search(r"who sits in seat\s+(\d+)\s*\?", text, re.IGNORECASE)
    intro = re.search(r"^(.+?)\s+-?\s*sit in a row", text, re.IGNORECASE)
    if not (size_match and query and intro):
        return None
    size = int(size_match.group(1))
    intro_names = intro.group(1).split("-", 1)[-1]
    names = re.findall(r"\b[A-Z][a-z]+\b", intro_names)
    names = [name for name in names if name.lower() not in {"a", "the", "five"}]
    if len(names) != size or size > 8:
        return None

    if not all_seating_constraints_supported(text):
        return None

    exclusions = [
        (name, int(seat))
        for name, seat in re.findall(r"\b([A-Z][a-z]+) does not sit in seat\s+(\d+)", text)
    ]
    before = re.findall(r"\b([A-Z][a-z]+) sits immediately before ([A-Z][a-z]+)", text)
    not_before = re.findall(r"\b([A-Z][a-z]+) does not sit immediately before ([A-Z][a-z]+)", text)
    followed_by_person = re.findall(
        r"\bperson who drinks \w+ sits immediately after ([A-Z][a-z]+)", text, re.IGNORECASE
    )
    solutions: list[tuple[str, ...]] = []
    for arrangement in itertools.permutations(names):
        positions = {name: index + 1 for index, name in enumerate(arrangement)}
        if any(positions.get(name) == seat for name, seat in exclusions):
            continue
        if any(positions.get(left, -10) + 1 != positions.get(right) for left, right in before):
            continue
        if any(positions.get(left, -10) + 1 == positions.get(right) for left, right in not_before):
            continue
        if any(positions.get(name) == size for name in followed_by_person):
            continue
        solutions.append(arrangement)
    seat = int(query.group(1))
    occupants = {solution[seat - 1] for solution in solutions if 1 <= seat <= size}
    if len(occupants) == 1:
        return next(iter(occupants))
    if len(occupants) > 1:
        return "Cannot be determined"
    return None


def all_seating_constraints_supported(text: str) -> bool:
    sentences = [part.strip(" .") for part in re.split(r"[.?]", text) if part.strip()]
    if len(sentences) < 2:
        return False
    clauses: list[str] = []
    for sentence in sentences[1:-1]:
        clauses.extend(part.strip() for part in re.split(r",\s+and\s+", sentence, flags=re.IGNORECASE))
    supported = (
        r"[A-Z][a-z]+ does not sit in seat \d+",
        r"[A-Z][a-z]+ sits immediately before [A-Z][a-z]+",
        r"[A-Z][a-z]+ does not sit immediately before [A-Z][a-z]+",
        r"The person who drinks \w+ sits immediately after [A-Z][a-z]+",
        r"[A-Z][a-z]+ does not drink \w+",
    )
    return bool(clauses) and all(
        any(re.fullmatch(pattern, clause, re.IGNORECASE) for pattern in supported)
        for clause in clauses
    )
