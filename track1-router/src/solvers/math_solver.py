"""Conservative arithmetic solver."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

from src.task_utils import task_text


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def solve(task: dict[str, Any]) -> tuple[Any | None, float]:
    text = task_text(task)
    expression = extract_expression(text)
    if not expression:
        return None, 0.0
    try:
        value = safe_eval(expression)
    except Exception:
        return None, 0.0
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value), 0.98


def extract_expression(text: str) -> str | None:
    if is_relation_question(text):
        return None
    word_expression = extract_word_expression(text)
    if word_expression:
        return word_expression
    candidates = re.findall(r"[-+*/%().\d\s^]+", text)
    candidates = [normalize_expression(item) for item in candidates if re.search(r"\d", item)]
    candidates = [item for item in candidates if re.search(r"[-+*/%]", item)]
    if len(candidates) != 1:
        return None
    return candidates[0]


def normalize_expression(expression: str) -> str:
    expression = expression.strip().replace("^", "**")
    while expression.endswith((".", "?")):
        expression = expression[:-1].rstrip()
    return expression


def is_relation_question(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in (" equal to ", " equals ", " greater than ", " less than "))


def extract_word_expression(text: str) -> str | None:
    lowered = text.lower().strip()
    number = r"(-?\d+(?:\.\d+)?)"

    match = re.search(rf"\bhalf of {number}\s+minus\s+{number}\b", lowered)
    if match:
        first, second = match.groups()
        return f"({first} / 2) - {second}"

    match = re.search(rf"\badd\s+{number}\s+and\s+{number},?\s+then\s+multiply\s+by\s+{number}\b", lowered)
    if match:
        first, second, third = match.groups()
        return f"({first} + {second}) * {third}"

    match = re.search(rf"\b{number}\s+divided by\s+{number}(?:,\s*divided by\s+{number})+\b", lowered)
    if match:
        numbers = re.findall(number, match.group(0))
        return " / ".join(numbers)

    match = re.search(rf"\b{number}\s+mod\s+{number}\b", lowered)
    if match:
        first, second = match.groups()
        return f"{first} % {second}"

    return None


def safe_eval(expression: str) -> int | float:
    node = ast.parse(expression, mode="eval")
    return _eval(node.body)


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")
