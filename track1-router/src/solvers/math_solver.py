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
    structured_answer = solve_structured_word_problem(text)
    if structured_answer is not None:
        return structured_answer, 0.99
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


def solve_structured_word_problem(text: str) -> str | None:
    lowered = text.lower()
    inventory = re.search(
        r"starts? with\s+([\d,]+)\s+units?.*?sells?\s+(\d+(?:\.\d+)?)%\s+of\s+stock"
        r".*?restocks?\s+([\d,]+)\s+units?.*?sells?\s+([\d,]+)\s+units?",
        lowered,
        re.S,
    )
    if inventory:
        start, percent, restock, final_sale = inventory.groups()
        value = (
            float(start.replace(",", "")) * (1 - float(percent) / 100)
            + float(restock.replace(",", ""))
            - float(final_sale.replace(",", ""))
        )
        return format_number(value)

    recipe = re.search(
        r"requires?\s+(\d+)\s*/\s*(\d+)\s+cups?[^.]*?for\s+(\d+)\s+cookies?.*?"
        r"needed\s+for\s+(\d+)\s+cookies?.*?costs?\s+\$(\d+(?:\.\d+)?)\s+per\s+cup",
        lowered,
        re.S,
    )
    if recipe:
        numerator, denominator, base_count, target_count, unit_cost = recipe.groups()
        cups = (int(numerator) / int(denominator)) * int(target_count) / int(base_count)
        cost = cups * float(unit_cost)
        return f"{format_number(cups)} cups; ${cost:.2f}"
    return None


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def extract_expression(text: str) -> str | None:
    if is_relation_question(text):
        return None
    word_expression = extract_word_expression(text)
    if word_expression:
        return word_expression
    candidates = re.findall(r"[-+*/%().\d\s^]+", text)
    candidates = [normalize_expression(item) for item in candidates if re.search(r"\d", item)]
    if any(is_date_shaped_expression(item) for item in candidates):
        return None
    candidates = [item for item in candidates if re.search(r"[-+*/%]", item)]
    if len(candidates) != 1:
        return None
    candidate = candidates[0]
    if len(numeric_tokens(candidate)) != len(numeric_tokens(text)):
        return None
    return candidate


def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text)


def normalize_expression(expression: str) -> str:
    expression = expression.strip().replace("^", "**")
    while expression.endswith((".", "?")):
        expression = expression[:-1].rstrip()
    return expression


def is_date_shaped_expression(expression: str) -> bool:
    return bool(re.fullmatch(r"\d{1,4}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4}", expression.strip()))


def is_relation_question(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in (" equal to ", " equals ", " greater than ", " less than "))


def extract_word_expression(text: str) -> str | None:
    lowered = text.lower().strip()
    number = r"(-?\d+(?:\.\d+)?)"

    rate_expression = extract_rate_sequence(lowered)
    if rate_expression:
        return rate_expression

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


def extract_rate_sequence(text: str) -> str | None:
    start = re.search(r"\b(?:starts? with|initially has?)\s+(\d+(?:\.\d+)?)", text)
    if not start:
        return None
    operations = re.findall(
        r"\b(drains?|loses?|uses?|refill(?:ed|s)?|gains?|adds?)\b[^.]*?"
        r"(?:at\s+)?(\d+(?:\.\d+)?)\s+[^.]*?per\s+(?:minute|hour|day)\s+for\s+"
        r"(\d+(?:\.\d+)?)\s+(?:minutes?|hours?|days?)",
        text,
    )
    if not operations:
        return None
    pieces = [start.group(1)]
    for verb, rate, duration in operations:
        sign = "+" if verb.startswith(("refill", "gain", "add")) else "-"
        pieces.append(f" {sign} ({rate} * {duration})")
    return "".join(pieces)


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
