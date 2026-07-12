"""Audit zero-token acceptance on public datasets never used by production code."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src import config
from src.solvers import get_solver
from src.validate import validate_answer


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Metric:
    total: int = 0
    accepted: int = 0
    correct: int = 0

    def record(self, actual: str | None, expected: str) -> None:
        self.total += 1
        if actual is None:
            return
        self.accepted += 1
        self.correct += int(actual == expected)

    @property
    def coverage(self) -> float:
        return self.accepted / self.total if self.total else 0.0

    @property
    def accepted_precision(self) -> float:
        return self.correct / self.accepted if self.accepted else 0.0

    @property
    def wilson_lower_95(self) -> float:
        if not self.accepted:
            return 0.0
        z = 1.96
        n = self.accepted
        p = self.accepted_precision
        denominator = 1 + z * z / n
        centre = p + z * z / (2 * n)
        margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
        return (centre - margin) / denominator

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "correct": self.correct,
            "coverage": round(self.coverage, 4),
            "accepted_precision": round(self.accepted_precision, 4),
            "wilson_lower_95": round(self.wilson_lower_95, 4),
        }


def normalize_number(value: str) -> str:
    cleaned = value.replace(",", "").strip()
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return cleaned
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized


def gsm8k_final_answer(answer: str) -> str:
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*$", answer)
    if not match:
        raise ValueError("GSM8K row has no final answer marker")
    return normalize_number(match.group(1))


def sentiment_label(answer: str) -> str | None:
    lowered = answer.strip().lower()
    if lowered.startswith("positive:"):
        return "1"
    if lowered.startswith("negative:"):
        return "0"
    return None


def parse_ner_answer(answer: str) -> tuple[tuple[str, str], ...] | None:
    if answer.strip().upper() == "NONE":
        return ()
    entities: list[tuple[str, str]] = []
    for item in answer.split(";"):
        if ":" not in item:
            return None
        label, text = item.split(":", 1)
        label = label.strip().upper()
        if label not in {"PERSON", "ORG", "ORGANIZATION", "LOCATION", "DATE"} or not text.strip():
            return None
        entities.append((label, text.strip()))
    return tuple(entities)


def accepted_solver_answer(category: str, task: dict[str, str]) -> str | None:
    policy = config.POLICY[category]
    if not policy.free_enabled:
        return None
    solver = get_solver(category)
    if solver is None:
        return None
    answer, confidence = solver.solve(task)
    if answer is None or confidence < policy.min_confidence or not validate_answer(category, answer):
        return None
    return str(answer).strip()


def sample_rows(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    return rng.sample(rows, min(limit, len(rows)))


def evaluate_gsm8k(limit: int = 250) -> Metric:
    path = ROOT / "evaluation" / "external" / "gsm8k-test.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metric = Metric()
    for row in sample_rows(rows, limit, 23102005):
        actual = accepted_solver_answer("math", {"prompt": row["question"]})
        actual = normalize_number(actual) if actual is not None else None
        metric.record(actual, gsm8k_final_answer(row["answer"]))
    return metric


def evaluate_sst2(limit: int = 250) -> Metric:
    path = ROOT / "evaluation" / "external" / "sst2" / "SST-2" / "dev.tsv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    metric = Metric()
    for row in sample_rows(rows, limit, 23102005):
        prompt = f"Classify the sentiment of this review: {row['sentence']}"
        raw = accepted_solver_answer("sentiment", {"prompt": prompt})
        metric.record(sentiment_label(raw) if raw is not None else None, row["label"])
    return metric


def conll_sentences() -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    path = ROOT / "evaluation" / "external" / "conll03" / "conll03_en_iobes" / "eng.test.tsv"
    sentences: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    tokens: list[str] = []
    entities: list[tuple[str, str]] = []
    current_label: str | None = None
    current_tokens: list[str] = []

    def finish_entity() -> None:
        nonlocal current_label, current_tokens
        if current_label in {"PER", "ORG", "LOC"} and current_tokens:
            mapped = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOCATION"}[current_label]
            entities.append((mapped, " ".join(current_tokens)))
        current_label, current_tokens = None, []

    def finish_sentence() -> None:
        finish_entity()
        if tokens and tokens[0] != "-DOCSTART-":
            sentence = " ".join(tokens).replace(" ,", ",").replace(" .", ".")
            sentences.append((sentence, tuple(entities)))
        tokens.clear()
        entities.clear()

    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if not line.strip():
            finish_sentence()
            continue
        token, tag = line.split("\t", 1)
        tokens.append(token)
        prefix, _, label = tag.partition("-")
        if prefix in {"S", "B"}:
            finish_entity()
            current_label, current_tokens = label, [token]
            if prefix == "S":
                finish_entity()
        elif prefix in {"I", "E"} and current_label == label:
            current_tokens.append(token)
            if prefix == "E":
                finish_entity()
        else:
            finish_entity()
    return sentences


def evaluate_conll(limit: int = 250) -> Metric:
    metric = Metric()
    rows = conll_sentences()
    for sentence, expected in sample_rows(rows, limit, 23102005):
        prompt = f"Extract all named entities: '{sentence}'"
        raw = accepted_solver_answer("ner", {"prompt": prompt})
        actual = parse_ner_answer(raw) if raw is not None else None
        metric.record(repr(actual) if actual is not None else None, repr(expected))
    return metric


def main() -> int:
    report = {
        "conll2003_ner": evaluate_conll().as_dict(),
        "gsm8k_math": evaluate_gsm8k().as_dict(),
        "sst2_sentiment": evaluate_sst2().as_dict(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
