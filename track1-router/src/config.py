"""Central configuration for official facts and routing policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


TASK_INPUT_PATH = os.getenv("TASK_INPUT_PATH", "/input/tasks.json")
RESULTS_OUTPUT_PATH = os.getenv("RESULTS_OUTPUT_PATH", "/output/results.json")

FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
FIREWORKS_API_KEY_ENV = os.getenv("FIREWORKS_API_KEY_ENV", "FIREWORKS_API_KEY")
ALLOWED_MODELS = _csv_env("ALLOWED_MODELS", ("UNKNOWN_MODEL",))
CHEAPEST_MODEL = os.getenv("CHEAPEST_MODEL", ALLOWED_MODELS[0])

DRY_RUN = os.getenv("DRY_RUN", "0").strip().lower() in {"1", "true", "yes", "on"}
DRY_RUN_MODE = os.getenv("DRY_RUN_MODE", "success").strip().lower()
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.95"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "15"))
LATENCY_LIMIT_SECONDS = float(os.getenv("LATENCY_LIMIT_SECONDS", "540"))
LATENCY_RESERVE_SECONDS = float(os.getenv("LATENCY_RESERVE_SECONDS", "1"))

VALID_CATEGORIES = {
    "math",
    "ner",
    "sentiment",
    "summarization",
    "factual knowledge",
    "code debugging",
    "logical reasoning",
    "code generation",
    "unknown",
}


@dataclass(frozen=True)
class CategoryPolicy:
    free_enabled: bool
    model: str
    max_tokens: int
    min_confidence: float = 0.90
    retry_on_invalid: bool = False
    stop: tuple[str, ...] = ()


POLICY: dict[str, CategoryPolicy] = {
    "math": CategoryPolicy(True, CHEAPEST_MODEL, 16, 0.95, stop=("\n",)),
    "ner": CategoryPolicy(True, CHEAPEST_MODEL, 96, 0.90),
    "sentiment": CategoryPolicy(True, CHEAPEST_MODEL, 80, 0.92),
    "summarization": CategoryPolicy(False, CHEAPEST_MODEL, 120),
    "factual knowledge": CategoryPolicy(False, CHEAPEST_MODEL, 64, stop=("\n",)),
    "code debugging": CategoryPolicy(False, CHEAPEST_MODEL, 260, retry_on_invalid=True),
    "logical reasoning": CategoryPolicy(False, CHEAPEST_MODEL, 150),
    "code generation": CategoryPolicy(False, CHEAPEST_MODEL, 420, retry_on_invalid=True),
    "unknown": CategoryPolicy(False, CHEAPEST_MODEL, 48),
}

FALLBACK_ANSWERS: dict[str, Any] = {
    "math": "",
    "ner": "",
    "sentiment": "",
    "summarization": "",
    "factual knowledge": "",
    "code debugging": "",
    "logical reasoning": "",
    "code generation": "",
    "unknown": "",
}
