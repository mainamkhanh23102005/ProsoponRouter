"""Minimal OpenAI-compatible Fireworks client."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from src import config
from src.task_utils import task_text


@dataclass
class FireworksResult:
    answer: Any
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class FireworksClient:
    def __init__(self) -> None:
        self.total_tokens = 0

    def complete(self, task: dict[str, Any], category: str) -> FireworksResult:
        policy = config.POLICY.get(category, config.POLICY["unknown"])
        prompt = build_prompt(task, category)

        if config.DRY_RUN:
            return dry_run_result(task, category)

        api_key = os.getenv(config.FIREWORKS_API_KEY_ENV)
        if not api_key:
            return FireworksResult(answer=None, error=f"missing {config.FIREWORKS_API_KEY_ENV}")
        if policy.model.startswith("UNKNOWN"):
            return FireworksResult(answer=None, error="CHEAPEST_MODEL/ALLOWED_MODELS not configured")
        if policy.model not in config.ALLOWED_MODELS:
            return FireworksResult(answer=None, error=f"model {policy.model} not in ALLOWED_MODELS")

        try:
            import requests
        except ImportError:
            return FireworksResult(answer=None, error="requests is not installed")

        payload: dict[str, Any] = {
            "model": policy.model,
            "temperature": 0,
            "max_tokens": policy.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if policy.stop:
            payload["stop"] = list(policy.stop)

        url = config.FIREWORKS_BASE_URL.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        last_error = None
        for attempt in range(2):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=config.HTTP_TIMEOUT_SECONDS)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                result = FireworksResult(
                    answer=content,
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                )
                self.total_tokens += result.total_tokens
                return result
            except Exception as exc:  # noqa: BLE001 - final fallback must not crash the run.
                last_error = str(exc)
                if attempt == 0:
                    time.sleep(0.5)

        return FireworksResult(answer=None, error=last_error)


def build_prompt(task: dict[str, Any], category: str) -> str:
    text = task_text(task).strip()
    if category == "sentiment":
        return f"Answer with sentiment label and one-sentence justification only: positive|negative|neutral: reason\n{text}"
    if category == "math":
        return f"Only final value.\n{text}"
    if category == "ner":
        return f"Extract PERSON/ORG/LOCATION/DATE. Output 'LABEL: text; ...' or NONE only.\n{text}"
    if category == "summarization":
        return f"Summarize in <=2 sentences. Output only summary.\n{text}"
    if category == "code debugging":
        return f"Return corrected Python code only. No markdown, no explanation.\n{text}"
    if category == "code generation":
        return f"Return Python code only. No markdown, no explanation.\n{text}"
    if category == "logical reasoning":
        return f"Answer yes/no/unknown only.\n{text}"
    if category == "factual knowledge":
        return f"Answer only the fact, no explanation.\n{text}"
    return f"Answer only.\n{text}"


def dry_run_result(task: dict[str, Any], category: str) -> FireworksResult:
    if config.DRY_RUN_MODE == "error":
        return FireworksResult(answer=None, error="dry-run simulated API error")
    if config.DRY_RUN_MODE == "invalid":
        return FireworksResult(answer=dry_run_invalid_answer(category), prompt_tokens=0, completion_tokens=0)
    return FireworksResult(answer=dry_run_success_answer(task, category), prompt_tokens=0, completion_tokens=0)


def dry_run_success_answer(task: dict[str, Any], category: str) -> Any:
    if category == "sentiment":
        return "neutral: dry-run sentiment placeholder"
    if category == "ner":
        return "PERSON: Sarah Johnson; ORG: Microsoft; LOCATION: Seattle; DATE: 2026-07-11"
    if category == "math":
        return "0"
    if category == "summarization":
        return "AMD hackathon teams build token-efficient AI apps."
    if category == "code debugging":
        return "def add(a, b):\n    return a + b"
    if category == "code generation":
        return "def larger(a, b):\n    return a if a >= b else b"
    if category == "logical reasoning":
        return "yes"
    if category == "factual knowledge":
        return "Hanoi"
    return "DRY_RUN"


def dry_run_invalid_answer(category: str) -> Any:
    if category in {"sentiment", "math", "summarization", "factual knowledge", "code debugging", "code generation", "logical reasoning"}:
        return ""
    if category == "ner":
        return []
    return ""
