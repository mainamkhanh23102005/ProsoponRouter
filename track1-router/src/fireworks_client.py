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
            return FireworksResult(answer=dry_run_answer(task, category), prompt_tokens=0, completion_tokens=0)

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
        return f"Label sentiment as positive, negative, or neutral. Output one label only.\n{text}"
    if category == "math":
        return f"Answer with only the final value.\n{text}"
    if category == "ner":
        return f"Extract named entities as compact JSON only.\n{text}"
    if category == "summarization":
        return f"Write a concise answer only, no preface.\n{text}"
    if category.startswith("code"):
        return f"Output only the answer or code requested. No explanation.\n{text}"
    return f"Answer only, no explanation.\n{text}"


def dry_run_answer(task: dict[str, Any], category: str) -> Any:
    if category == "sentiment":
        return "neutral"
    if category == "ner":
        return []
    if category == "math":
        return "0"
    if category == "logical reasoning":
        return "yes"
    return "DRY_RUN"
