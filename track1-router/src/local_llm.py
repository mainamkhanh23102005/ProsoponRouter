"""Local llama.cpp chat rung used before paid Fireworks fallback.

The Docker image can copy llama-server plus a small Gemma GGUF from the audited
GHCR image. This module treats that server as opportunistic: if it is not ready,
too slow, or returns invalid output, the cascade continues to Fireworks.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from src import config
from src.fireworks_client import (
    FireworksResult,
    build_prompt,
    extract_message_content,
    sanitize_error,
)
from src.task_utils import task_text


_STATE = {"ready": False, "fails": 0, "next_check": 0.0}


def can_attempt(task: dict[str, Any], category: str) -> bool:
    if config.DRY_RUN or not config.LOCAL_LLM_ENABLED:
        return False
    if _STATE["fails"] >= config.LOCAL_LLM_FAILS_TO_DISABLE:
        return False
    text = task_text(task)
    if not text or len(text) > config.LOCAL_LLM_MAX_PROMPT_CHARS:
        return False
    if category not in config.LOCAL_LLM_CATEGORIES:
        return False
    return _healthy()


def complete(task: dict[str, Any], category: str) -> FireworksResult:
    policy = config.POLICY.get(category, config.POLICY["unknown"])
    prompt = build_prompt(task, category)
    payload: dict[str, Any] = {
        "model": config.LOCAL_LLM_MODEL,
        "temperature": 0,
        "max_tokens": policy.max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    url = config.LOCAL_LLM_URL.rstrip("/") + "/chat/completions"
    try:
        response = requests.post(url, json=payload, timeout=config.LOCAL_LLM_TIMEOUT_SECONDS)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        content, content_error = extract_message_content(message)
        usage = response.json().get("usage", {})
        result = FireworksResult(
            answer=content,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            error=content_error,
        )
        _STATE["fails"] = 0 if not content_error else _STATE["fails"] + 1
        return result
    except Exception as exc:  # noqa: BLE001 - local rung must never break fallback.
        _STATE["fails"] += 1
        return FireworksResult(answer=None, error=sanitize_error(str(exc)))


def _healthy() -> bool:
    if _STATE["ready"]:
        return True
    now = time.monotonic()
    if now < _STATE["next_check"]:
        return False
    _STATE["next_check"] = now + config.LOCAL_LLM_HEALTH_RECHECK_SECONDS
    health_url = config.LOCAL_LLM_URL.rsplit("/v1", 1)[0].rstrip("/") + "/health"
    try:
        response = requests.get(health_url, timeout=2)
        _STATE["ready"] = response.status_code == 200
    except Exception:
        _STATE["ready"] = False
    return bool(_STATE["ready"])
