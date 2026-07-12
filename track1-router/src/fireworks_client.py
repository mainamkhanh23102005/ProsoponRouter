"""Minimal OpenAI-compatible Fireworks client."""

from __future__ import annotations

import os
import re
import sys
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

    def complete(self, task: dict[str, Any], category: str, retry_feedback: str | None = None) -> FireworksResult:
        policy = config.POLICY.get(category, config.POLICY["unknown"])
        prompt = build_retry_prompt(task, category, retry_feedback) if retry_feedback else build_prompt(task, category)

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

        payload = build_payload(policy, prompt)

        url = config.FIREWORKS_BASE_URL.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        last_error = None
        prompt_tokens = 0
        completion_tokens = 0
        for attempt in range(2):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=config.HTTP_TIMEOUT_SECONDS)
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                content, content_error = extract_message_content(message)
                usage = data.get("usage", {})
                prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                if content_error:
                    last_error = content_error
                    print(content_error, file=sys.stderr)
                    if attempt == 0:
                        time.sleep(0.5)
                        continue
                    result = FireworksResult(
                        answer="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        error=content_error,
                    )
                    self.total_tokens += result.total_tokens
                    return result
                result = FireworksResult(
                    answer=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                self.total_tokens += result.total_tokens
                return result
            except Exception as exc:  # noqa: BLE001 - final fallback must not crash the run.
                last_error = sanitize_error(str(exc))
                if attempt == 0:
                    time.sleep(0.5)

        return FireworksResult(
            answer=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=last_error,
        )


def extract_message_content(message: dict[str, Any]) -> tuple[str, str | None]:
    content = message.get("content")
    if not str(content or "").strip():
        content = message.get("reasoning_content")
    if not str(content or "").strip():
        keys = list(message.keys())
        return "", f"empty content, message keys were: {keys}"
    return str(content).strip(), None


def sanitize_error(error: str) -> str:
    sanitized = error
    api_key = os.getenv(config.FIREWORKS_API_KEY_ENV)
    if api_key:
        sanitized = sanitized.replace(api_key, "[REDACTED_API_KEY]")
    sanitized = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
        r"\1[REDACTED_API_KEY]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+",
        r"\1[REDACTED_API_KEY]",
        sanitized,
    )
    return sanitized


def normalize_model_id(model: str) -> str:
    if model.startswith("accounts/"):
        return model
    return f"accounts/fireworks/models/{model}"


def build_payload(policy: config.CategoryPolicy, prompt: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": normalize_model_id(policy.model),
        "temperature": 0,
        "max_tokens": policy.max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if policy.reasoning_mode == "reasoning_none":
        payload["reasoning_effort"] = "none"
    else:
        payload["thinking"] = {"type": "disabled"}
    if policy.stop:
        payload["stop"] = list(policy.stop)
    return payload


def build_prompt(task: dict[str, Any], category: str) -> str:
    text = task_text(task).strip()
    if category == "sentiment":
        return f"Answer with sentiment label and one-sentence justification only: positive|negative|neutral: reason\n{text}"
    if category == "math":
        return f"Compute one operation per line, no prose. Last line: Answer: include every requested result.\n{text}"
    if category == "ner":
        return (
            "Extract every PERSON/ORGANIZATION/LOCATION/DATE in order. "
            "Keep compound organization names intact (e.g. ETH Zurich, not only ETH or Zurich). "
            "Before finalizing, enumerate every named organization, person, location, and date; omit none. "
            "Output only 'PERSON: name; ORGANIZATION: name; LOCATION: name; DATE: text' or NONE.\n"
            f"{text}"
        )
    if category == "summarization":
        return f"Follow the prompt's length/format instruction exactly. Output only the summary, no preamble.\n{text}"
    if category == "code debugging":
        return (
            "Return corrected Python code, then '# SELF_CHECK:' and 2-3 assert statements. "
            "No markdown, no explanation.\n"
            f"{text}"
        )
    if category == "code generation":
        return (
            "Return Python code, then '# SELF_CHECK:' and 2-3 assert statements. "
            "No markdown, no explanation.\n"
            f"{text}"
        )
    if category == "logical reasoning":
        return f"Use telegraphic deduction notes, no prose. Last line: Answer:\n{text}"
    if category == "factual knowledge":
        return (
            "Answer every part in at most three concise sentences. For comparisons include relationship, "
            "key properties or mechanism, and uses. No markdown.\n"
            f"{text}"
        )
    return f"Answer only.\n{text}"


def build_retry_prompt(task: dict[str, Any], category: str, feedback: str | None = None) -> str:
    expected = expected_format(category)
    text = task_text(task).strip()
    feedback_line = f"\nPrevious failure: {feedback}" if feedback else ""
    return f"Previous answer invalid. Reply with ONLY {expected}. Nothing else.{feedback_line}\n{text}"


def expected_format(category: str) -> str:
    if category == "code debugging":
        return "corrected Python code followed by # SELF_CHECK: assert statements"
    if category == "code generation":
        return "Python code followed by # SELF_CHECK: assert statements"
    if category == "sentiment":
        return "<positive|negative|neutral>: one sentence reason"
    if category == "ner":
        return "LABEL: text; ... or NONE"
    if category == "math":
        return "the final value"
    return "the final answer"


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
        return "def add(a, b):\n    return a + b\n\n# SELF_CHECK:\nassert add(2, 3) == 5\nassert add(-1, 1) == 0"
    if category == "code generation":
        return "def larger(a, b):\n    return a if a >= b else b\n\n# SELF_CHECK:\nassert larger(2, 3) == 3\nassert larger(5, 1) == 5"
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
