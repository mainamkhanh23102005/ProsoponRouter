"""Free local validation for model-provided Python code."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class CodeValidationResult:
    ok: bool
    code: str
    error: str | None = None


def validate_code_answer(task: dict[str, Any], answer: str) -> CodeValidationResult:
    code = extract_code(answer)
    if not code:
        return CodeValidationResult(False, code, "empty code answer")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        solution_path = temp_path / "solution.py"
        solution_path.write_text(code, encoding="utf-8")

        compile_result = run_python(["-m", "py_compile", str(solution_path)], cwd=temp_path)
        if compile_result.returncode != 0:
            return CodeValidationResult(False, code, combined_output(compile_result))

        tests = list(iter_tests(task))
        for index, test in enumerate(tests):
            result = run_test_case(temp_path, test, index)
            if result.returncode != 0:
                return CodeValidationResult(False, code, combined_output(result))

    return CodeValidationResult(True, code)


def extract_code(answer: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", answer, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return answer.strip()


def iter_tests(task: dict[str, Any]):
    for key in ("tests", "examples"):
        value = task.get(key)
        if isinstance(value, list):
            yield from (item for item in value if isinstance(item, dict))
    if "expected_output" in task:
        yield {"input": task.get("input", ""), "expected_output": task["expected_output"]}


def run_test_case(temp_path: Path, test: dict[str, Any], index: int) -> subprocess.CompletedProcess[str]:
    if "call" in test:
        runner_path = temp_path / f"runner_{index}.py"
        runner_path.write_text(build_call_runner(str(test["call"]), test.get("expected")), encoding="utf-8")
        return run_python([str(runner_path)], cwd=temp_path)

    expected_output = test.get("expected_output")
    if expected_output is not None:
        input_text = str(test.get("input", ""))
        result = run_python(["solution.py"], cwd=temp_path, input_text=input_text)
        if result.returncode != 0:
            return result
        if result.stdout.strip() != str(expected_output).strip():
            return subprocess.CompletedProcess(
                result.args,
                1,
                result.stdout,
                f"expected stdout {expected_output!r}, got {result.stdout.strip()!r}",
            )
        return result

    return subprocess.CompletedProcess([sys.executable], 0, "", "")


def build_call_runner(call: str, expected: Any) -> str:
    expected_json = json.dumps(expected)
    return (
        "import json\n"
        "import solution\n"
        "ns = vars(solution).copy()\n"
        f"result = eval({call!r}, ns)\n"
        f"expected = json.loads({expected_json!r})\n"
        "if result != expected:\n"
        "    raise SystemExit(f'expected {expected!r}, got {result!r}')\n"
    )


def run_python(args: list[str], cwd: Path, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [sys.executable, *args],
            cwd=str(cwd),
            input=input_text,
            text=True,
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(exc.cmd, 124, exc.stdout or "", "code validation timed out")


def combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout or f"process exited {result.returncode}").strip()
