"""Free local validation for model-provided Python code."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.task_utils import task_text


TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class CodeValidationResult:
    ok: bool
    code: str
    error: str | None = None


@dataclass(frozen=True)
class ParsedCodeAnswer:
    code: str
    self_checks: tuple[str, ...] = ()


def validate_code_answer(
    task: dict[str, Any],
    answer: str,
    *,
    require_task_tests: bool = False,
) -> CodeValidationResult:
    parsed = parse_code_answer(answer)
    code = parsed.code
    if not code:
        return CodeValidationResult(False, code, "empty code answer")

    expected_names = expected_function_names(task)
    if expected_names:
        generated_names, parse_error = top_level_function_names(code)
        if parse_error:
            return CodeValidationResult(False, code, parse_error)
        if not generated_names & expected_names:
            expected = ", ".join(sorted(expected_names))
            actual = ", ".join(sorted(generated_names)) or "none"
            return CodeValidationResult(
                False,
                code,
                f"function signature mismatch: expected {expected}; generated {actual}",
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        solution_path = temp_path / "solution.py"
        solution_path.write_text(code, encoding="utf-8")

        compile_result = run_python(["-m", "py_compile", str(solution_path)], cwd=temp_path)
        if compile_result.returncode != 0:
            return CodeValidationResult(False, code, combined_output(compile_result))

        tests = list(iter_tests(task))
        if require_task_tests and not tests:
            return CodeValidationResult(False, code, "local code requires task-owned semantic tests")
        if tests:
            for index, test in enumerate(tests):
                result = run_test_case(temp_path, test, index)
                if result.returncode != 0:
                    return CodeValidationResult(False, code, combined_output(result))
        elif parsed.self_checks:
            solution_path.write_text(code + "\n\n" + "\n".join(parsed.self_checks) + "\n", encoding="utf-8")
            result = run_python(["solution.py"], cwd=temp_path)
            if result.returncode != 0:
                return CodeValidationResult(False, code, combined_output(result))

    return CodeValidationResult(True, code)


def extract_code(answer: str) -> str:
    return parse_code_answer(answer).code


def parse_code_answer(answer: str) -> ParsedCodeAnswer:
    text = strip_code_fences(answer)
    marker = re.search(r"(?im)^#\s*SELF_CHECK\s*:\s*$", text)
    if not marker:
        return ParsedCodeAnswer(text.strip())

    code = text[: marker.start()].strip()
    raw_checks = text[marker.end() :].strip()
    checks = tuple(line.strip() for line in raw_checks.splitlines() if line.strip().startswith("assert "))
    return ParsedCodeAnswer(code, checks)


def strip_code_fences(answer: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", answer, flags=re.IGNORECASE | re.DOTALL)
    if match:
        answer = answer[: match.start()] + match.group(1) + answer[match.end() :]
    return answer.replace("```", "").strip()


def expected_function_names(task: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for test in iter_tests(task):
        call = test.get("call")
        if isinstance(call, str):
            name = function_name_from_call(call)
            if name:
                names.add(name)

    text = task_text(task)
    names.update(function_names_from_prompt(text))
    return names


def function_name_from_call(call: str) -> str | None:
    try:
        node = ast.parse(call, mode="eval").body
    except SyntaxError:
        return None
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
    return None


def function_names_from_prompt(text: str) -> set[str]:
    names: set[str] = set()
    patterns = (
        r"\bdef\s+([A-Za-z_]\w*)\s*\(",
        r"`([A-Za-z_]\w*)\s*\([^`]*\)`",
        r"\bfunction\s+([A-Za-z_]\w*)\s*\(",
        r"\bfunction\s+(?:called|named)\s+`?([A-Za-z_]\w*)`?",
        r"\b(?:called|named)\s+`?([A-Za-z_]\w*)`?\s+function\b",
        r"\bfunction\s+that\s+returns\s+the\s+([A-Za-z_]\w*)\s+of\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            name = match.group(1)
            if name and name.isidentifier():
                names.add(name)
    return names


def top_level_function_names(code: str) -> tuple[set[str], str | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return set(), f"SyntaxError: {exc}"
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return names, None


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
