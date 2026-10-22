"""Generic JSON verifier with configurable exit code conventions."""

import json

from ..environments.types import CommandResult, EnvironmentError


def parse_verification(result: CommandResult, task_id: str, allowed_exit_codes: list[int],
                       *, parser: str = "json", fields: dict | None = None) -> dict:
    if result.returncode not in allowed_exit_codes:
        raise EnvironmentError(
            f"Verifier exited {result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
        )
    if parser == "json":
        payload = result.stdout
    elif parser == "json_last_line":
        lines = result.stdout.strip().splitlines()
        payload = lines[-1] if lines else ""
    else:
        raise EnvironmentError(f"Unknown verifier result parser: {parser}")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise EnvironmentError(f"Verifier returned invalid JSON: {exc}") from exc
    fields = fields or {"id": "task_id", "passed": "passed", "score": "score", "details": "details"}
    if not isinstance(value, dict) or value.get(fields.get("id", "task_id")) != task_id:
        raise EnvironmentError("Verifier returned the wrong task ID or result shape")
    passed = value.get(fields.get("passed", "passed"))
    score = value.get(fields.get("score", "score"))
    if not isinstance(passed, bool) or isinstance(score, bool) or not isinstance(score, (int, float)):
        raise EnvironmentError("Verifier result needs boolean passed and numeric score")
    if not 0 <= float(score) <= 1:
        raise EnvironmentError("Verifier score must be between 0 and 1")
    return {"passed": passed, "score": float(score),
            "details": value.get(fields.get("details", "details"), {})}
