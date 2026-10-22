"""mini-swe-agent adapter: run its control loop on the host, tools in Docker."""

import json
import os
import subprocess
import sys

from .base import AgentError, AgentRunRequest
from ..environments.types import CommandTimeout, ContainerUnavailable, EnvironmentError


class MiniSweAgent:
    def run(self, request: AgentRunRequest) -> dict:
        metadata = request.log.with_suffix(".json")
        command = [sys.executable, "-m", "agentic_harness.agents.mini_runner", "--container", request.container,
                   "--image", request.image, "--task", request.task.prompt, "--output", str(metadata),
                   "--timeout", str(request.timeout)]
        environment = os.environ.copy()
        if request.model_name is not None:
            environment["MSWEA_MODEL_NAME"] = request.model_name
        with request.log.open("w") as output:
            try:
                completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, timeout=request.timeout + 5,
                                           env=environment, check=False)
            except subprocess.TimeoutExpired as exc:
                raise CommandTimeout(f"Agent timed out after {request.timeout}s") from exc
        if completed.returncode:
            try:
                failure = json.loads(metadata.read_text())
            except (OSError, json.JSONDecodeError):
                failure = {}
            if isinstance(failure, dict) and isinstance(failure.get("infrastructure_error"), str):
                raise ContainerUnavailable(failure["infrastructure_error"])
            raise EnvironmentError(f"mini-swe-agent exited {completed.returncode}; see {request.log}")
        try:
            result = json.loads(metadata.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise EnvironmentError(f"mini-swe-agent returned no valid status: {exc}") from exc
        if not isinstance(result, dict) or not isinstance(result.get("exit_status"), str):
            raise EnvironmentError("mini-swe-agent returned invalid status (missing exit_status)")
        return result


def resolve_agent(path: str):
    """Backward-compatible import path for older callers."""
    from .registry import resolve_agent as resolve

    return resolve(path)
