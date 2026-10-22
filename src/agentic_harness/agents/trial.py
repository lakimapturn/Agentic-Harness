"""Deterministic trial adapter that runs a shell prompt in the task container.

This is only for exercising the adapter and benchmark extension points. It does
not use a model or attempt to solve natural-language tasks.
"""

from .base import AgentRunRequest
from ..environments.docker import docker_exec


class TrialAgent:
    def run(self, request: AgentRunRequest) -> dict:
        result = docker_exec(
            "docker",
            request.container,
            ["sh", "-c", request.task.prompt],
            request.timeout,
            cwd="/workspace",
        )
        request.log.write_text(
            f"exit={result.returncode}\n{result.stdout}{result.stderr}"
        )
        return {
            "exit_status": "Submitted" if result.returncode == 0 else "Failed",
            "returncode": result.returncode,
        }
