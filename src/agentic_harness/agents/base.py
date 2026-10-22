"""The execution contract shared by the harness and agent integrations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..core.models import Task


class AgentError(ValueError):
    """An agent identifier or adapter implementation is invalid."""


@dataclass(frozen=True)
class AgentRunRequest:
    """Public task and execution settings; never includes private grading data."""

    task: Task
    container: str
    image: str
    timeout: int
    log: Path
    model_name: str | None = None


@runtime_checkable
class AgentAdapter(Protocol):
    """Common interface implemented by agent integrations."""

    def run(self, request: AgentRunRequest) -> dict[str, Any]:
        ...
