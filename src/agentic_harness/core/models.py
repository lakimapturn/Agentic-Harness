"""Serializable values passed between the CLI, executor, and future workers."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    image: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskJob:
    run_id: str
    benchmark: str
    task_id: str
    agent: str
    benchmark_root: str
    database: str
    logs_root: str
    attempt_id: str | None = None
    model_name: str | None = None
    docker_memory: str | None = None
    docker_cpus: float | None = None
    docker_pids_limit: int | None = None
    job_timeout: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskResult:
    run_id: str
    task_id: str
    status: str = "pending"
    passed: bool = False
    score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration: float = 0.0
    logs: dict[str, str] = field(default_factory=dict)
    error_class: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
