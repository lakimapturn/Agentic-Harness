"""Execution results and errors shared by environment implementations."""

from dataclasses import dataclass


class EnvironmentError(RuntimeError):
    pass


class CommandTimeout(EnvironmentError):
    pass


class ContainerUnavailable(EnvironmentError):
    """The execution container is unavailable."""


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str
