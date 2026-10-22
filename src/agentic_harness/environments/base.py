"""Operations an execution environment provides to the task executor."""

from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from .types import CommandResult


@runtime_checkable
class BaseEnvironment(Protocol):
    image: str
    workspace: Path
    container: str | None

    def __enter__(self) -> Self:
        ...

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: TracebackType | None) -> None:
        ...

    def exec(self, args: list[str], timeout: int, *, cwd: str = "/workspace") -> CommandResult:
        ...

    def copy_benchmark(self, directory: Path) -> None:
        ...

    def remove_benchmark(self) -> None:
        ...
