"""Discover agent adapters by stable CLI identifier."""

from importlib.metadata import entry_points
from inspect import signature
from pathlib import Path
from typing import Callable

from .base import AgentAdapter, AgentError
from .mini_swe import MiniSweAgent


_BUILTINS: dict[str, Callable[[], AgentAdapter]] = {
    "agents/mini-swe-agent": MiniSweAgent,
}


def resolve_agent(identifier: str) -> AgentAdapter:
    """Instantiate a built-in or installed third-party agent adapter."""
    factory = _BUILTINS.get(identifier)
    if factory is None:
        # The former resolver also accepted relative and absolute path spellings.
        path = Path(identifier)
        if path.name == "mini-swe-agent" and path.parent.name == "agents":
            factory = MiniSweAgent
    if factory is None:
        matches = [entry for entry in entry_points(group="agentic_harness.agents")
                   if entry.name == identifier]
        if len(matches) != 1:
            raise AgentError(f"Agent not found or ambiguous: {identifier}")
        try:
            factory = matches[0].load()
        except (ImportError, AttributeError, ValueError) as exc:
            raise AgentError(f"Cannot load agent {identifier}: {exc}") from exc

    try:
        adapter = factory()
    except Exception as exc:
        raise AgentError(f"Cannot initialize agent {identifier}: {exc}") from exc

    if not isinstance(adapter, AgentAdapter) or not callable(getattr(adapter, "run", None)):
        raise AgentError(f"Invalid adapter {identifier}: expected run(AgentRunRequest)")
    try:
        signature(adapter.run).bind(object())
    except (TypeError, ValueError) as exc:
        raise AgentError(f"Invalid adapter {identifier}: expected run(AgentRunRequest)") from exc
    return adapter
