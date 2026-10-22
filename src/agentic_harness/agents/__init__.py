"""Agent adapter contract, resolver, and built-in integrations."""

from .base import AgentAdapter, AgentError, AgentRunRequest
from .mini_swe import MiniSweAgent
from .registry import resolve_agent

__all__ = ["AgentAdapter", "AgentError", "AgentRunRequest", "MiniSweAgent", "resolve_agent"]
