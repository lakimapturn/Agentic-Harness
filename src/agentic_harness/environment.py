"""Compatibility imports for the environment contract and Docker provider."""

from .environments.base import BaseEnvironment
from .environments.docker import DockerEnvironment, cleanup_attempt_containers, run_process
from .environments.types import CommandResult, CommandTimeout, ContainerUnavailable, EnvironmentError

__all__ = [
    "BaseEnvironment", "CommandResult", "CommandTimeout", "ContainerUnavailable",
    "DockerEnvironment", "EnvironmentError",
    "cleanup_attempt_containers", "run_process",
]
