"""Compatibility imports; implementation: persistence/sqlite.py."""

from .persistence.sqlite import TERMINAL, RunStore

# Historical status set for imports from Part 1/early Part 2. Retry decisions
# now use the error-class allowlist in persistence.sqlite.
RETRYABLE = {"error", "timeout"}

__all__ = ["RETRYABLE", "TERMINAL", "RunStore"]
