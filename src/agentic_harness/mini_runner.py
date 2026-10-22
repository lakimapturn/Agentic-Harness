"""Legacy module path for mini-swe-agent's subprocess runner."""

from .agents.mini_runner import ExistingContainerEnvironment, main, model_config_for

__all__ = ["ExistingContainerEnvironment", "main", "model_config_for"]

if __name__ == "__main__":
    raise SystemExit(main())
