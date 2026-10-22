"""Run mini-swe-agent in a child process so the harness can enforce a wall timeout."""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from ..environments.docker import docker_exec
from ..environments.types import ContainerUnavailable


class ExistingContainerEnvironment:
    """mini-swe-agent Environment API backed by the task's running container."""

    def __init__(self, container: str, image: str):
        self.container = container
        self.image = image

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        command = action.get("command", "")
        try:
            result = docker_exec("docker", self.container, ["sh", "-c", command],
                                 timeout or 30, cwd=cwd or "/workspace")
            output = {"output": result.stdout + result.stderr, "returncode": result.returncode, "exception_info": ""}
        except ContainerUnavailable:
            raise
        except Exception as exc:
            output = {"output": "", "returncode": -1, "exception_info": str(exc)}
        lines = output["output"].lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            from minisweagent.exceptions import Submitted

            raise Submitted({"role": "exit", "content": "".join(lines[1:]),
                             "extra": {"exit_status": "Submitted", "submission": "".join(lines[1:])}})
        return output

    def get_template_vars(self, **kwargs):
        return {"system": "Linux", "release": "container", "version": self.image,
                "machine": platform.machine(), "cwd": "/workspace", **kwargs}

    def serialize(self):
        return {"info": {"config": {"environment": {"image": self.image, "cwd": "/workspace"},
                                    "environment_type": "agentic_harness.agents.mini_runner.ExistingContainerEnvironment"}}}


def model_config_for(config: dict) -> dict:
    """Use OpenRouter's own cost data for OpenRouter model names."""
    model_config = dict(config.get("model", {}))
    model_name = model_config.get("model_name") or os.getenv("MSWEA_MODEL_NAME", "")
    if model_name.startswith("openrouter/") and not model_config.get("model_class"):
        model_config["model_name"] = model_name.removeprefix("openrouter/")
        model_config["model_class"] = "openrouter"
        model_kwargs = dict(model_config.get("model_kwargs", {}))
        model_kwargs.pop("drop_params", None)  # LiteLLM-only option.
        model_config["model_kwargs"] = model_kwargs
    return model_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    args = parser.parse_args()
    try:
        from minisweagent.agents import get_agent
        from minisweagent.config import builtin_config_dir, get_config_from_spec
        from minisweagent.models import get_model
    except ImportError as exc:
        print(f"mini-swe-agent is unavailable: {exc}. Install with pip install -e '.[mini]'", file=sys.stderr)
        return 2
    config = get_config_from_spec(str(builtin_config_dir / "mini.yaml"))
    agent_config = dict(config["agent"])
    agent_config.update({
        "agent_class": "default", "wall_time_limit_seconds": args.timeout,
        "output_path": str(Path(args.output).with_suffix(".trajectory.json")),
        "instance_template": (
            "Solve this task in /workspace: {{task}}\n"
            "Commands execute inside the task Docker container using POSIX sh. "
            "Keep all work in /workspace. Inspect existing files before editing. "
            "Modify the provided solution files instead of inventing a new filename, "
            "and test your changes before submitting. When finished, execute "
            "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as a command by itself."
        ),
    })
    agent = get_agent(get_model(config=model_config_for(config)),
                      ExistingContainerEnvironment(args.container, args.image), agent_config)
    try:
        result = agent.run(args.task)
    except ContainerUnavailable as exc:
        # The parent adapter reads this small typed report from outside the sandbox.
        Path(args.output).write_text(json.dumps({"infrastructure_error": str(exc)}))
        return 1
    Path(args.output).write_text(json.dumps(result))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
