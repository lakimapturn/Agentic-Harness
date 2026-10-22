"""Serializable job in, persisted result out. No CLI or process-global task state."""

import json
import hashlib
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, cast

from ..agents import AgentError, AgentRunRequest, resolve_agent
from ..benchmarks.config import resolve_benchmark
from ..environments.base import BaseEnvironment
from ..environments.docker import DockerEnvironment
from ..environments.types import CommandResult, CommandTimeout, ContainerUnavailable, EnvironmentError
from ..core.models import Task, TaskJob, TaskResult
from ..persistence.sqlite import RunStore
from ..verification.json_result import parse_verification
from .submission import _copy_submission


def _record(logs: Path, stage: str, value: CommandResult):
    path = logs / f"{stage}.log"
    path.write_text(f"$ {' '.join(value.args)}\nexit={value.returncode}\n{value.stdout}\n{value.stderr}")
    return str(path)


def _command(env: BaseEnvironment, benchmark, task, stage: str, logs: Path, workspace: Path,
             *, log_name: str | None = None) -> CommandResult:
    spec = benchmark.config[stage]
    runtime = spec.get("runtime", "container")
    cwd = "/opt/harness" if runtime == "container" else str(benchmark.directory)
    args = benchmark.command(stage, task, "/workspace" if runtime == "container" else str(workspace))
    if runtime == "container":
        value = env.exec(args, benchmark.timeout(stage), cwd=cwd)
    elif runtime == "host":
        from ..environments.docker import run_process
        # The host mode is for trusted setup scripts that need no container Python.
        if args[0] == "python":
            args[0] = sys.executable
        value = run_process(args, benchmark.timeout(stage), cwd=benchmark.directory)
    else:
        raise EnvironmentError(f"Unknown {stage} runtime: {runtime}")
    _record(logs, log_name or stage, value)
    return value


def _log_component(value: str) -> str:
    """Keep dataset IDs readable when safe, and contain arbitrary IDs in one directory."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        return value
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _environment(factory: Callable[..., BaseEnvironment], image: str, workspace: Path,
                 job: TaskJob) -> BaseEnvironment:
    # Docker alone accepts these resource and attempt options; test fakes use the basic constructor.
    if factory is DockerEnvironment:
        return factory(image, workspace, memory=job.docker_memory, cpus=job.docker_cpus,
                       pids_limit=job.docker_pids_limit, attempt_id=job.attempt_id)
    return factory(image, workspace)


def _stage(store: RunStore, job: TaskJob, stage: str):
    if not store.set_stage(job, stage):
        raise EnvironmentError(f"Attempt is no longer active: {job.attempt_id}")


def execute_task(job: TaskJob, *, environment_factory: Callable[..., BaseEnvironment] = DockerEnvironment,
                 agent_factory=resolve_agent) -> TaskResult:
    """Execute one task. Factories make deterministic agent/container tests possible."""
    store = RunStore(job.database)
    result = TaskResult(run_id=job.run_id, task_id=job.task_id)
    started = time.monotonic()
    stage = "resolve"
    logs = Path(job.logs_root).resolve() / _log_component(job.run_id) / _log_component(job.task_id)
    if job.attempt_id:
        logs = logs / _log_component(job.attempt_id)
    logs.mkdir(parents=True, exist_ok=True)
    workspaces = Path(job.database).resolve().parent / "workspaces"
    workspaces.mkdir(parents=True, exist_ok=True)
    try:
        _stage(store, job, "resolve")
        benchmark = resolve_benchmark(job.benchmark, Path(job.benchmark_root))
        task = benchmark.get_task(job.task_id)
        agent = agent_factory(job.agent)
        with TemporaryDirectory(prefix=f"harness-agent-{(job.attempt_id or '')[:12]}-",
                                dir=workspaces) as agent_dir:
            workspace = Path(agent_dir)
            stage = "environment"
            _stage(store, job, stage)
            with _environment(environment_factory, task.image, workspace, job) as env:
                stage = "setup"
                _stage(store, job, stage)
                if benchmark.config["setup"].get("runtime", "container") == "container":
                    env.copy_benchmark(benchmark.directory)
                setup = _command(env, benchmark, task, "setup", logs, workspace)
                result.logs["setup"] = str(logs / "setup.log")
                if setup.returncode:
                    raise EnvironmentError(f"Setup exited {setup.returncode}; see {result.logs['setup']}")
                if benchmark.config["setup"].get("runtime", "container") == "container":
                    env.remove_benchmark()
                for hidden in benchmark.config.get("hide_from_agent", []):
                    relative = Path(hidden)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise EnvironmentError(f"Invalid hidden path: {hidden}")
                    hidden_path = workspace / relative
                    if hidden_path.is_symlink() or hidden_path.is_file():
                        hidden_path.unlink()
                    elif hidden_path.is_dir():
                        shutil.rmtree(hidden_path)
                stage = "agent"
                _stage(store, job, stage)
                agent_error = None
                agent_timed_out = False
                agent_log = logs / "agent.log"
                result.logs["agent"] = str(agent_log)
                try:
                    public_task = Task(id=task.id, prompt=task.prompt, image=task.image)
                    # An active Docker context has a container; before entry it can be None.
                    request = AgentRunRequest(task=public_task, container=cast(str, env.container),
                                              image=task.image, timeout=benchmark.timeout("agent"),
                                              log=agent_log, model_name=job.model_name)
                    agent_status = agent.run(request)
                    if (not isinstance(agent_status, dict)
                            or not isinstance(agent_status.get("exit_status"), str)):
                        raise AgentError("Agent adapter must return a status with exit_status")
                    (logs / "agent_status.json").write_text(json.dumps(agent_status, indent=2))
                    if agent_status.get("exit_status") != "Submitted":
                        agent_error = f"Agent stopped: {agent_status.get('exit_status', 'unknown')}"
                except CommandTimeout as exc:
                    agent_error = str(exc)
                    agent_timed_out = True
                    agent_log.touch(exist_ok=True)
                except ContainerUnavailable:
                    raise
                except Exception as exc:
                    agent_error = str(exc)
                    agent_log.touch(exist_ok=True)
                agent_trajectory = agent_log.with_suffix(".trajectory.json")
                if agent_trajectory.exists():
                    result.logs["agent_trajectory"] = str(agent_trajectory)
                if agent_error:
                    agent_error_log = logs / "agent_error.log"
                    agent_error_log.write_text(agent_error)
                    result.logs["agent_error"] = str(agent_error_log)
            # Closing the task container stops any agent process before workspace transfer.
            stage = "verifier"
            _stage(store, job, stage)
            with TemporaryDirectory(prefix=f"harness-grade-{(job.attempt_id or '')[:12]}-",
                                    dir=workspaces) as grade_dir:
                grade_workspace = Path(grade_dir)
                image = benchmark.config["verifier"].get("image", task.image)
                with _environment(environment_factory, image, grade_workspace, job) as grade_env:
                    grade_env.copy_benchmark(benchmark.directory)
                    # Prepare trusted test files and dependencies in the grader container.
                    if benchmark.config["setup"].get("runtime", "container") == "container":
                        grade_setup = _command(grade_env, benchmark, task, "setup", logs, grade_workspace,
                                               log_name="grader_setup")
                        result.logs["grader_setup"] = str(logs / "grader_setup.log")
                        if grade_setup.returncode:
                            raise EnvironmentError(f"Grader setup exited {grade_setup.returncode}")
                    submission_values = {key: value for key, value in task.data.items()
                                         if isinstance(value, (str, int, float))}
                    submission_values["task_id"] = task.id
                    submission_paths = [path.format_map(submission_values)
                                        for path in benchmark.config.get("submission_paths", ["."])]
                    _copy_submission(workspace, grade_workspace, submission_paths,
                                     benchmark.config.get("protected_paths", []) +
                                     benchmark.config.get("hide_from_agent", []))
                    verification = _command(grade_env, benchmark, task, "verifier", logs, grade_workspace)
                    result.logs["verifier"] = str(logs / "verifier.log")
                    scored = parse_verification(verification, task.id,
                                                benchmark.config["verifier"].get("allowed_exit_codes", [0]),
                                                parser=benchmark.config["verifier"].get("result", "json"),
                                                fields=benchmark.config["verifier"].get("fields"))
                    result.passed = scored["passed"]
                    result.score = scored["score"]
                    result.details = scored["details"]
            result.status = "timeout" if agent_timed_out else (
                "agent_error" if agent_error else ("passed" if result.passed else "failed"))
            result.error = agent_error
            if agent_error:
                result.error_class = "agent_timeout" if agent_timed_out else "agent"
    except CommandTimeout as exc:
        result.status = "timeout"
        result.error_class = f"{stage}_timeout"
        result.error = str(exc)
        result.logs["error"] = str(logs / "error.log")
        (logs / "error.log").write_text(traceback.format_exc())
    except Exception as exc:
        result.status = "error"
        result.error_class = "environment" if isinstance(exc, ContainerUnavailable) else stage
        result.error = f"{type(exc).__name__}: {exc}"
        result.logs["error"] = str(logs / "error.log")
        (logs / "error.log").write_text(traceback.format_exc())
    finally:
        result.duration = round(time.monotonic() - started, 3)
        store.save_result(result, job.attempt_id)
    return result
