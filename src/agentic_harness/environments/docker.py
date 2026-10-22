"""One Docker container per task phase, with a bind mounted private workspace."""

import subprocess
import uuid
from pathlib import Path

from .types import CommandResult, CommandTimeout, ContainerUnavailable, EnvironmentError


def run_process(args: list[str], timeout: int, cwd: Path | None = None) -> CommandResult:
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                                errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise CommandTimeout(f"Command timed out after {timeout}s: {args[0]}") from exc
    except OSError as exc:
        raise EnvironmentError(f"Cannot execute {args[0]}: {exc}") from exc
    return CommandResult(args, result.returncode, result.stdout, result.stderr)


def docker_exec(docker: str, container: str, args: list[str], timeout: int,
                *, cwd: str = "/workspace") -> CommandResult:
    """Keep command exit codes as observations; detect a lost Docker sandbox."""
    try:
        result = run_process([docker, "exec", "-w", cwd, container, *args], timeout)
    except CommandTimeout:
        raise
    except EnvironmentError as exc:
        raise ContainerUnavailable(f"Cannot execute in container {container}: {exc}") from exc
    if result.returncode:
        try:
            state = run_process([docker, "inspect", "-f", "{{.State.Running}}", container], 10)
        except EnvironmentError as exc:
            raise ContainerUnavailable(f"Cannot inspect container {container}: {exc}") from exc
        if state.returncode or state.stdout.strip() != "true":
            raise ContainerUnavailable(f"Container {container} is unavailable: "
                                       f"{result.stderr.strip() or state.stderr.strip()}")
    return result


class DockerEnvironment:
    def __init__(self, image: str, workspace: Path, *, docker: str = "docker", startup_timeout: int = 180,
                 memory: str | None = None, cpus: float | None = None, pids_limit: int | None = None,
                 attempt_id: str | None = None):
        self.image = image
        self.workspace = workspace.resolve()
        self.docker = docker
        self.startup_timeout = startup_timeout
        self.container: str | None = None
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.attempt_id = attempt_id

    def __enter__(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        name = f"harness-{uuid.uuid4().hex[:12]}"
        try:
            options = []
            if self.memory:
                options.extend(["--memory", self.memory])
            if self.cpus:
                options.extend(["--cpus", str(self.cpus)])
            if self.pids_limit:
                options.extend(["--pids-limit", str(self.pids_limit)])
            if self.attempt_id:
                options.extend(["--label", f"agentic-harness.attempt={self.attempt_id}"])
            result = run_process([
                self.docker, "run", "-d", "--name", name, "--security-opt", "no-new-privileges",
                "--mount", f"type=bind,source={self.workspace},target=/workspace",
                "-w", "/workspace", *options, self.image,
                "sh", "-c", "while :; do sleep 3600; done",
            ], self.startup_timeout)
        except Exception:
            try:
                run_process([self.docker, "rm", "-f", name], 30)
            except EnvironmentError:
                pass
            raise
        if result.returncode:
            # Docker can create a container before reporting a failed start.
            run_process([self.docker, "rm", "-f", name], 30)
            raise EnvironmentError(f"Docker start failed: {result.stderr.strip() or result.stdout.strip()}")
        self.container = name
        return self

    def __exit__(self, *_):
        if self.container:
            try:
                try:
                    result = run_process([self.docker, "rm", "-f", self.container], 30)
                except EnvironmentError as exc:
                    raise ContainerUnavailable(f"Docker cleanup failed: {exc}") from exc
                if result.returncode:
                    raise ContainerUnavailable(f"Docker cleanup failed: {result.stderr}")
            finally:
                self.container = None

    def exec(self, args: list[str], timeout: int, *, cwd: str = "/workspace") -> CommandResult:
        if not self.container:
            raise ContainerUnavailable("Container is not running")
        return docker_exec(self.docker, self.container, args, timeout, cwd=cwd)

    def copy_benchmark(self, directory: Path):
        if not self.container:
            raise EnvironmentError("Container is not running")
        make = self.exec(["mkdir", "-p", "/opt/harness"], 30)
        if make.returncode:
            raise EnvironmentError(make.stderr)
        try:
            copy = run_process([self.docker, "cp", f"{directory}/.", f"{self.container}:/opt/harness"], 60)
        except EnvironmentError as exc:
            raise ContainerUnavailable(f"Cannot copy into container {self.container}: {exc}") from exc
        if copy.returncode:
            # A missing source is a deterministic benchmark error; a missing
            # container or Docker daemon is retryable infrastructure failure.
            try:
                state = run_process([self.docker, "inspect", "-f", "{{.State.Running}}", self.container], 10)
            except EnvironmentError as exc:
                raise ContainerUnavailable(f"Cannot inspect container {self.container}: {exc}") from exc
            if state.returncode or state.stdout.strip() != "true":
                raise ContainerUnavailable(f"Container {self.container} unavailable during copy")
            raise EnvironmentError(f"Cannot copy benchmark into container: {copy.stderr}")

    def remove_benchmark(self):
        result = self.exec(["rm", "-rf", "/opt/harness"], 30)
        if result.returncode:
            raise EnvironmentError(result.stderr)


def cleanup_attempt_containers(attempt_id: str):
    """Remove containers left when an RQ work process dies mid-task."""
    listing = run_process(["docker", "ps", "-aq", "--filter",
                           f"label=agentic-harness.attempt={attempt_id}"], 15)
    if listing.returncode:
        raise EnvironmentError(listing.stderr)
    for container in listing.stdout.split():
        result = run_process(["docker", "rm", "-f", container], 30)
        if result.returncode:
            raise EnvironmentError(result.stderr)
