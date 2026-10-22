import json
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from agentic_harness.agents import AgentAdapter, AgentError, AgentRunRequest, resolve_agent
from agentic_harness.agents import mini_runner
from agentic_harness.agents.mini_swe import MiniSweAgent
from agentic_harness.benchmarks.config import BenchmarkError, resolve_benchmark
from agentic_harness.cli import cli
from agentic_harness import environment as legacy_environment
from agentic_harness import environments as environment_package
from agentic_harness.environments.base import BaseEnvironment
from agentic_harness.environments import types as environment_types
from agentic_harness.environments.docker import (CommandResult, CommandTimeout,
                                                ContainerUnavailable, DockerEnvironment,
                                                EnvironmentError, docker_exec)
from agentic_harness.execution.submission import _copy_submission
from agentic_harness.execution.task import execute_task
from agentic_harness.core.models import Task, TaskJob, TaskResult
from agentic_harness.agents.mini_runner import ExistingContainerEnvironment, model_config_for
from agentic_harness.persistence.sqlite import RunStore
from agentic_harness.verification.json_result import parse_verification

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


class FakeEnvironment:
    created = []

    def __init__(self, image, workspace):
        self.image = image
        self.workspace = Path(workspace)
        self.container = f"fake-{len(self.created)}"
        self.directory = None
        self.closed = False
        self.created.append(self)

    def __enter__(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *_):
        self.closed = True

    def copy_benchmark(self, directory):
        self.directory = Path(directory)

    def remove_benchmark(self):
        self.directory = None

    def exec(self, args, timeout, *, cwd="/workspace"):
        if args[1] == "prepare.py":
            (self.workspace / "solution.py").write_text("# stub\n")
            (self.workspace / "test_solution.py").write_text("trusted tests")
            return CommandResult(args, 0, "setup complete", "")
        if args[1] == "grade.py":
            assert (self.workspace / "test_solution.py").read_text() == "trusted tests"
            passed = "return 4" in (self.workspace / "solution.py").read_text()
            value = {"task_id": args[2], "passed": passed, "score": float(passed),
                     "details": {"tests": "passed" if passed else "failed"}}
            return CommandResult(args, 0, json.dumps(value), "")
        if args[1] == "evaluate.py":
            mapped = [str(self.workspace) if x == "/workspace" else x for x in args]
            completed = subprocess.run([sys.executable, str(self.directory / "evaluate.py"), *mapped[2:]],
                                       capture_output=True, text=True)
            return CommandResult(args, completed.returncode, completed.stdout, completed.stderr)
        raise AssertionError(args)


class FakeAgent:
    def __init__(self, fail=False, timeout=False):
        self.fail = fail
        self.timeout = timeout

    def run(self, request: AgentRunRequest):
        task, container, image, timeout, log = (
            request.task, request.container, request.image, request.timeout, request.log
        )
        log.write_text(f"ran {task.id} in {container} on {image}")
        workspace = FakeEnvironment.created[-1].workspace
        if task.id.startswith("py-"):
            assert not (workspace / "test_solution.py").exists()
        if task.id == "bash-001" and not self.fail:
            (workspace / "hello.txt").write_text("Hello World")
        if task.id == "py-001" and not self.fail:
            (workspace / "solution.py").write_text("def add():\n    return 4\n")
        if self.timeout:
            raise CommandTimeout("agent timed out")
        return {"exit_status": "LimitsExceeded" if self.fail else "Submitted"}


def job(tmp, benchmark, task_id):
    store = RunStore(tmp / "runs.db")
    store.create_run("run-1", benchmark, "agents/mini-swe-agent", [task_id])
    return TaskJob("run-1", benchmark, task_id, "agents/mini-swe-agent",
                   str(BENCHMARKS), str(tmp / "runs.db"), str(tmp / "logs"))


def queued_store(tmp, *, attempts=2):
    store = RunStore(tmp / "runs.sqlite3")
    task = TaskJob("queued-run", "bash-operations", "bash-001", "agents/mini-swe-agent",
                   str(BENCHMARKS), str(store.path), str(tmp / "logs"))
    store.submit([task], max_attempts=attempts)
    return store


class HarnessTests(unittest.TestCase):
    def setUp(self):
        FakeEnvironment.created = []

    def test_environment_protocol_accepts_docker_and_fake_providers(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            self.assertIsInstance(DockerEnvironment("alpine:latest", workspace), BaseEnvironment)
            self.assertIsInstance(FakeEnvironment("alpine:latest", workspace), BaseEnvironment)

    def test_environment_types_share_class_identity_across_compatibility_imports(self):
        for name in ("CommandResult", "EnvironmentError", "CommandTimeout", "ContainerUnavailable"):
            shared = getattr(environment_types, name)
            self.assertIs(getattr(environment_package, name), shared)
            self.assertIs(getattr(legacy_environment, name), shared)
            self.assertIs(getattr(importlib.import_module(
                "agentic_harness.environments.docker"), name), shared)
        self.assertIsInstance(CommandResult(["sh", "-c", "exit 7"], 7, "", "failed"),
                              environment_types.CommandResult)
        self.assertTrue(issubclass(CommandTimeout, EnvironmentError))
        self.assertTrue(issubclass(ContainerUnavailable, EnvironmentError))

    def test_agent_registry_resolves_builtin_and_legacy_path(self):
        agent = resolve_agent("agents/mini-swe-agent")
        self.assertIsInstance(agent, MiniSweAgent)
        self.assertIsInstance(agent, AgentAdapter)
        self.assertIsInstance(resolve_agent("./agents/mini-swe-agent"), MiniSweAgent)
        from agentic_harness.agents.mini_swe import resolve_agent as legacy_resolve
        self.assertIsInstance(legacy_resolve("agents/mini-swe-agent"), MiniSweAgent)

    def test_agent_registry_resolves_plugin_and_rejects_invalid_adapters(self):
        class ExampleAgent:
            def run(self, request: AgentRunRequest) -> dict:
                return {"exit_status": "Submitted"}

        plugin = types.SimpleNamespace(name="agents/example", load=lambda: ExampleAgent)
        with patch("agentic_harness.agents.registry.entry_points", return_value=[plugin]) as discover:
            self.assertIsInstance(resolve_agent("agents/example"), ExampleAgent)
            discover.assert_called_once_with(group="agentic_harness.agents")
            with self.assertRaisesRegex(AgentError, "not found or ambiguous"):
                resolve_agent("agents/missing")

        with patch("agentic_harness.agents.registry.entry_points", return_value=[plugin, plugin]):
            with self.assertRaisesRegex(AgentError, "not found or ambiguous"):
                resolve_agent("agents/example")

        broken = types.SimpleNamespace(name="agents/broken",
                                       load=lambda: (_ for _ in ()).throw(ImportError("missing package")))
        with patch("agentic_harness.agents.registry.entry_points", return_value=[broken]):
            with self.assertRaisesRegex(AgentError, "Cannot load agent"):
                resolve_agent("agents/broken")

        for factory in (lambda: object(), lambda: types.SimpleNamespace(run=42),
                        lambda: types.SimpleNamespace(run=lambda: {}),
                        lambda: (_ for _ in ()).throw(RuntimeError("bad initialization"))):
            invalid = types.SimpleNamespace(name="agents/invalid", load=lambda: factory)
            with self.subTest(factory=factory), patch(
                    "agentic_harness.agents.registry.entry_points", return_value=[invalid]):
                with self.assertRaises(AgentError):
                    resolve_agent("agents/invalid")

    def test_third_party_adapter_runs_through_executor(self):
        requests = []

        class ExampleAgent:
            def run(self, request: AgentRunRequest) -> dict:
                requests.append(request)
                self_outer.assertEqual(request.task.data, {})
                self_outer.assertEqual(request.task.id, "bash-001")
                self_outer.assertEqual(request.model_name, "example/model")
                self_outer.assertEqual(request.container, FakeEnvironment.created[-1].container)
                (FakeEnvironment.created[-1].workspace / "hello.txt").write_text("Hello World")
                request.log.write_text("example adapter submitted")
                return {"exit_status": "Submitted"}

        self_outer = self
        plugin = types.SimpleNamespace(name="agents/example", load=lambda: ExampleAgent)
        with tempfile.TemporaryDirectory() as tmp, patch(
                "agentic_harness.agents.registry.entry_points", return_value=[plugin]):
            root = Path(tmp)
            database = root / "runs.db"
            RunStore(database).create_run("plugin-run", "bash-operations", "agents/example", ["bash-001"])
            task = TaskJob("plugin-run", "bash-operations", "bash-001", "agents/example",
                           str(BENCHMARKS), str(database), str(root / "logs"),
                           model_name="example/model")
            result = execute_task(task, environment_factory=FakeEnvironment)
            self.assertEqual((result.status, result.score), ("passed", 1.0), result.error)
            self.assertEqual(len(requests), 1)
            self.assertTrue(all(env.closed for env in FakeEnvironment.created))
            saved = RunStore(database).get_run("plugin-run")
            self.assertEqual(saved["tasks"][0]["result"]["status"], "passed")
            self.assertEqual(saved["aggregate"]["accuracy"], 1.0)

    def test_cli_accepts_plugin_identifier_as_serializable_job(self):
        class ExampleAgent:
            def run(self, request: AgentRunRequest) -> dict:
                return {"exit_status": "Submitted"}

        plugin = types.SimpleNamespace(name="agents/example", load=lambda: ExampleAgent)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"HARNESS_HOME": tmp}), patch(
                "agentic_harness.agents.registry.entry_points", return_value=[plugin]):
            response = CliRunner().invoke(cli, ["run", "--agent", "agents/example",
                                               "--benchmark", "bash-operations",
                                               "--task_ids", "bash-001"])
            self.assertEqual(response.exit_code, 0, response.output)
            run_id = response.output.splitlines()[0].split(": ")[1]
            stored_job = RunStore(Path(tmp) / "runs.sqlite3").get_job(run_id, "bash-001")
            self.assertEqual(stored_job.agent, "agents/example")

    def test_invalid_adapter_status_is_agent_error(self):
        class InvalidStatusAgent:
            def __init__(self, status):
                self.status = status

            def run(self, request: AgentRunRequest):
                return self.status

        for status in (None, {}, {"exit_status": 4}):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                task = job(Path(tmp), "bash-operations", "bash-001")
                result = execute_task(task, environment_factory=FakeEnvironment,
                                      agent_factory=lambda _: InvalidStatusAgent(status))
                self.assertEqual((result.status, result.error_class), ("agent_error", "agent"))
                self.assertIn("exit_status", result.error)
                self.assertEqual(RunStore(task.database).get_run(task.run_id)["tasks"][0]
                                 ["result"]["status"], "agent_error")

    def execute_queued_attempt(self, store, agent, *, environment_factory=FakeEnvironment):
        dispatch = store.pending_dispatches()[0]["id"]
        attempt = store.claim("queued-run", "bash-001", dispatch, "test-worker")
        self.assertIsNotNone(attempt)
        task = replace(store.get_job("queued-run", "bash-001"), attempt_id=attempt)
        result = execute_task(task, environment_factory=environment_factory,
                              agent_factory=lambda _: agent)
        return result, attempt

    def test_legacy_imports_reexport_component_implementations(self):
        paths = [
            ("benchmark", "benchmarks.config", "resolve_benchmark"),
            ("environment", "environments.docker", "DockerEnvironment"),
            ("verifier", "verification.json_result", "parse_verification"),
            ("executor", "execution.task", "execute_task"),
            ("dispatch", "scheduling.worker", "process_task"),
            ("storage", "persistence.sqlite", "RunStore"),
            ("models", "core.models", "TaskJob"),
            ("mini_runner", "agents.mini_runner", "ExistingContainerEnvironment"),
        ]
        for old, new, name in paths:
            with self.subTest(old=old):
                legacy = importlib.import_module(f"agentic_harness.{old}")
                component = importlib.import_module(f"agentic_harness.{new}")
                self.assertIs(getattr(legacy, name), getattr(component, name))
        # RQ stores import paths in Redis, so pending jobs may use this path.
        self.assertIs(importlib.import_module("agentic_harness.scheduling.rq").process_task,
                      importlib.import_module("agentic_harness.scheduling.worker").process_task)

    def test_benchmark_fields_and_images(self):
        bash = resolve_benchmark("bash-operations", BENCHMARKS)
        python = resolve_benchmark("python-tasks", BENCHMARKS)
        self.assertEqual((len(bash.tasks), len(python.tasks)), (5, 3))
        self.assertEqual(bash.get_task("bash-001").prompt, "Create a file named 'hello.txt' containing the text 'Hello World'")
        self.assertEqual(python.get_task("py-001").image, "python:3.11-slim")
        self.assertEqual(bash.get_task("bash-001").image, "alpine:latest")

    def test_manifest_command_mapping(self):
        bash = resolve_benchmark("bash-operations", BENCHMARKS)
        py = resolve_benchmark("python-tasks", BENCHMARKS)
        self.assertEqual(bash.command("setup", bash.get_task("bash-001")),
                         ["python", "setup.py", "--workspace-dir", "/workspace", "--task", "bash-001"])
        self.assertEqual(py.command("verifier", py.get_task("py-001")),
                         ["python", "grade.py", "py-001", "--workspace", "/workspace"])

    def test_default_image_when_dataset_omits_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "extra"
            directory.mkdir()
            (directory / "dataset.json").write_text('[{"key": "x", "text": "do work"}]')
            (directory / "benchmark.yaml").write_text(
                "dataset: dataset.json\nfields: {id: key, prompt: text, image: docker_image}\n"
                "default_image: alpine:latest\nsetup: {command: [true]}\nverifier: {command: [true]}\n")
            self.assertEqual(resolve_benchmark("extra", Path(tmp)).get_task("x").image, "alpine:latest")

    def test_custom_benchmark_adapter_escape_hatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "custom"
            directory.mkdir()
            (directory / "benchmark.yaml").write_text("adapter: harness_test_custom.CustomBenchmark\n")
            module = types.ModuleType("harness_test_custom")
            class CustomBenchmark:
                def __init__(self, path):
                    self.directory = path
            module.CustomBenchmark = CustomBenchmark
            with patch.dict(sys.modules, {"harness_test_custom": module}):
                self.assertEqual(resolve_benchmark("custom", Path(tmp)).directory, directory)

    def test_mini_environment_routes_actions_to_task_container(self):
        fake_package = types.ModuleType("minisweagent")
        fake_exceptions = types.ModuleType("minisweagent.exceptions")
        fake_exceptions.Submitted = type("Submitted", (Exception,), {})
        fake_package.exceptions = fake_exceptions
        calls = []
        def fake_docker_exec(docker, container, args, timeout, *, cwd="/workspace"):
            calls.append([docker, "exec", "-w", cwd, container, *args])
            return CommandResult(args, 0, "ok\n", "")
        with patch.dict(sys.modules, {"minisweagent": fake_package,
                                      "minisweagent.exceptions": fake_exceptions}), \
             patch("agentic_harness.agents.mini_runner.docker_exec", fake_docker_exec):
            value = ExistingContainerEnvironment("task-container", "alpine:latest").execute(
                {"command": "pwd"})
        self.assertEqual(value["output"], "ok\n")
        self.assertEqual(calls[0], ["docker", "exec", "-w", "/workspace", "task-container", "sh", "-c", "pwd"])

    def test_normal_shell_observation_does_not_import_mini_dependency(self):
        with patch.dict(sys.modules, {"minisweagent.exceptions": None}), patch(
                "agentic_harness.agents.mini_runner.docker_exec",
                return_value=CommandResult(["sh", "-c", "exit 7"], 7, "", "failed")):
            observation = ExistingContainerEnvironment("task", "alpine:latest").execute(
                {"command": "exit 7"})
        self.assertEqual(observation["returncode"], 7)
        self.assertEqual(observation["output"], "failed")
        self.assertEqual(observation["exception_info"], "")

    def test_docker_exec_preserves_nonzero_tool_exit_and_detects_lost_container(self):
        calls = []

        def fake_process(args, timeout, cwd=None):
            calls.append(args)
            if args[1] == "exec":
                return CommandResult(args, 7, "", "command failed")
            return CommandResult(args, 0, "true\n", "")

        with patch("agentic_harness.environments.docker.run_process", fake_process):
            self.assertEqual(docker_exec("docker", "task", ["sh", "-c", "exit 7"], 10).returncode, 7)
        self.assertEqual(calls[1][:3], ["docker", "inspect", "-f"])

        def crashed_process(args, timeout, cwd=None):
            if args[1] == "exec":
                return CommandResult(args, 1, "", "container stopped")
            return CommandResult(args, 0, "false\n", "")

        with patch("agentic_harness.environments.docker.run_process", crashed_process):
            with self.assertRaises(ContainerUnavailable):
                docker_exec("docker", "task", ["sh", "-c", "pwd"], 10)

    def test_mini_adapter_preserves_typed_container_failure_from_child(self):
        def failed_child(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_text(
                json.dumps({"infrastructure_error": "task container exited"}))
            return types.SimpleNamespace(returncode=1)

        with tempfile.TemporaryDirectory() as tmp, patch(
                "agentic_harness.agents.mini_swe.subprocess.run", failed_child):
            with self.assertRaisesRegex(ContainerUnavailable, "task container exited"):
                MiniSweAgent().run(AgentRunRequest(
                    Task("id", "prompt", "alpine:latest"), "container",
                    "alpine:latest", 10, Path(tmp) / "agent.log"))

    def test_mini_adapter_passes_request_model_to_child(self):
        seen = {}

        def successful_child(command, **kwargs):
            seen["command"] = command
            seen["environment"] = kwargs["env"]
            Path(command[command.index("--output") + 1]).write_text(
                json.dumps({"exit_status": "Submitted"}))
            return types.SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"MSWEA_MODEL_NAME": "previous/model"}, clear=True), patch(
                "agentic_harness.agents.mini_swe.subprocess.run", successful_child):
            request = AgentRunRequest(Task("id", "prompt", "alpine:latest"), "task-container",
                                      "alpine:latest", 10, Path(tmp) / "agent.log",
                                      model_name="openrouter/example/model")
            result = MiniSweAgent().run(request)
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(seen["environment"]["MSWEA_MODEL_NAME"], "openrouter/example/model")
        self.assertEqual(seen["command"][seen["command"].index("--container") + 1], "task-container")
        self.assertEqual(seen["command"][seen["command"].index("--task") + 1], "prompt")

    def test_mini_adapter_rejects_invalid_child_status(self):
        def invalid_child(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_text("[]")
            return types.SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as tmp, patch(
                "agentic_harness.agents.mini_swe.subprocess.run", invalid_child):
            request = AgentRunRequest(Task("id", "prompt", "alpine:latest"), "task",
                                      "alpine:latest", 10, Path(tmp) / "agent.log")
            with self.assertRaisesRegex(EnvironmentError, "invalid status"):
                MiniSweAgent().run(request)

    def test_mini_runner_writes_typed_container_failure_report(self):
        modules = {name: types.ModuleType(name) for name in (
            "minisweagent", "minisweagent.agents", "minisweagent.config", "minisweagent.models")}
        modules["minisweagent.agents"].get_agent = lambda *_: types.SimpleNamespace(
            run=lambda _: (_ for _ in ()).throw(ContainerUnavailable("sandbox lost")))
        modules["minisweagent.config"].builtin_config_dir = Path("/unused")
        modules["minisweagent.config"].get_config_from_spec = lambda _: {"agent": {}, "model": {}}
        modules["minisweagent.models"].get_model = lambda config: object()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "status.json"
            argv = ["mini_runner", "--container", "task", "--image", "alpine:latest",
                    "--task", "prompt", "--output", str(output), "--timeout", "10"]
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv):
                self.assertEqual(mini_runner.main(), 1)
            self.assertEqual(json.loads(output.read_text()),
                             {"infrastructure_error": "sandbox lost"})

    def test_container_crash_retries_in_fresh_workspace(self):
        class CrashingOnceAgent:
            calls = 0
            workspaces = []

            def run(self, request: AgentRunRequest):
                task, container, image, timeout, log = (
                    request.task, request.container, request.image, request.timeout, request.log
                )
                workspace = FakeEnvironment.created[-1].workspace
                self.workspaces.append(workspace)
                self.calls += 1
                if self.calls == 1:
                    (workspace / "old-attempt").write_text("partial")
                    raise ContainerUnavailable("task container exited")
                assert not (workspace / "old-attempt").exists()
                (workspace / "hello.txt").write_text("Hello World")
                log.write_text("second attempt")
                return {"exit_status": "Submitted"}

        with tempfile.TemporaryDirectory() as tmp:
            store = queued_store(Path(tmp))
            agent = CrashingOnceAgent()
            first, first_id = self.execute_queued_attempt(store, agent)
            self.assertEqual((first.status, first.error_class), ("error", "environment"))
            self.assertEqual(store.get_run("queued-run")["status"], "queued")
            second, second_id = self.execute_queued_attempt(store, agent)
            self.assertEqual(second.status, "passed", second.error)
            self.assertNotEqual(first_id, second_id)
            self.assertNotEqual(*agent.workspaces)
            self.assertFalse(agent.workspaces[0].exists())
            self.assertTrue(all(env.closed for env in FakeEnvironment.created))
            saved = store.get_run("queued-run")
            self.assertEqual([a["error_class"] for a in saved["tasks"][0]["attempts"]],
                             ["environment", None])
            self.assertEqual(saved["aggregate"]["accuracy"], 1.0)

    def test_nonzero_agent_tool_result_does_not_retry(self):
        class NonzeroAgent:
            def run(self, request: AgentRunRequest):
                task, container, image, timeout, log = (
                    request.task, request.container, request.image, request.timeout, request.log
                )
                observation = ExistingContainerEnvironment(container, image).execute(
                    {"command": "exit 7"})
                assert observation["returncode"] == 7
                assert observation["exception_info"] == ""
                log.write_text("tool returned 7")
                return {"exit_status": "Submitted"}

        with tempfile.TemporaryDirectory() as tmp, patch(
                "agentic_harness.agents.mini_runner.docker_exec",
                return_value=CommandResult(["sh", "-c", "exit 7"], 7, "", "")):
            store = queued_store(Path(tmp))
            result, _ = self.execute_queued_attempt(store, NonzeroAgent())
            self.assertEqual(result.status, "failed")
            self.assertEqual(store.pending_dispatches(), [])
            self.assertEqual(len(store.get_run("queued-run")["tasks"][0]["attempts"]), 1)

    def test_agent_exception_remains_terminal_agent_error(self):
        class FailingAgent:
            def run(self, request: AgentRunRequest):
                task, container, image, timeout, log = (
                    request.task, request.container, request.image, request.timeout, request.log
                )
                raise AgentError("model request failed")

        with tempfile.TemporaryDirectory() as tmp:
            store = queued_store(Path(tmp))
            result, _ = self.execute_queued_attempt(store, FailingAgent())
            self.assertEqual((result.status, result.error_class), ("agent_error", "agent"))
            self.assertEqual(store.pending_dispatches(), [])
            self.assertEqual(len(store.get_run("queued-run")["tasks"][0]["attempts"]), 1)

    def test_repeated_container_crashes_are_terminal_and_persist_across_processes(self):
        class AlwaysCrashes:
            def run(self, request: AgentRunRequest):
                task, container, image, timeout, log = (
                    request.task, request.container, request.image, request.timeout, request.log
                )
                raise ContainerUnavailable("task container exited")

        with tempfile.TemporaryDirectory() as tmp:
            store = queued_store(Path(tmp))
            first, first_id = self.execute_queued_attempt(store, AlwaysCrashes())
            self.assertEqual(first.error_class, "environment")
            second, second_id = self.execute_queued_attempt(store, AlwaysCrashes())
            self.assertEqual((second.status, second.error_class), ("error", "environment"))
            self.assertNotEqual(first_id, second_id)
            self.assertEqual(store.pending_dispatches(), [])
            env = {**os.environ, "HARNESS_HOME": tmp}
            for command in ("status", "results"):
                output = subprocess.run([sys.executable, "-m", "agentic_harness.cli",
                                         command, "--run_id", "queued-run"], env=env,
                                        capture_output=True, text=True, check=True)
                persisted = json.loads(output.stdout)
                self.assertEqual(persisted["status"], "completed_with_errors")
                self.assertEqual(len(persisted["tasks"][0]["attempts"]), 2)
                self.assertEqual(persisted["tasks"][0]["result"]["error_class"], "environment")
                self.assertIn("task container exited", persisted["tasks"][0]["result"]["error"])
                self.assertTrue(all(Path(a["logs"]["error"]).is_file()
                                    for a in persisted["tasks"][0]["attempts"]))

    def test_openrouter_model_uses_native_backend_for_cost(self):
        with patch.dict(os.environ, {"MSWEA_MODEL_NAME": "openrouter/anthropic/claude-sonnet-5"}):
            config = model_config_for({"model": {"model_kwargs": {"drop_params": True}}})
        self.assertEqual(config["model_class"], "openrouter")
        self.assertEqual(config["model_name"], "anthropic/claude-sonnet-5")
        self.assertEqual(config["model_kwargs"], {})
        self.assertEqual(model_config_for({"model": {"model_name": "openai/gpt-5-mini"}}),
                         {"model_name": "openai/gpt-5-mini"})

    def test_openrouter_agent_processes_response_cost_and_container_action(self):
        try:
            from minisweagent.agents import get_agent
            from minisweagent.models import get_model
        except ImportError:
            self.skipTest("mini-swe-agent is not installed")
        model = get_model(config=model_config_for(
            {"model": {"model_name": "openrouter/anthropic/claude-sonnet-5"}}))
        self.assertEqual(type(model).__name__, "OpenRouterModel")
        response = {
            "usage": {"cost": 0.002},
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "", "tool_calls": [{
                    "id": "call-1", "type": "function", "function": {"name": "bash",
                    "arguments": json.dumps({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"})},
                }],
            }}],
        }
        calls = []
        def fake_docker_exec(docker, container, args, timeout, *, cwd="/workspace"):
            calls.append([docker, "exec", "-w", cwd, container, *args])
            return CommandResult(args, 0, "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", "")
        with patch.object(model, "_query", return_value=response), \
             patch("agentic_harness.agents.mini_runner.docker_exec", fake_docker_exec):
            agent = get_agent(model, ExistingContainerEnvironment("task-container", "alpine:latest"), {
                "agent_class": "default", "system_template": "Use a shell.",
                "instance_template": "{{task}}", "step_limit": 2, "cost_limit": 0.10,
            })
            result = agent.run("Finish")
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(agent.cost, 0.002)
        self.assertEqual(calls[0][4], "task-container")

    def test_missing_items(self):
        with self.assertRaises(BenchmarkError):
            resolve_benchmark("missing", BENCHMARKS)
        with self.assertRaises(BenchmarkError):
            resolve_benchmark("../etc", BENCHMARKS)
        with self.assertRaises(BenchmarkError):
            resolve_benchmark("bash-operations", BENCHMARKS).get_task("missing")
        with self.assertRaises(AgentError):
            resolve_agent("agents/missing")

    def test_verifier_exit_conventions(self):
        fail = CommandResult([], 1, json.dumps({"task_id": "bash-001", "passed": False,
                                                 "score": 0.0, "details": {}}), "")
        self.assertFalse(parse_verification(fail, "bash-001", [0, 1])["passed"])
        with self.assertRaises(EnvironmentError):
            parse_verification(fail, "bash-001", [0])
        pyfail = CommandResult([], 0, json.dumps({"task_id": "py-001", "passed": False,
                                                   "score": 0.0, "details": {}}), "")
        self.assertFalse(parse_verification(pyfail, "py-001", [0])["passed"])
        with self.assertRaises(EnvironmentError):
            parse_verification(CommandResult([], 0, "not json", ""), "py-001", [0])
        mapped = CommandResult([], 0, 'log line\n{"id":"x","ok":true,"points":0.5,"info":"partial"}', "")
        parsed = parse_verification(mapped, "x", [0], parser="json_last_line",
                                    fields={"id": "id", "passed": "ok", "score": "points", "details": "info"})
        self.assertEqual(parsed, {"passed": True, "score": 0.5, "details": "partial"})

    def test_real_python_grader_returns_failure_json_with_zero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run([sys.executable, str(BENCHMARKS / "python-tasks" / "grade.py"),
                                        "py-001", "--workspace", tmp], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0)
            self.assertFalse(json.loads(completed.stdout)["passed"])

    def test_docker_start_and_cleanup(self):
        calls = []
        def fake_run(args, timeout, cwd=None):
            calls.append(args)
            return CommandResult(args, 0, "id\n", "")
        with tempfile.TemporaryDirectory() as tmp, patch("agentic_harness.environments.docker.run_process", fake_run):
            with DockerEnvironment("alpine:latest", Path(tmp)) as env:
                self.assertTrue(env.container.startswith("harness-"))
                env.exec(["sh", "-c", "true"], 5)
        self.assertEqual(calls[0][1], "run")
        self.assertEqual(calls[-1][1:3], ["rm", "-f"])

    def test_docker_resource_limits_and_attempt_label(self):
        calls = []
        def fake_run(args, timeout, cwd=None):
            calls.append(args)
            return CommandResult(args, 0, "id\n", "")
        with tempfile.TemporaryDirectory() as tmp, patch("agentic_harness.environments.docker.run_process", fake_run):
            with DockerEnvironment("alpine:latest", Path(tmp), memory="512m", cpus=0.5,
                                   pids_limit=64, attempt_id="attempt-1"):
                pass
        self.assertIn("--memory", calls[0])
        self.assertIn("512m", calls[0])
        self.assertIn("--cpus", calls[0])
        self.assertIn("0.5", calls[0])
        self.assertIn("--pids-limit", calls[0])
        self.assertIn("agentic-harness.attempt=attempt-1", calls[0])

    def test_docker_cleanup_failure_is_retryable_infrastructure_error(self):
        for raise_error in (False, True):
            with self.subTest(raise_error=raise_error):
                def fake_run(args, timeout, cwd=None):
                    if args[1] == "rm":
                        if raise_error:
                            raise EnvironmentError("daemon unavailable")
                        return CommandResult(args, 1, "", "daemon unavailable")
                    return CommandResult(args, 0, "container-id", "")
                with tempfile.TemporaryDirectory() as tmp, patch(
                        "agentic_harness.environments.docker.run_process", fake_run):
                    with self.assertRaisesRegex(ContainerUnavailable, "Docker cleanup failed"):
                        with DockerEnvironment("alpine:latest", Path(tmp)):
                            pass

    def test_container_cleanup_failure_schedules_fresh_attempt(self):
        class CleanupFailure(FakeEnvironment):
            failed = False

            def __exit__(self, *_):
                super().__exit__(*_)
                if not self.failed:
                    self.failed = True
                    raise ContainerUnavailable("Docker cleanup failed")

        with tempfile.TemporaryDirectory() as tmp:
            store = queued_store(Path(tmp), attempts=2)
            result, first = self.execute_queued_attempt(
                store, FakeAgent(), environment_factory=CleanupFailure)
            self.assertEqual((result.status, result.error_class), ("error", "environment"))
            self.assertEqual(store.get_run("queued-run")["tasks"][0]["status"], "queued")
            second, next_attempt = self.execute_queued_attempt(
                store, FakeAgent(), environment_factory=FakeEnvironment)
            self.assertNotEqual(first, next_attempt)
            self.assertEqual(second.status, "passed")
            self.assertEqual(len(store.get_run("queued-run")["tasks"][0]["attempts"]), 2)

    def test_docker_copy_distinguishes_infrastructure_from_bad_source(self):
        for mode, expected in (("cli-unavailable", ContainerUnavailable),
                               ("container-dead", ContainerUnavailable),
                               ("bad-source", EnvironmentError)):
            with self.subTest(mode=mode):
                def fake_run(args, timeout, cwd=None):
                    action = args[1]
                    if action == "cp":
                        if mode == "cli-unavailable":
                            raise EnvironmentError("Docker CLI unavailable")
                        return CommandResult(args, 1, "", "copy failed")
                    if action == "inspect":
                        return CommandResult(args, 0, "false" if mode == "container-dead" else "true", "")
                    return CommandResult(args, 0, "container-id", "")
                with tempfile.TemporaryDirectory() as tmp, patch(
                        "agentic_harness.environments.docker.run_process", fake_run):
                    with DockerEnvironment("alpine:latest", Path(tmp)) as env:
                        with self.assertRaises(expected) as caught:
                            env.copy_benchmark(Path(tmp))
                        self.assertIs(type(caught.exception), expected)

    def test_failed_docker_start_attempts_cleanup(self):
        calls = []
        def fake_run(args, timeout, cwd=None):
            calls.append(args)
            if args[1] == "run":
                raise CommandTimeout("pull timed out")
            return CommandResult(args, 0, "", "")
        with tempfile.TemporaryDirectory() as tmp, patch("agentic_harness.environments.docker.run_process", fake_run):
            with self.assertRaises(CommandTimeout):
                with DockerEnvironment("alpine:latest", Path(tmp)):
                    pass
        self.assertEqual(calls[-1][1:3], ["rm", "-f"])

    def test_passed_bash_task_persists_logs_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = job(Path(tmp), "bash-operations", "bash-001")
            result = execute_task(j, environment_factory=FakeEnvironment,
                                  agent_factory=lambda _: FakeAgent())
            self.assertEqual(result.status, "passed", result.error)
            self.assertEqual(result.score, 1.0)
            self.assertEqual(len(FakeEnvironment.created), 2)
            self.assertEqual([env.image for env in FakeEnvironment.created],
                             ["alpine:latest", "python:3.11-slim"])
            self.assertTrue(all(env.closed for env in FakeEnvironment.created))
            self.assertTrue(Path(result.logs["verifier"]).is_file())
            saved = RunStore(j.database).get_run(j.run_id)
            self.assertEqual(saved["aggregate"]["accuracy"], 1.0)

    def test_python_trusted_test_restored_only_in_grader(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = job(Path(tmp), "python-tasks", "py-001")
            result = execute_task(j, environment_factory=FakeEnvironment,
                                  agent_factory=lambda _: FakeAgent())
            self.assertEqual(result.status, "passed", result.error)
            self.assertTrue(all(env.closed for env in FakeEnvironment.created))
            self.assertIn("grader_setup", result.logs)

    def test_agent_receives_no_private_dataset_metadata(self):
        class InspectingAgent(FakeAgent):
            def run(self, request: AgentRunRequest):
                task, container, image, timeout, log = (
                    request.task, request.container, request.image, request.timeout, request.log
                )
                self_outer.assertEqual(task.data, {})
                return super().run(request)
        self_outer = self
        with tempfile.TemporaryDirectory() as tmp:
            j = job(Path(tmp), "python-tasks", "py-001")
            result = execute_task(j, environment_factory=FakeEnvironment,
                                  agent_factory=lambda _: InspectingAgent())
            self.assertEqual(result.status, "passed", result.error)

    def test_hidden_test_directory_is_removed_before_agent(self):
        class DirectoryEnvironment(FakeEnvironment):
            def exec(self, args, timeout, *, cwd="/workspace"):
                result = super().exec(args, timeout, cwd=cwd)
                if args[1] == "prepare.py":
                    hidden = self.workspace / "private_tests"
                    hidden.mkdir()
                    (hidden / "secret.py").write_text("trusted test")
                return result
        class InspectingAgent(FakeAgent):
            def run(self, request: AgentRunRequest):
                task, container, image, timeout, log = (
                    request.task, request.container, request.image, request.timeout, request.log
                )
                self_outer.assertFalse((DirectoryEnvironment.created[-1].workspace / "private_tests").exists())
                return super().run(request)
        self_outer = self
        with tempfile.TemporaryDirectory() as tmp:
            j = job(Path(tmp), "python-tasks", "py-001")
            benchmark = resolve_benchmark("python-tasks", BENCHMARKS)
            benchmark.config["hide_from_agent"].append("private_tests")
            with patch("agentic_harness.execution.task.resolve_benchmark", return_value=benchmark):
                result = execute_task(j, environment_factory=DirectoryEnvironment,
                                      agent_factory=lambda _: InspectingAgent())
            self.assertEqual(result.status, "passed", result.error)

    def test_submission_cannot_overwrite_grader_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "submission"
            target = Path(tmp) / "grader"
            source.mkdir()
            target.mkdir()
            (source / "test_solution.py").write_text("assert False")
            trusted = target / "test_solution.py"
            trusted.write_text("assert True")
            with self.assertRaises(EnvironmentError):
                _copy_submission(source, target, ["."], ["test_solution.py"])
            self.assertEqual(trusted.read_text(), "assert True")
            (source / "link.py").symlink_to("test_solution.py")
            with self.assertRaises(EnvironmentError):
                _copy_submission(source, target, ["link.py"], ["test_solution.py"])

    def test_submission_cannot_follow_source_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "submission"
            outside = root / "outside"
            source.mkdir()
            outside.mkdir()
            (outside / "solution.py").write_text("external")
            (source / "nested").symlink_to(outside, target_is_directory=True)
            for index, paths in enumerate((["nested/solution.py"], ["."])):
                target = root / f"grader-{index}"
                target.mkdir()
                with self.subTest(paths=paths), self.assertRaises(EnvironmentError):
                    _copy_submission(source, target, paths)
                self.assertFalse((target / "nested").exists())
            self.assertEqual((outside / "solution.py").read_text(), "external")

    def test_submission_rejects_dangling_source_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "submission", root / "grader"
            source.mkdir()
            target.mkdir()
            (source / "link.txt").symlink_to("missing.txt")
            with self.assertRaises(EnvironmentError):
                _copy_submission(source, target, ["."])
            self.assertFalse((target / "link.txt").exists())

    def test_submission_cannot_follow_grader_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target, outside = root / "submission", root / "grader", root / "outside"
            (source / "nested").mkdir(parents=True)
            target.mkdir()
            outside.mkdir()
            (source / "nested" / "solution.py").write_text("agent solution")
            (target / "nested").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(EnvironmentError):
                _copy_submission(source, target, ["nested/solution.py"])
            self.assertFalse((outside / "solution.py").exists())

            (source / "direct.txt").write_text("agent solution")
            (outside / "trusted.txt").write_text("trusted")
            (target / "direct.txt").symlink_to(outside / "trusted.txt")
            with self.assertRaises(EnvironmentError):
                _copy_submission(source, target, ["direct.txt"])
            self.assertEqual((outside / "trusted.txt").read_text(), "trusted")

    def test_submission_preserves_internal_source_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "submission", root / "grader"
            source.mkdir()
            target.mkdir()
            (source / "real.txt").write_text("inside")
            (source / "link.txt").symlink_to("real.txt")
            (source / "absolute.txt").symlink_to(source / "real.txt")
            _copy_submission(source, target, ["."])
            self.assertTrue((target / "link.txt").is_symlink())
            self.assertEqual((target / "link.txt").readlink(), Path("real.txt"))
            self.assertEqual((target / "link.txt").read_text(), "inside")
            self.assertTrue((target / "absolute.txt").is_symlink())
            self.assertEqual((target / "absolute.txt").read_text(), "inside")

    def test_manifest_only_benchmark_with_fake_runtime(self):
        class ScriptEnvironment(FakeEnvironment):
            def exec(self, args, timeout, *, cwd="/workspace"):
                mapped = [str(self.workspace) if value == "/workspace" else value for value in args[2:]]
                completed = subprocess.run([sys.executable, str(self.directory / args[1]), *mapped],
                                           capture_output=True, text=True, timeout=timeout)
                return CommandResult(args, completed.returncode, completed.stdout, completed.stderr)

        class WritingAgent:
            def run(self, request: AgentRunRequest):
                task, container, image, timeout, log = (
                    request.task, request.container, request.image, request.timeout, request.log
                )
                self_outer.assertEqual(task.data, {})
                workspace = ScriptEnvironment.created[-1].workspace
                self_outer.assertEqual((workspace / "seed.txt").read_text(), "ready")
                (workspace / "answer.txt").write_text("done")
                log.write_text("wrote answer.txt")
                return {"exit_status": "Submitted"}

        self_outer = self
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "benchmarks" / "synthetic"
            directory.mkdir(parents=True)
            (directory / "dataset.json").write_text(json.dumps([
                {"case": "case-001", "instruction": "Write done to answer.txt", "private_answer": "done"}
            ]))
            (directory / "benchmark.yaml").write_text(
                "dataset: dataset.json\n"
                "fields: {id: case, prompt: instruction, image: docker_image}\n"
                "default_image: python:3.11-slim\n"
                "setup: {command: [python, setup.py, '{task_id}', '{workspace}'], runtime: container}\n"
                "verifier: {command: [python, verify.py, '{task_id}', '{workspace}'], runtime: container, result: json}\n"
                "submission_paths: [answer.txt]\n"
            )
            (directory / "setup.py").write_text(
                "import pathlib, sys\n"
                "(pathlib.Path(sys.argv[2]) / 'seed.txt').write_text('ready')\n"
            )
            (directory / "verify.py").write_text(
                "import json, pathlib, sys\n"
                "passed = (pathlib.Path(sys.argv[2]) / 'answer.txt').read_text() == 'done'\n"
                "print(json.dumps({'task_id': sys.argv[1], 'passed': passed, 'score': float(passed), 'details': {}}))\n"
            )
            database = root / "runs.db"
            RunStore(database).create_run("synthetic-run", "synthetic", "agents/mini-swe-agent", ["case-001"])
            j = TaskJob("synthetic-run", "synthetic", "case-001", "agents/mini-swe-agent",
                        str(root / "benchmarks"), str(database), str(root / "logs"))
            result = execute_task(j, environment_factory=ScriptEnvironment,
                                  agent_factory=lambda _: WritingAgent())
            self.assertEqual(result.status, "passed", result.error)
            self.assertEqual(result.score, 1.0)
            stored = RunStore(database).get_run("synthetic-run")
            self.assertEqual(stored["tasks"][0]["result"]["status"], "passed")
            self.assertEqual(stored["aggregate"]["accuracy"], 1.0)

    def test_malicious_task_id_cannot_escape_log_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RunStore(root / "runs.db")
            store.create_run("../run", "bash-operations", "agents/mini-swe-agent", ["../task"])
            j = TaskJob("../run", "bash-operations", "../task", "agents/mini-swe-agent",
                        str(BENCHMARKS), str(root / "runs.db"), str(root / "logs"))
            result = execute_task(j, environment_factory=FakeEnvironment,
                                  agent_factory=lambda _: FakeAgent())
            self.assertEqual(result.status, "error")
            self.assertTrue(Path(result.logs["error"]).resolve().is_relative_to((root / "logs").resolve()))
            self.assertFalse((root / "task").exists())

    def test_long_task_id_persists_inspectable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_id = "a" * 300
            j = job(Path(tmp), "bash-operations", task_id)
            result = execute_task(j, environment_factory=FakeEnvironment,
                                  agent_factory=lambda _: FakeAgent())
            self.assertEqual(result.status, "error")
            error_log = Path(result.logs["error"])
            self.assertTrue(error_log.is_file())
            self.assertEqual(len(error_log.parent.name), 16)
            self.assertIn("Task not found", error_log.read_text())
            saved = RunStore(j.database).get_run(j.run_id)
            self.assertEqual(saved["status"], "completed_with_errors")
            self.assertEqual(saved["tasks"][0]["status"], "error")
            self.assertEqual(saved["tasks"][0]["result"]["status"], "error")

    def test_failed_solution_is_not_successful_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = job(Path(tmp), "bash-operations", "bash-001")
            result = execute_task(j, environment_factory=FakeEnvironment,
                                  agent_factory=lambda _: FakeAgent(fail=True))
            self.assertEqual(result.status, "agent_error")
            self.assertEqual(result.error_class, "agent")
            self.assertFalse(result.passed)
            self.assertEqual(RunStore(j.database).get_run(j.run_id)["status"], "completed_with_errors")

    def test_agent_timeout_still_verifies_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = job(Path(tmp), "bash-operations", "bash-001")
            result = execute_task(j, environment_factory=FakeEnvironment,
                                  agent_factory=lambda _: FakeAgent(timeout=True))
            self.assertEqual(result.status, "timeout")
            self.assertEqual(result.error_class, "agent_timeout")
            self.assertTrue(all(env.closed for env in FakeEnvironment.created))

    def test_setup_failure_and_timeout_persist(self):
        class FailingEnvironment(FakeEnvironment):
            def exec(self, args, timeout, *, cwd="/workspace"):
                if args[1] == "prepare.py":
                    raise CommandTimeout("setup timed out")
                return super().exec(args, timeout, cwd=cwd)
        with tempfile.TemporaryDirectory() as tmp:
            j = job(Path(tmp), "python-tasks", "py-001")
            result = execute_task(j, environment_factory=FailingEnvironment,
                                  agent_factory=lambda _: FakeAgent())
            self.assertEqual(result.status, "timeout")
            self.assertEqual(result.error_class, "setup_timeout")
            self.assertIn("setup timed out", result.error)
            self.assertTrue(FakeEnvironment.created[0].closed)
            self.assertEqual(RunStore(j.database).get_run(j.run_id)["tasks"][0]["status"], "timeout")

    def test_store_aggregation_includes_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "runs.db")
            store.create_run("r", "b", "a", ["one", "two"])
            store.save_result(TaskResult("r", "one", "passed", True, 1.0))
            value = store.get_run("r")
            self.assertEqual(value["status"], "running")
            self.assertEqual(value["aggregate"]["accuracy"], 0.5)
            with self.assertRaises(KeyError):
                store.get_run("missing")

    def test_cli_run_status_results(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"HARNESS_HOME": tmp}):
            def fake_execute(j):
                result = TaskResult(j.run_id, j.task_id, "failed", False, 0.0,
                                    duration=0.1, logs={"agent": "mock.log"})
                RunStore(j.database).save_result(result)
                return result
            with patch("agentic_harness.cli.execute_task", fake_execute):
                output = CliRunner().invoke(cli, ["run", "--agent", "agents/mini-swe-agent",
                                                  "--benchmark", "bash-operations", "--task_ids", "bash-001",
                                                  "--sync"])
            self.assertEqual(output.exit_code, 0, output.output)
            run_id = output.output.splitlines()[0].split(": ")[1]
            status = CliRunner().invoke(cli, ["status", "--run_id", run_id])
            results = CliRunner().invoke(cli, ["results", "--run_id", run_id])
            self.assertEqual(json.loads(status.output)["aggregate"]["accuracy"], 0.0)
            self.assertEqual(json.loads(results.output)["tasks"][0]["result"]["logs"]["agent"], "mock.log")
            missing = CliRunner().invoke(cli, ["run", "--agent", "agents/nope", "--benchmark", "bash-operations"])
            self.assertNotEqual(missing.exit_code, 0)
            missing_task = CliRunner().invoke(cli, ["run", "--agent", "agents/mini-swe-agent",
                                                         "--benchmark", "bash-operations", "--task_ids", "missing"])
            self.assertNotEqual(missing_task.exit_code, 0)

    def test_cli_defaults_to_all_tasks(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"HARNESS_HOME": tmp}):
            def fake_execute(j):
                seen.append(j.task_id)
                result = TaskResult(j.run_id, j.task_id, "failed")
                RunStore(j.database).save_result(result)
                return result
            with patch("agentic_harness.cli.execute_task", fake_execute):
                output = CliRunner().invoke(cli, ["run", "--agent", "agents/mini-swe-agent",
                                                  "--benchmark", "bash-operations", "--sync"])
        self.assertEqual(output.exit_code, 0, output.output)
        self.assertEqual(len(seen), 5)

    def test_task_job_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = job(Path(tmp), "bash-operations", "bash-001")
            self.assertEqual(json.loads(json.dumps(j.to_dict()))["task_id"], "bash-001")


def docker_available():
    return bool(shutil.which("docker")) and subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@unittest.skipUnless(docker_available(), "Docker daemon unavailable")
class DockerIntegrationTests(unittest.TestCase):
    def test_task_container_crash_retries_in_fresh_docker_sandbox(self):
        class CrashOnceAgent:
            def __init__(self):
                self.containers = []

            def run(self, request: AgentRunRequest):
                task, container, image, timeout, log = (
                    request.task, request.container, request.image, request.timeout, request.log
                )
                self.containers.append(container)
                if len(self.containers) == 1:
                    subprocess.run(["docker", "kill", container], capture_output=True,
                                   text=True, check=True)
                    ExistingContainerEnvironment(container, image).execute({"command": "pwd"})
                    raise AssertionError("Lost container was treated as a tool result")
                observation = ExistingContainerEnvironment(container, image).execute(
                    {"command": "printf 'Hello World' > hello.txt"})
                assert observation["returncode"] == 0, observation
                log.write_text("second attempt succeeded")
                return {"exit_status": "Submitted"}

        with tempfile.TemporaryDirectory() as tmp:
            store = queued_store(Path(tmp))
            agent = CrashOnceAgent()
            results = []
            for _ in range(2):
                dispatch = store.pending_dispatches()[0]["id"]
                attempt = store.claim("queued-run", "bash-001", dispatch, "docker-test")
                task = replace(store.get_job("queued-run", "bash-001"), attempt_id=attempt)
                results.append(execute_task(task, agent_factory=lambda _: agent))
            self.assertEqual((results[0].status, results[0].error_class), ("error", "environment"))
            self.assertEqual(results[1].status, "passed", results[1].error)
            self.assertEqual(len(set(agent.containers)), 2)
            self.assertEqual(store.get_run("queued-run")["aggregate"]["accuracy"], 1.0)
            for container in agent.containers:
                self.assertNotEqual(subprocess.run(["docker", "inspect", container],
                                                  capture_output=True).returncode, 0)

    def test_real_container_exec_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            with DockerEnvironment("alpine:latest", Path(tmp)) as env:
                container = env.container
                result = env.exec(["sh", "-c", "echo container > sample.txt"], 15)
                self.assertEqual(result.returncode, 0)
                self.assertEqual((Path(tmp) / "sample.txt").read_text().strip(), "container")
            remaining = subprocess.run(["docker", "inspect", container], capture_output=True)
            self.assertNotEqual(remaining.returncode, 0)

    def test_bash_task_end_to_end_with_deterministic_container_agent(self):
        class DockerWritingAgent:
            def run(self, request: AgentRunRequest):
                task, container, image, timeout, log = (
                    request.task, request.container, request.image, request.timeout, request.log
                )
                completed = subprocess.run(["docker", "exec", "-w", "/workspace", container,
                                            "sh", "-c", "echo 'Hello World' > hello.txt"],
                                           capture_output=True, text=True, timeout=timeout)
                log.write_text(completed.stdout + completed.stderr)
                return {"exit_status": "Submitted"}
        with tempfile.TemporaryDirectory() as tmp:
            j = job(Path(tmp), "bash-operations", "bash-001")
            result = execute_task(j, agent_factory=lambda _: DockerWritingAgent())
            self.assertEqual(result.status, "passed", result.error)
            self.assertEqual(result.score, 1.0)

    def test_new_manifest_only_benchmark_runs_end_to_end(self):
        class DockerWritingAgent:
            container = None

            def run(self, request: AgentRunRequest):
                task, container, image, timeout, log = (
                    request.task, request.container, request.image, request.timeout, request.log
                )
                self.container = container
                self.assert_public(task)
                completed = subprocess.run(
                    ["docker", "exec", "-w", "/workspace", container, "sh", "-c",
                     "test ! -e private && echo synthetic > answer.txt"],
                    capture_output=True, text=True, timeout=timeout,
                )
                log.write_text(completed.stdout + completed.stderr)
                if completed.returncode:
                    raise AssertionError(completed.stderr)
                return {"exit_status": "Submitted"}

            @staticmethod
            def assert_public(task):
                assert task.data == {}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "benchmarks" / "synthetic"
            directory.mkdir(parents=True)
            (directory / "dataset.json").write_text(json.dumps([
                {"key": "synthetic-001", "instruction": "Write synthetic to answer.txt",
                 "private_answer": "synthetic"}
            ]))
            (directory / "benchmark.yaml").write_text(
                "dataset: dataset.json\n"
                "fields: {id: key, prompt: instruction, image: docker_image}\n"
                "default_image: python:3.11-slim\n"
                "setup: {command: [python, prepare.py, '{task_id}', '{workspace}'], runtime: container}\n"
                "verifier: {command: [python, grade.py, '{task_id}', '{workspace}'], runtime: container, result: json}\n"
                "submission_paths: [answer.txt]\n"
                "hide_from_agent: [private]\n"
                "protected_paths: [private]\n"
            )
            (directory / "prepare.py").write_text(
                "import pathlib, sys\n"
                "private = pathlib.Path(sys.argv[2]) / 'private'\n"
                "private.mkdir()\n"
                "(private / 'expected.txt').write_text('synthetic')\n"
            )
            (directory / "grade.py").write_text(
                "import json, pathlib, sys\n"
                "root = pathlib.Path(sys.argv[2])\n"
                "passed = (root / 'answer.txt').read_text().strip() == (root / 'private' / 'expected.txt').read_text()\n"
                "print(json.dumps({'task_id': sys.argv[1], 'passed': passed, 'score': float(passed), 'details': {}}))\n"
            )
            database = root / "runs.db"
            RunStore(database).create_run("synthetic-run", "synthetic", "agents/mini-swe-agent",
                                          ["synthetic-001"])
            j = TaskJob("synthetic-run", "synthetic", "synthetic-001", "agents/mini-swe-agent",
                        str(root / "benchmarks"), str(database), str(root / "logs"))
            agent = DockerWritingAgent()
            result = execute_task(j, agent_factory=lambda _: agent)
            self.assertEqual(result.status, "passed", result.error)
            self.assertEqual(result.score, 1.0)
            self.assertFalse(Path(result.logs["agent"]).read_text())
            self.assertNotEqual(subprocess.run(["docker", "inspect", agent.container],
                                               capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
