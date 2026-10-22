"""Exercise the real extension points with obvious deterministic examples."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from agentic_harness.agents import AgentAdapter, AgentRunRequest, resolve_agent
from agentic_harness.agents.trial import TrialAgent
from agentic_harness.benchmarks.config import resolve_benchmark
from agentic_harness.cli import cli
from agentic_harness.core.models import Task
from agentic_harness.environments.types import CommandResult
from agentic_harness.persistence.sqlite import RunStore


ROOT = Path(__file__).resolve().parents[1]


class TrialExtensionTests(unittest.TestCase):
    def test_new_manifest_maps_fields_and_builds_commands(self):
        benchmark = resolve_benchmark("trial-benchmark", ROOT / "benchmarks")
        self.assertEqual(list(benchmark.tasks), ["trial-001", "trial-002"])
        task = benchmark.get_task("trial-001")
        self.assertIn("printf 'trial-one'", task.prompt)
        self.assertEqual(task.image, "python:3.11-slim")
        self.assertEqual(benchmark.get_task("trial-002").image, "python:3.11-slim")
        self.assertEqual(benchmark.command("setup", task),
                         ["python", "prepare.py", "trial-001", "--workspace", "/workspace"])
        self.assertEqual(benchmark.command("verifier", task),
                         ["python", "grade.py", "trial-001", "--workspace", "/workspace"])

    def test_trial_agent_resolves_via_installed_entry_point(self):
        agent = resolve_agent("agents/trial-agent")
        self.assertIsInstance(agent, TrialAgent)
        self.assertIsInstance(agent, AgentAdapter)

    def test_trial_agent_executes_in_existing_container(self):
        calls = []

        def fake_exec(docker, container, args, timeout, *, cwd):
            calls.append((docker, container, args, timeout, cwd))
            return CommandResult(args, 0, "created answer.txt\n", "")

        with tempfile.TemporaryDirectory() as tmp, patch(
                "agentic_harness.agents.trial.docker_exec", fake_exec):
            log = Path(tmp) / "agent.log"
            request = AgentRunRequest(Task("trial-001", "printf trial > answer.txt",
                                           "python:3.11-slim"),
                                      "task-container", "python:3.11-slim", 30, log)
            status = TrialAgent().run(request)
            self.assertEqual(status["exit_status"], "Submitted")
            self.assertEqual(calls, [("docker", "task-container",
                                      ["sh", "-c", request.task.prompt], 30, "/workspace")])
            self.assertIn("created answer.txt", log.read_text())

    def test_nonzero_shell_exit_is_agent_failure(self):
        def fake_exec(docker, container, args, timeout, *, cwd):
            return CommandResult(args, 7, "", "command failed")

        with tempfile.TemporaryDirectory() as tmp, patch(
                "agentic_harness.agents.trial.docker_exec", fake_exec):
            request = AgentRunRequest(Task("trial-fail", "exit 7", "python:3.11-slim"),
                                      "task-container", "python:3.11-slim", 30,
                                      Path(tmp) / "agent.log")
            self.assertEqual(TrialAgent().run(request),
                             {"exit_status": "Failed", "returncode": 7})


def docker_available():
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "info"], capture_output=True
    ).returncode == 0


@unittest.skipUnless(docker_available(), "Docker daemon unavailable")
class TrialDockerIntegrationTests(unittest.TestCase):
    def test_trial_adapter_and_benchmark_run_through_cli(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"HARNESS_HOME": tmp}):
            output = CliRunner().invoke(cli, [
                "run", "--agent", "agents/trial-agent", "--benchmark",
                "trial-benchmark", "--sync",
            ])
            self.assertEqual(output.exit_code, 0, output.output)
            run_id = output.output.splitlines()[0].split(": ")[1]
            saved = RunStore(Path(tmp) / "runs.sqlite3").get_run(run_id)
            self.assertEqual(saved["aggregate"]["accuracy"], 1.0)
            self.assertEqual({task["task_id"] for task in saved["tasks"]},
                             {"trial-001", "trial-002"})
            for task in saved["tasks"]:
                self.assertEqual(task["status"], "passed")
                self.assertEqual(task["result"]["score"], 1.0)
                self.assertTrue(Path(task["result"]["logs"]["verifier"]).is_file())
            self.assertFalse(list((Path(tmp) / "workspaces").iterdir()))


if __name__ == "__main__":
    unittest.main()
