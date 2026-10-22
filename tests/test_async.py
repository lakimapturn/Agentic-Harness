"""Deterministic Part 2 state-machine and CLI tests; no paid model calls."""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner
from rq.exceptions import DuplicateJobError

from agentic_harness.benchmarks.config import resolve_benchmark
from agentic_harness.cli import cli
from agentic_harness.scheduling.dispatch import dispatch_once, recover_once
from agentic_harness.scheduling.transport import redis_connection, task_queue
from agentic_harness.scheduling.worker import process_task
from agentic_harness.core.models import TaskJob, TaskResult
from agentic_harness.persistence.sqlite import RunStore


ROOT = Path(__file__).resolve().parents[1]


class RecordingQueue:
    name = "harness-tasks"

    def __init__(self):
        self.calls = []
        self.jobs = {}

    def enqueue_call(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["job_id"] in self.jobs:
            raise DuplicateJobError(kwargs["job_id"])
        self.jobs[kwargs["job_id"]] = SimpleNamespace(
            id=kwargs["job_id"], origin=self.name, func_name=kwargs["func"],
            args=kwargs["args"])

    def fetch_job(self, job_id):
        return self.jobs.get(job_id)


def submit(root: Path, run_id="run", task_ids=("one",), attempts=2):
    db = root / "runs.db"
    store = RunStore(db)
    jobs = [TaskJob(run_id, "synthetic", task_id, "agents/mini-swe-agent",
                    str(root / "snapshots"), str(db), str(root / "logs"),
                    model_name="openrouter/example/model") for task_id in task_ids]
    store.submit(jobs, max_attempts=attempts)
    return store


class AsyncTests(unittest.TestCase):
    def test_concurrent_schema_migration_from_existing_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.db"
            with sqlite3.connect(path) as db:
                db.executescript("""
                    CREATE TABLE runs (id TEXT PRIMARY KEY, benchmark TEXT, agent TEXT, created_at TEXT);
                    CREATE TABLE tasks (run_id TEXT, task_id TEXT, status TEXT, stage TEXT,
                                        result_json TEXT, updated_at TEXT, PRIMARY KEY(run_id, task_id));
                    CREATE TABLE attempts (id TEXT PRIMARY KEY, run_id TEXT, task_id TEXT,
                                           worker_id TEXT, status TEXT, started_at REAL,
                                           heartbeat_at REAL, lease_until REAL, ended_at REAL,
                                           error_class TEXT, duration REAL, logs_json TEXT,
                                           result_json TEXT);
                """)
            barrier = threading.Barrier(3)
            errors = []

            def migrate():
                barrier.wait()
                try:
                    RunStore(path)
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=migrate) for _ in range(2)]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            with sqlite3.connect(path) as db:
                self.assertIn("cleanup_status", {row[1] for row in db.execute("PRAGMA table_info(attempts)")})

    def test_submission_is_atomic_and_durable_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = submit(root, task_ids=("one", "two", "three"))
            self.assertEqual(RunStore(root / "runs.db").get_run("run")["status"], "queued")
            self.assertEqual(len(store.pending_dispatches()), 3)
            self.assertEqual(store.get_job("run", "one").model_name, "openrouter/example/model")
            with self.assertRaises(Exception):
                store.submit([store.get_job("run", "one"), store.get_job("run", "one")])
            self.assertEqual(len(store.get_run("run")["tasks"]), 3)
            self.assertEqual(len(store.pending_dispatches()), 3)

    def test_cli_submits_multiple_runs_without_redis_and_status_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"HARNESS_HOME": tmp}):
            with patch("agentic_harness.scheduling.transport.redis_connection", side_effect=AssertionError("Redis contacted")):
                first = CliRunner().invoke(cli, ["run", "--agent", "agents/mini-swe-agent",
                                                 "--benchmark", "bash-operations", "--task_ids",
                                                 "bash-001,bash-002"])
                second = CliRunner().invoke(cli, ["run", "--agent", "agents/mini-swe-agent",
                                                  "--benchmark", "python-tasks"])
            self.assertEqual(first.exit_code, 0, first.output)
            self.assertEqual(second.exit_code, 0, second.output)
            first_id = first.output.splitlines()[0].split(": ")[1]
            second_id = second.output.splitlines()[0].split(": ")[1]
            self.assertNotEqual(first_id, second_id)
            for run_id, count in [(first_id, 2), (second_id, 3)]:
                data = json.loads(CliRunner().invoke(cli, ["results", "--run_id", run_id]).output)
                self.assertEqual(data["status"], "queued")
                self.assertEqual(data["aggregate"]["total"], count)
                self.assertEqual(data["aggregate"]["completed"], 0)
                self.assertEqual(data["aggregate"]["accuracy"], 0)
                self.assertEqual(len(RunStore(Path(tmp) / "runs.sqlite3").pending_dispatches()), 5)
            invalid = CliRunner().invoke(cli, ["run", "--agent", "agents/mini-swe-agent",
                                               "--benchmark", "bash-operations", "--task_ids", "missing"])
            self.assertNotEqual(invalid.exit_code, 0)

    def test_submission_freezes_benchmark_files_without_storing_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "benchmarks").mkdir()
            shutil.copytree(ROOT / "benchmarks" / "bash-operations",
                            root / "benchmarks" / "bash-operations")
            state = root / "state"
            secret = "test-only-secret-should-not-be-persisted"
            with patch("agentic_harness.cli.PROJECT_ROOT", root), patch.dict(
                os.environ, {"HARNESS_HOME": str(state), "OPENROUTER_API_KEY": secret,
                             "MSWEA_MODEL_NAME": "openrouter/example/model"}):
                output = CliRunner().invoke(cli, ["run", "--agent", "agents/mini-swe-agent",
                                                  "--benchmark", "bash-operations", "--task_ids", "bash-001"])
            self.assertEqual(output.exit_code, 0, output.output)
            run_id = output.output.splitlines()[0].split(": ")[1]
            original = root / "benchmarks" / "bash-operations" / "dataset.json"
            original.write_text("[]")
            job = RunStore(state / "runs.sqlite3").get_job(run_id, "bash-001")
            frozen = resolve_benchmark("bash-operations", Path(job.benchmark_root))
            self.assertIn("hello.txt", frozen.get_task("bash-001").prompt)
            self.assertEqual(job.model_name, "openrouter/example/model")
            for path in state.glob("runs.sqlite3*"):
                self.assertNotIn(secret.encode(), path.read_bytes())

    def test_claim_is_atomic_and_duplicate_messages_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = submit(Path(tmp))
            dispatch_id = store.pending_dispatches()[0]["id"]
            barrier = threading.Barrier(3)
            claims = []

            def compete():
                barrier.wait()
                claims.append(RunStore(store.path).claim("run", "one", dispatch_id, "worker"))

            threads = [threading.Thread(target=compete) for _ in range(2)]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join()
            self.assertEqual(sum(bool(value) for value in claims), 1)
            self.assertIsNone(store.claim("run", "one", dispatch_id, "duplicate"))
            self.assertEqual(len(store.get_run("run")["tasks"][0]["attempts"]), 1)

    def test_interrupted_dispatch_redelivers_with_stable_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = submit(Path(tmp), task_ids=("one", "two"))
            queue = RecordingQueue()
            with patch.object(store, "mark_delivered", side_effect=RuntimeError("crash")):
                with self.assertRaises(RuntimeError):
                    dispatch_once(store, queue)
            self.assertEqual(dispatch_once(store, queue), 2)
            self.assertEqual(queue.calls[0]["job_id"], queue.calls[2]["job_id"])
            self.assertEqual(queue.calls[1]["job_id"], queue.calls[3]["job_id"])
            self.assertNotEqual(queue.calls[0]["job_id"], queue.calls[1]["job_id"])
            self.assertTrue(all(call["unique"] for call in queue.calls))
            self.assertEqual(store.pending_dispatches(), [])

    def test_failed_dispatch_does_not_block_other_pending_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = submit(Path(tmp), task_ids=("one", "two"))
            first = store.pending_dispatches()[0]["id"]

            class FailingQueue(RecordingQueue):
                def enqueue_call(self, **kwargs):
                    if kwargs["job_id"] == first:
                        raise RuntimeError("injected per-task enqueue failure")
                    return super().enqueue_call(**kwargs)

            queue = FailingQueue()
            with self.assertRaisesRegex(RuntimeError, "per-task enqueue failure"):
                dispatch_once(store, queue)
            self.assertEqual([row["task_id"] for row in store.pending_dispatches()], ["one"])
            self.assertEqual(len(queue.jobs), 1)
            self.assertEqual(next(iter(queue.jobs.values())).args[2], "two")

    def test_duplicate_job_is_acknowledged_only_when_payload_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = submit(Path(tmp))
            queue = RecordingQueue()
            with patch.object(store, "mark_delivered", side_effect=RuntimeError("crash")):
                with self.assertRaises(RuntimeError):
                    dispatch_once(store, queue)
            request_id = store.pending_dispatches()[0]["id"]
            queue.jobs[request_id].args = ("different.db", "run", "one", request_id)
            with self.assertRaises(DuplicateJobError):
                dispatch_once(store, queue)
            self.assertEqual(len(store.pending_dispatches()), 1)

    def test_unclaimed_delivery_is_reconciled_and_old_message_is_fenced(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = submit(Path(tmp))
            old_id = store.pending_dispatches()[0]["id"]
            store.mark_delivered(old_id)
            store.reconcile(now=time.time() + 120)
            new_id = store.pending_dispatches()[0]["id"]
            self.assertNotEqual(old_id, new_id)
            self.assertIsNone(store.claim("run", "one", old_id, "old"))
            self.assertIsNotNone(store.claim("run", "one", new_id, "new"))

    def test_queued_rq_job_is_not_invalidated_during_long_backlog(self):
        class Queue:
            present = True

            def fetch_job(self, _):
                return self if self.present else None

            def get_status(self, *, refresh=False):
                return "queued"

            def get_job_ids(self):
                return [dispatch_id] if self.present else []

        with tempfile.TemporaryDirectory() as tmp:
            store = submit(Path(tmp))
            dispatch_id = store.pending_dispatches()[0]["id"]
            store.mark_delivered(dispatch_id)
            queue = Queue()
            recover_once(store, queue=queue, now=time.time() + 120)
            self.assertEqual(store.get_run("run")["tasks"][0]["status"], "queued")
            self.assertEqual(store.claim("run", "one", dispatch_id, "worker") is not None, True)

    def test_missing_rq_queue_entry_is_reconciled(self):
        class Queue:
            def get_job_ids(self):
                return []

            def fetch_job(self, _):
                return self

            def get_status(self, *, refresh=False):
                return "queued"

        with tempfile.TemporaryDirectory() as tmp:
            store = submit(Path(tmp))
            old_id = store.pending_dispatches()[0]["id"]
            store.mark_delivered(old_id)
            recover_once(store, queue=Queue(), now=time.time() + 120)
            self.assertNotEqual(store.pending_dispatches()[0]["id"], old_id)

    def test_retry_uses_new_attempt_and_stale_result_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = submit(Path(tmp))
            first_dispatch = store.pending_dispatches()[0]["id"]
            first = store.claim("run", "one", first_dispatch, "worker")
            self.assertTrue(store.heartbeat(first))
            self.assertTrue(store.save_result(TaskResult("run", "one", status="error", error="Docker unavailable",
                                                         error_class="environment"),
                                              first))
            self.assertEqual(store.get_run("run")["status"], "queued")
            self.assertEqual(store.get_run("run")["aggregate"]["completed"], 0)
            second_dispatch = store.pending_dispatches()[0]["id"]
            second = store.claim("run", "one", second_dispatch, "worker")
            self.assertNotEqual(first, second)
            self.assertFalse(store.save_result(TaskResult("run", "one", status="passed", passed=True, score=1), first))
            self.assertFalse(store.set_stage(replace(store.get_job("run", "one"), attempt_id=first), "agent"))
            store.save_result(TaskResult("run", "one", status="passed", passed=True, score=1), second)
            data = RunStore(store.path).get_run("run")
            self.assertEqual(data["aggregate"]["accuracy"], 1)
            self.assertEqual([a["error_class"] for a in data["tasks"][0]["attempts"]],
                             ["environment", None])
            self.assertEqual(data["tasks"][0]["status"], "passed")

    def test_result_write_rolls_back_attempt_and_task_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = submit(Path(tmp), attempts=1)
            attempt = store.claim("run", "one", store.pending_dispatches()[0]["id"], "worker")
            with sqlite3.connect(store.path) as db:
                db.execute("CREATE TRIGGER fail_result BEFORE UPDATE OF result_json ON tasks "
                           "WHEN NEW.result_json IS NOT NULL BEGIN "
                           "SELECT RAISE(ABORT, 'injected result write failure'); END")
            result = TaskResult("run", "one", status="passed", passed=True, score=1.0)
            with self.assertRaises(sqlite3.DatabaseError):
                store.save_result(result, attempt)
            pending = RunStore(store.path).get_run("run")["tasks"][0]
            self.assertEqual(pending["status"], "running")
            self.assertIsNone(pending["result"])
            self.assertEqual(pending["attempts"][0]["status"], "running")
            self.assertIsNone(pending["attempts"][0]["result"])
            with sqlite3.connect(store.path) as db:
                db.execute("DROP TRIGGER fail_result")
            self.assertTrue(store.save_result(result, attempt))
            self.assertEqual(RunStore(store.path).get_run("run")["aggregate"]["accuracy"], 1.0)

    def test_only_infrastructure_failures_retry(self):
        cases = [("resolve", "error", False), ("setup", "error", False),
                 ("verifier", "error", False), ("agent_timeout", "timeout", False),
                 ("setup_timeout", "timeout", False), ("agent", "agent_error", False),
                 ("environment", "error", True), ("environment_timeout", "timeout", True),
                 ("worker", "error", True)]
        with tempfile.TemporaryDirectory() as tmp:
            store = submit(Path(tmp), task_ids=tuple(item[0] for item in cases))
            requests = {item["task_id"]: item["id"] for item in store.pending_dispatches()}
            for error_class, status, retry in cases:
                with self.subTest(error_class=error_class):
                    attempt = store.claim("run", error_class, requests[error_class], "worker")
                    store.save_result(TaskResult("run", error_class, status=status,
                                                 error_class=error_class), attempt)
                    final = next(item for item in store.get_run("run")["tasks"]
                                 if item["task_id"] == error_class)
                    self.assertEqual(final["status"] == "queued", retry)

    def test_failed_evaluation_does_not_retry_and_zero_score_is_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = submit(Path(tmp))
            attempt = store.claim("run", "one", store.pending_dispatches()[0]["id"], "worker")
            store.save_result(TaskResult("run", "one", status="failed", passed=False, score=0), attempt)
            self.assertEqual(store.pending_dispatches(), [])
            data = store.get_run("run")
            self.assertEqual(data["status"], "completed_with_failures")
            self.assertEqual(data["aggregate"]["completed"], 1)
            self.assertEqual(data["aggregate"]["errors"], 0)
            self.assertEqual(data["aggregate"]["accuracy"], 0)

    def test_worker_crash_recovery_and_attempt_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = submit(Path(tmp))
            first = store.claim("run", "one", store.pending_dispatches()[0]["id"], "dead-worker", 1)
            self.assertEqual(store.reconcile(now=time.time() + 5), [first])
            self.assertEqual(store.get_run("run")["status"], "queued")
            self.assertFalse(store.save_result(TaskResult("run", "one", status="passed", passed=True, score=1),
                                               first))
            second = store.claim("run", "one", store.pending_dispatches()[0]["id"], "dead-worker", 1)
            self.assertEqual(store.reconcile(now=time.time() + 5), [second])
            data = store.get_run("run")
            self.assertEqual(data["status"], "completed_with_errors")
            self.assertIn("Worker lost", data["tasks"][0]["result"]["error"])
            self.assertEqual([a["status"] for a in data["tasks"][0]["attempts"]],
                             ["abandoned", "abandoned"])
            self.assertEqual(store.pending_dispatches(), [])

    def test_abandoned_attempt_workspace_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = submit(Path(tmp), attempts=1)
            attempt = store.claim("run", "one", store.pending_dispatches()[0]["id"], "dead-worker", 1)
            workspace = Path(tmp) / "workspaces" / f"harness-agent-{attempt[:12]}-orphan"
            workspace.mkdir(parents=True)
            (workspace / "output.txt").write_text("orphan")
            with patch("agentic_harness.scheduling.dispatch.cleanup_attempt_containers"):
                self.assertEqual(recover_once(store, now=time.time() + 5), [attempt])
            self.assertFalse(workspace.exists())
            self.assertEqual(store.pending_cleanup_attempts(), [])

    def test_failed_orphan_cleanup_is_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = submit(Path(tmp), attempts=1)
            attempt = store.claim("run", "one", store.pending_dispatches()[0]["id"], "dead-worker", 1)
            with patch("agentic_harness.scheduling.dispatch.cleanup_attempt_containers",
                       side_effect=RuntimeError("Docker down")):
                recover_once(store, now=time.time() + 5)
            self.assertEqual(store.pending_cleanup_attempts(), [attempt])
            with patch("agentic_harness.scheduling.dispatch.cleanup_attempt_containers") as cleanup:
                self.assertEqual(recover_once(store, now=time.time() + 6), [])
                cleanup.assert_called_once_with(attempt)
            self.assertEqual(store.pending_cleanup_attempts(), [])

    def test_process_task_claims_and_persists_with_fencing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = submit(Path(tmp))
            request = store.pending_dispatches()[0]

            def fake_execute(job):
                self.assertIsNotNone(job.attempt_id)
                result = TaskResult(job.run_id, job.task_id, status="passed", passed=True, score=1.0)
                RunStore(job.database).save_result(result, job.attempt_id)
                return result

            with patch("agentic_harness.scheduling.worker.execute_task", fake_execute):
                self.assertEqual(process_task(str(store.path), "run", "one", request["id"])["status"], "passed")
                self.assertIsNone(process_task(str(store.path), "run", "one", request["id"]))
            self.assertEqual(store.get_run("run")["aggregate"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()


def docker_available():
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "info"], capture_output=True).returncode == 0


@unittest.skipUnless(docker_available(), "Docker daemon unavailable")
class RedisWorkerIntegrationTests(unittest.TestCase):
    def test_redis_rq_worker_process_persists_result_across_restart(self):
        container = subprocess.run(
            ["docker", "run", "-d", "--rm", "-p", "127.0.0.1::6379", "redis:7-alpine"],
            capture_output=True, text=True, check=True).stdout.strip()
        try:
            port_output = subprocess.run(["docker", "port", container, "6379/tcp"],
                                         capture_output=True, text=True, check=True).stdout.strip()
            port = port_output.rsplit(":", 1)[1]
            url = f"redis://127.0.0.1:{port}/0"
            connection = redis_connection(url)
            for _ in range(30):
                try:
                    if connection.ping():
                        break
                except Exception:
                    time.sleep(0.1)
            else:
                self.fail("Redis did not start")
            with tempfile.TemporaryDirectory() as tmp:
                store = submit(Path(tmp), task_ids=("one", "two"), attempts=1)
                queue = task_queue(connection)
                with patch.object(store, "mark_delivered", side_effect=RuntimeError("SQLite write interrupted")):
                    with self.assertRaises(RuntimeError):
                        dispatch_once(store, queue)
                self.assertEqual(dispatch_once(store, queue), 2)
                self.assertEqual(queue.count, 2)
                self.assertEqual(store.pending_dispatches(), [])
                env = {**os.environ, "HARNESS_REDIS_URL": url}
                worker = subprocess.run([sys.executable, "-m", "agentic_harness.cli",
                                         "worker", "--burst"], env=env, capture_output=True,
                                        text=True, timeout=90)
                self.assertEqual(worker.returncode, 0, worker.stderr)
                data = RunStore(store.path).get_run("run")
                self.assertEqual(data["status"], "completed_with_errors")
                self.assertEqual(len(data["tasks"]), 2)
                for task in data["tasks"]:
                    self.assertEqual(len(task["attempts"]), 1)
                    self.assertIn("Benchmark manifest not found", task["result"]["error"])
                subprocess.run(["docker", "restart", container], capture_output=True, check=True)
                self.assertEqual(RunStore(store.path).get_run("run")["status"], "completed_with_errors")
        finally:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)

    def test_two_host_workers_execute_jobs_concurrently(self):
        container = subprocess.run(
            ["docker", "run", "-d", "--rm", "-p", "127.0.0.1::6379", "redis:7-alpine"],
            capture_output=True, text=True, check=True).stdout.strip()
        try:
            port = subprocess.run(["docker", "port", container, "6379/tcp"],
                                  capture_output=True, text=True, check=True).stdout.strip().rsplit(":", 1)[1]
            url = f"redis://127.0.0.1:{port}/0"
            connection = redis_connection(url)
            for _ in range(30):
                try:
                    if connection.ping():
                        break
                except Exception:
                    time.sleep(0.1)
            queue = task_queue(connection)
            for _ in range(2):
                queue.enqueue_call(func="time.sleep", args=(2.5,), timeout=10)
            started = time.monotonic()
            worker = subprocess.run([sys.executable, "-m", "agentic_harness.cli",
                                     "worker", "--concurrency", "2", "--burst"],
                                    env={**os.environ, "HARNESS_REDIS_URL": url},
                                    capture_output=True, text=True, timeout=30)
            elapsed = time.monotonic() - started
            self.assertEqual(worker.returncode, 0, worker.stderr)
            self.assertLess(elapsed, 4.6, worker.stdout)
            self.assertEqual(queue.count, 0)
        finally:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)
