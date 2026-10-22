"""SQLite authority for runs, tasks, attempts, and dispatch requests."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..core.models import TaskJob, TaskResult


TERMINAL = {"passed", "failed", "agent_error", "error", "timeout"}
# Retry only failures that point to the execution infrastructure. Setup,
# verifier, resolution, agent failures, and agent time limits are terminal.
RETRYABLE_ERROR_CLASSES = {"environment", "environment_timeout", "worker"}


class RunStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, benchmark TEXT NOT NULL, agent TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    config_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    run_id TEXT NOT NULL, task_id TEXT NOT NULL, status TEXT NOT NULL,
                    stage TEXT NOT NULL DEFAULT 'pending', result_json TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    job_json TEXT, current_attempt_id TEXT, dispatch_id TEXT,
                    attempts_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (run_id, task_id),
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, task_id TEXT NOT NULL,
                    worker_id TEXT NOT NULL, status TEXT NOT NULL,
                    started_at REAL NOT NULL, heartbeat_at REAL NOT NULL,
                    lease_until REAL NOT NULL, ended_at REAL,
                    error_class TEXT, duration REAL, logs_json TEXT, result_json TEXT,
                    cleanup_status TEXT NOT NULL DEFAULT 'done'
                );
                CREATE TABLE IF NOT EXISTS outbox (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, task_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', created_at REAL NOT NULL,
                    delivered_at REAL, deliveries INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS outbox_pending ON outbox(status, created_at);
                CREATE INDEX IF NOT EXISTS attempts_lease ON attempts(status, lease_until);
            """)
            # The schema transaction serializes migration inspection and ALTER TABLE.
            self._add_columns(db, "runs", {"config_json": "TEXT NOT NULL DEFAULT '{}'"})
            self._add_columns(db, "tasks", {
                "job_json": "TEXT", "current_attempt_id": "TEXT", "dispatch_id": "TEXT",
                "attempts_count": "INTEGER NOT NULL DEFAULT 0",
            })
            self._add_columns(db, "attempts", {"cleanup_status": "TEXT NOT NULL DEFAULT 'done'"})

    @staticmethod
    def _add_columns(db, table: str, columns: dict[str, str]):
        existing = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
        for name, definition in columns.items():
            if name not in existing:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA busy_timeout=30000")
            # SQLite may reject a simultaneous WAL mode switch immediately,
            # despite busy_timeout. Wait for the other initializer to finish.
            deadline = time.monotonic() + 30
            while True:
                try:
                    if db.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                        db.execute("PRAGMA journal_mode=WAL")
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc) or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
            db.execute("PRAGMA foreign_keys=ON")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def create_run(self, run_id: str, benchmark: str, agent: str, task_ids: list[str]):
        """Compatibility for explicit synchronous Part 1 runs."""
        with self._connect() as db:
            db.execute("INSERT INTO runs(id, benchmark, agent) VALUES(?, ?, ?)", (run_id, benchmark, agent))
            db.executemany("INSERT INTO tasks(run_id, task_id, status) VALUES(?, ?, 'pending')",
                           [(run_id, task_id) for task_id in task_ids])

    def submit(self, jobs: list[TaskJob], *, max_attempts: int = 2, config: dict | None = None):
        if not jobs or max_attempts < 1:
            raise ValueError("Submission needs tasks and at least one attempt")
        first = jobs[0]
        if any((j.run_id, j.benchmark, j.agent, j.database) !=
               (first.run_id, first.benchmark, first.agent, first.database) for j in jobs):
            raise ValueError("Jobs must belong to one run")
        now = time.time()
        with self._connect() as db:
            db.execute("INSERT INTO runs(id, benchmark, agent, config_json) VALUES(?, ?, ?, ?)",
                       (first.run_id, first.benchmark, first.agent,
                        json.dumps({**(config or {}), "max_attempts": max_attempts})))
            for job in jobs:
                dispatch_id = uuid.uuid4().hex
                db.execute("INSERT INTO tasks(run_id, task_id, status, job_json, dispatch_id) "
                           "VALUES(?, ?, 'queued', ?, ?)",
                           (job.run_id, job.task_id, json.dumps(job.to_dict()), dispatch_id))
                db.execute("INSERT INTO outbox(id, run_id, task_id, created_at) VALUES(?, ?, ?, ?)",
                           (dispatch_id, job.run_id, job.task_id, now))

    def get_job(self, run_id: str, task_id: str) -> TaskJob:
        with self._connect() as db:
            row = db.execute("SELECT job_json FROM tasks WHERE run_id=? AND task_id=?",
                             (run_id, task_id)).fetchone()
        if not row or row[0] is None:
            raise KeyError(f"Queued task not found: {run_id}/{task_id}")
        return TaskJob(**json.loads(row[0]))

    def pending_dispatches(self, limit: int = 100) -> list[dict]:
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT o.* FROM outbox o JOIN tasks t "
                              "ON t.run_id=o.run_id AND t.task_id=o.task_id "
                              "WHERE o.status='pending' AND t.status='queued' "
                              "AND t.dispatch_id=o.id ORDER BY o.created_at LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def mark_delivered(self, dispatch_id: str):
        with self._connect() as db:
            db.execute("UPDATE outbox SET status='delivered', delivered_at=?, deliveries=deliveries+1 "
                       "WHERE id=? AND status='pending'", (time.time(), dispatch_id))

    def claim(self, run_id: str, task_id: str, dispatch_id: str, worker_id: str,
              lease_seconds: int = 60) -> str | None:
        now, attempt_id = time.time(), uuid.uuid4().hex
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            updated = db.execute("UPDATE tasks SET status='running', stage='resolve', "
                                 "current_attempt_id=?, attempts_count=attempts_count+1, "
                                 "updated_at=CURRENT_TIMESTAMP WHERE run_id=? AND task_id=? "
                                 "AND status='queued' AND dispatch_id=?",
                                 (attempt_id, run_id, task_id, dispatch_id)).rowcount
            if not updated:
                return None
            db.execute("INSERT INTO attempts(id, run_id, task_id, worker_id, status, "
                       "started_at, heartbeat_at, lease_until) VALUES(?, ?, ?, ?, 'running', ?, ?, ?)",
                       (attempt_id, run_id, task_id, worker_id, now, now, now + lease_seconds))
        return attempt_id

    def heartbeat(self, attempt_id: str, lease_seconds: int = 60) -> bool:
        now = time.time()
        with self._connect() as db:
            return bool(db.execute("UPDATE attempts SET heartbeat_at=?, lease_until=? "
                                   "WHERE id=? AND status='running' AND EXISTS "
                                   "(SELECT 1 FROM tasks t WHERE t.run_id=attempts.run_id "
                                   "AND t.task_id=attempts.task_id AND t.current_attempt_id=attempts.id "
                                   "AND t.status='running')",
                                   (now, now + lease_seconds, attempt_id)).rowcount)

    def set_stage(self, job: TaskJob, stage: str) -> bool:
        with self._connect() as db:
            if job.attempt_id:
                return bool(db.execute("UPDATE tasks SET stage=?, updated_at=CURRENT_TIMESTAMP "
                                       "WHERE run_id=? AND task_id=? AND status='running' "
                                       "AND current_attempt_id=?",
                                       (stage, job.run_id, job.task_id, job.attempt_id)).rowcount)
            return bool(db.execute("UPDATE tasks SET status='running', stage=?, updated_at=CURRENT_TIMESTAMP "
                                   "WHERE run_id=? AND task_id=? AND job_json IS NULL",
                                   (stage, job.run_id, job.task_id)).rowcount)

    def save_result(self, result: TaskResult, attempt_id: str | None = None) -> bool:
        encoded = json.dumps(result.to_dict())
        with self._connect() as db:
            if not attempt_id:
                return bool(db.execute("UPDATE tasks SET status=?, result_json=?, "
                                       "updated_at=CURRENT_TIMESTAMP WHERE run_id=? AND task_id=? "
                                       "AND job_json IS NULL",
                                       (result.status, encoded, result.run_id, result.task_id)).rowcount)
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT t.current_attempt_id, t.status, t.attempts_count, r.config_json "
                             "FROM tasks t JOIN runs r ON r.id=t.run_id "
                             "WHERE t.run_id=? AND t.task_id=?", (result.run_id, result.task_id)).fetchone()
            if not row or row[0] != attempt_id or row[1] != "running":
                return False
            now = time.time()
            error_class = result.error_class or (
                result.status if result.status in {"error", "agent_error", "timeout"} else None)
            db.execute("UPDATE attempts SET status='completed', ended_at=?, error_class=?, duration=?, "
                       "logs_json=?, result_json=? WHERE id=? AND status='running'",
                       (now, error_class, result.duration, json.dumps(result.logs), encoded, attempt_id))
            if (result.status in {"error", "timeout"} and
                    error_class in RETRYABLE_ERROR_CLASSES and
                    row[2] < json.loads(row[3]).get("max_attempts", 1)):
                dispatch_id = uuid.uuid4().hex
                db.execute("UPDATE tasks SET status='queued', stage='retry', result_json=NULL, "
                           "current_attempt_id=NULL, dispatch_id=?, updated_at=CURRENT_TIMESTAMP "
                           "WHERE run_id=? AND task_id=? AND current_attempt_id=?",
                           (dispatch_id, result.run_id, result.task_id, attempt_id))
                db.execute("INSERT INTO outbox(id, run_id, task_id, created_at) VALUES(?, ?, ?, ?)",
                           (dispatch_id, result.run_id, result.task_id, now))
            else:
                db.execute("UPDATE tasks SET status=?, stage='done', result_json=?, "
                           "updated_at=CURRENT_TIMESTAMP WHERE run_id=? AND task_id=? "
                           "AND current_attempt_id=?",
                           (result.status, encoded, result.run_id, result.task_id, attempt_id))
        return True

    def stale_deliveries(self, *, now: float | None = None, delivery_grace: int = 60) -> list[str]:
        now = time.time() if now is None else now
        with self._connect() as db:
            rows = db.execute("SELECT o.id FROM outbox o JOIN tasks t "
                              "ON t.run_id=o.run_id AND t.task_id=o.task_id "
                              "WHERE o.status='delivered' AND o.delivered_at<? "
                              "AND t.status='queued' AND t.dispatch_id=o.id",
                              (now - delivery_grace,)).fetchall()
        return [row[0] for row in rows]

    def reconcile(self, *, now: float | None = None, delivery_grace: int = 60,
                  eligible_dispatches: set[str] | None = None) -> list[str]:
        """Expire abandoned attempts and replace unclaimed queue messages."""
        now = time.time() if now is None else now
        abandoned = []
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT a.id, a.run_id, a.task_id, t.attempts_count, r.config_json "
                              "FROM attempts a JOIN tasks t ON t.run_id=a.run_id AND t.task_id=a.task_id "
                              "JOIN runs r ON r.id=a.run_id WHERE a.status='running' AND a.lease_until<? "
                              "AND t.current_attempt_id=a.id AND t.status='running'", (now,)).fetchall()
            for attempt_id, run_id, task_id, count, config_json in rows:
                abandoned.append(attempt_id)
                db.execute("UPDATE attempts SET status='abandoned', ended_at=?, "
                           "error_class='worker_lost', cleanup_status='pending' WHERE id=?",
                           (now, attempt_id))
                if count < json.loads(config_json).get("max_attempts", 1):
                    dispatch_id = uuid.uuid4().hex
                    db.execute("UPDATE tasks SET status='queued', stage='retry', current_attempt_id=NULL, "
                               "dispatch_id=?, result_json=NULL, updated_at=CURRENT_TIMESTAMP "
                               "WHERE run_id=? AND task_id=? AND current_attempt_id=?",
                               (dispatch_id, run_id, task_id, attempt_id))
                    db.execute("INSERT INTO outbox(id, run_id, task_id, created_at) VALUES(?, ?, ?, ?)",
                               (dispatch_id, run_id, task_id, now))
                else:
                    result = TaskResult(run_id, task_id, status="error",
                                        error="Worker lost; attempt lease expired",
                                        error_class="worker_lost")
                    db.execute("UPDATE tasks SET status='error', stage='done', result_json=?, "
                               "updated_at=CURRENT_TIMESTAMP WHERE run_id=? AND task_id=? "
                               "AND current_attempt_id=?", (json.dumps(result.to_dict()), run_id, task_id, attempt_id))
            lost = db.execute("SELECT o.id, o.run_id, o.task_id FROM outbox o JOIN tasks t "
                              "ON t.run_id=o.run_id AND t.task_id=o.task_id "
                              "WHERE o.status='delivered' AND o.delivered_at<? "
                              "AND t.status='queued' AND t.dispatch_id=o.id",
                              (now - delivery_grace,)).fetchall()
            for old_id, run_id, task_id in lost:
                if eligible_dispatches is not None and old_id not in eligible_dispatches:
                    continue
                new_id = uuid.uuid4().hex
                db.execute("UPDATE outbox SET status='superseded' WHERE id=?", (old_id,))
                db.execute("UPDATE tasks SET dispatch_id=? WHERE run_id=? AND task_id=? AND dispatch_id=?",
                           (new_id, run_id, task_id, old_id))
                db.execute("INSERT INTO outbox(id, run_id, task_id, created_at) VALUES(?, ?, ?, ?)",
                           (new_id, run_id, task_id, now))
        return abandoned

    def pending_cleanup_attempts(self) -> list[str]:
        with self._connect() as db:
            return [row[0] for row in db.execute(
                "SELECT id FROM attempts WHERE cleanup_status='pending' ORDER BY ended_at").fetchall()]

    def mark_cleanup_done(self, attempt_id: str):
        with self._connect() as db:
            db.execute("UPDATE attempts SET cleanup_status='done' WHERE id=? AND cleanup_status='pending'",
                       (attempt_id,))

    def get_run(self, run_id: str) -> dict:
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            run = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if run is None:
                raise KeyError(f"Run not found: {run_id}")
            tasks = db.execute("SELECT * FROM tasks WHERE run_id=? ORDER BY task_id", (run_id,)).fetchall()
            attempts = db.execute("SELECT * FROM attempts WHERE run_id=? ORDER BY started_at, id", (run_id,)).fetchall()
        by_task = {}
        for row in attempts:
            item = dict(row)
            logs_json = item.pop("logs_json")
            result_json = item.pop("result_json")
            item["logs"] = json.loads(logs_json) if logs_json else {}
            item["result"] = json.loads(result_json) if result_json else None
            by_task.setdefault(item["task_id"], []).append(item)
        results = []
        for row in tasks:
            value = json.loads(row["result_json"]) if row["result_json"] else None
            results.append({"task_id": row["task_id"], "status": row["status"],
                            "stage": row["stage"], "result": value,
                            "attempts": by_task.get(row["task_id"], [])})
        completed = [item for item in results if item["status"] in TERMINAL and item["result"] is not None]
        passed = sum(item["status"] == "passed" for item in completed)
        total = len(results)
        errors = sum(item["status"] in {"error", "timeout", "agent_error"} for item in results)
        failed = sum(item["status"] == "failed" for item in results)
        aggregate = {"total": total, "completed": len(completed), "passed": passed,
                     "failed": failed, "errors": errors, "accuracy": passed / total if total else 0.0,
                     "mean_score": sum(item["result"]["score"] for item in completed) / total if total else 0.0}
        status = ("running" if any(item["status"] in {"running", "pending"} for item in results) else
                  "queued" if len(completed) < total else
                  "completed_with_errors" if errors else
                  "completed_with_failures" if failed else "completed")
        return {"run_id": run_id, "benchmark": run["benchmark"], "agent": run["agent"],
                "created_at": run["created_at"], "configuration": json.loads(run["config_json"]),
                "status": status, "aggregate": aggregate, "tasks": results}
