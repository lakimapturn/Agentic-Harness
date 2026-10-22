"""RQ worker entry point for one fenced task attempt."""

import os
import socket
import threading
from dataclasses import replace

from ..core.models import TaskResult
from ..execution.task import execute_task
from ..persistence.sqlite import RunStore

LEASE_SECONDS = 60

def process_task(database: str, run_id: str, task_id: str, dispatch_id: str):
    store = RunStore(database)
    owner = f"{socket.gethostname()}:{os.getpid()}"
    attempt_id = store.claim(run_id, task_id, dispatch_id, owner, LEASE_SECONDS)
    if attempt_id is None:
        return None
    job = replace(store.get_job(run_id, task_id), attempt_id=attempt_id)
    stopped = threading.Event()

    def keep_alive():
        while not stopped.wait(LEASE_SECONDS / 3):
            if not store.heartbeat(attempt_id, LEASE_SECONDS):
                return

    heartbeat = threading.Thread(target=keep_alive, daemon=True)
    heartbeat.start()
    try:
        try:
            result = execute_task(job)
        except Exception as exc:
            result = TaskResult(run_id, task_id, status="error",
                                error=f"Worker execution failed: {type(exc).__name__}: {exc}",
                                error_class="worker")
            store.save_result(result, attempt_id)
        return result.to_dict()
    finally:
        stopped.set()
        heartbeat.join(timeout=2)


