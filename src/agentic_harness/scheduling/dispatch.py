"""Deliver SQLite outbox rows to RQ and recover abandoned attempts."""

import shutil

from rq.exceptions import DuplicateJobError
from redis.exceptions import RedisError

from ..environments.docker import cleanup_attempt_containers
from ..persistence.sqlite import RunStore


def _matches_dispatch(existing, queue, request: dict, database: str) -> bool:
    """Acknowledge a duplicate only when it is the intended RQ task."""
    return (existing is not None and existing.id == request["id"] and
            existing.origin == queue.name and
            existing.func_name in {
                "agentic_harness.scheduling.worker.process_task",
                "agentic_harness.scheduling.rq.process_task",
                "agentic_harness.dispatch.process_task",
            } and
            tuple(existing.args) == (database, request["run_id"],
                                      request["task_id"], request["id"]))


def dispatch_once(store: RunStore, queue, limit: int = 100) -> int:
    sent = 0
    first_error = None
    for request in store.pending_dispatches(limit):
        try:
            job = store.get_job(request["run_id"], request["task_id"])
            try:
                queue.enqueue_call(
                    func="agentic_harness.scheduling.worker.process_task",
                    args=(str(store.path), request["run_id"], request["task_id"], request["id"]),
                    job_id=request["id"], unique=True, timeout=job.job_timeout or 1200,
                    result_ttl=0, failure_ttl=3600,
                )
            except DuplicateJobError:
                # Redis accepted the previous delivery, but SQLite did not record it.
                existing = queue.fetch_job(request["id"])
                if not _matches_dispatch(existing, queue, request, str(store.path)):
                    raise
            store.mark_delivered(request["id"])
            sent += 1
        except RedisError:
            # A Redis outage affects every row; retry the batch later.
            raise
        except Exception as exc:
            # One malformed or interrupted request must not block other tasks.
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error
    return sent


def recover_once(store: RunStore, *, queue=None, now: float | None = None, delivery_grace: int = 60):
    # Keep jobs that are still waiting in RQ. Replacing their dispatch IDs
    # would make a busy queue's pending messages stale.
    eligible = set()
    if queue is not None:
        try:
            stale_ids = store.stale_deliveries(now=now, delivery_grace=delivery_grace)
            queued_ids = set(queue.get_job_ids()) if stale_ids else set()
            for dispatch_id in stale_ids:
                job = queue.fetch_job(dispatch_id)
                status = job.get_status(refresh=True) if job is not None else None
                if (status not in {"queued", "started", "deferred", "scheduled"} or
                        (status == "queued" and dispatch_id not in queued_ids)):
                    eligible.add(dispatch_id)
        except Exception:
            # Redis failure must not prevent lease recovery, but redelivery waits.
            eligible.clear()
    abandoned = store.reconcile(now=now, delivery_grace=delivery_grace,
                                eligible_dispatches=eligible)
    for attempt_id in store.pending_cleanup_attempts():
        cleaned = False
        try:
            cleanup_attempt_containers(attempt_id)
            cleaned = True
        except Exception:
            # Retry on a later dispatcher pass when Docker is available.
            pass
        workspaces = store.path.parent / "workspaces"
        pattern = f"harness-*-{attempt_id[:12]}-*"
        for workspace in workspaces.glob(pattern):
            shutil.rmtree(workspace, ignore_errors=True)
        if cleaned and not any(workspaces.glob(pattern)):
            store.mark_cleanup_done(attempt_id)
    return abandoned
