"""Submission, status, dispatcher, and host RQ worker CLI."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import click

from .agents import AgentError, resolve_agent
from .benchmarks.config import BenchmarkError, resolve_benchmark
from .scheduling.dispatch import dispatch_once, recover_once
from .scheduling.transport import redis_connection, task_queue
from .execution.task import execute_task
from .core.models import TaskJob
from .persistence.sqlite import RunStore
from rq import Worker
from rq.serializers import JSONSerializer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _paths():
    root = Path(os.environ.get("HARNESS_HOME", PROJECT_ROOT / ".harness")).resolve()
    return root / "runs.sqlite3", root / "logs"


@click.group()
def cli():
    """Run agentic benchmarks."""


@cli.command()
@click.option("--agent", required=True, help="Agent adapter path, e.g. agents/mini-swe-agent")
@click.option("--benchmark", required=True, help="Benchmark directory name")
@click.option("--task_ids", default=None, help="Comma-separated task IDs (default: all)")
@click.option("--sync", is_flag=True, help="Run in this process (Part 1 behavior)")
@click.option("--max-attempts", type=click.IntRange(min=1, max=10), default=2, show_default=True)
@click.option("--docker-memory", default="1g", show_default=True)
@click.option("--docker-cpus", type=click.FloatRange(min=0.1), default=1.0, show_default=True)
@click.option("--docker-pids", type=click.IntRange(min=16), default=256, show_default=True)
def run(agent: str, benchmark: str, task_ids: str | None, sync: bool, max_attempts: int,
        docker_memory: str, docker_cpus: float, docker_pids: int):
    """Run an agent against a benchmark"""
    try:
        resolved = resolve_benchmark(benchmark, PROJECT_ROOT / "benchmarks")
        resolve_agent(agent)
        ids = [part.strip() for part in task_ids.split(",")] if task_ids else list(resolved.tasks)
        if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
            raise BenchmarkError("Task IDs must be nonempty and unique")
        for task_id in ids:
            resolved.get_task(task_id)
    except (BenchmarkError, AgentError, OSError, KeyError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    database, logs = _paths()
    run_id = uuid.uuid4().hex
    store = RunStore(database)
    if sync:
        store.create_run(run_id, benchmark, agent, ids)
        click.echo(f"Run ID: {run_id}")
        for task_id in ids:
            job = TaskJob(run_id, benchmark, task_id, agent, str(PROJECT_ROOT / "benchmarks"),
                          str(database), str(logs))
            result = execute_task(job)
            click.echo(f"{task_id}: {result.status} score={result.score:.2f}" +
                       (f" error={result.error}" if result.error else ""))
        aggregate = store.get_run(run_id)["aggregate"]
        click.echo(json.dumps(aggregate, sort_keys=True))
        if aggregate["errors"]:
            click.get_current_context().exit(1)
        return

    # Freeze benchmark files and non-secret configuration before the durable commit.
    snapshot_root = database.parent / "snapshots" / run_id
    snapshot_root.mkdir(parents=True)
    try:
        shutil.copytree(resolved.directory, snapshot_root / benchmark)
        model_name = os.environ.get("MSWEA_MODEL_NAME", "")
        timeout = (360 + 2 * resolved.timeout("setup") + resolved.timeout("agent") +
                   resolved.timeout("verifier") + 180)
        jobs = [TaskJob(run_id, benchmark, task_id, agent, str(snapshot_root),
                        str(database), str(logs), model_name=model_name,
                        docker_memory=docker_memory, docker_cpus=docker_cpus,
                        docker_pids_limit=docker_pids, job_timeout=timeout) for task_id in ids]
        store.submit(jobs, max_attempts=max_attempts,
                     config={"model_name": model_name, "docker_memory": docker_memory,
                             "docker_cpus": docker_cpus, "docker_pids_limit": docker_pids,
                             "benchmark_snapshot": str(snapshot_root), "job_timeout": timeout})
    except Exception as exc:
        shutil.rmtree(snapshot_root, ignore_errors=True)
        raise click.ClickException(f"Submission failed: {exc}") from exc
    click.echo(f"Run ID: {run_id}")
    click.echo(f"Queued {len(ids)} task(s)")


@cli.command()
@click.option("--once", is_flag=True, help="Dispatch and recover once, then exit")
@click.option("--interval", type=click.FloatRange(min=0.1), default=3.0, show_default=True)
def dispatcher(once: bool, interval: float):
    """Recover abandoned tasks and deliver SQLite outbox rows to Redis."""
    store = RunStore(_paths()[0])
    while True:
        try:
            queue = task_queue()
            recover_once(store, queue=queue)
            dispatch_once(store, queue)
        except Exception as exc:
            click.echo(f"Dispatcher retrying: {type(exc).__name__}: {exc}", err=True)
            if once:
                raise SystemExit(1) from exc
        if once:
            return
        time.sleep(interval)


@cli.command()
@click.option("--concurrency", type=click.IntRange(min=1, max=32), default=1, show_default=True)
@click.option("--burst", is_flag=True, help="Exit after the queue is empty")
@click.option("--child", is_flag=True, hidden=True)
def worker(concurrency: int, burst: bool, child: bool):
    """Run a bounded pool of host RQ worker processes."""
    if concurrency > 1 and not child:
        children = []
        def stop_pool(*_):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, stop_pool)
        try:
            for _ in range(concurrency):
                children.append(subprocess.Popen([sys.executable, "-m", "agentic_harness.cli",
                                                  "worker", "--child", "--concurrency", "1",
                                                  *(["--burst"] if burst else [])]))
            returncode = max(child_process.wait() for child_process in children)
        except KeyboardInterrupt:
            returncode = 0
        finally:
            for child_process in children:
                if child_process.poll() is None:
                    child_process.terminate()
            for child_process in children:
                child_process.wait()
            signal.signal(signal.SIGTERM, previous)
        if returncode:
            raise SystemExit(returncode)
        return
    connection = redis_connection()
    connection.ping()
    rq_worker = Worker([task_queue(connection)], connection=connection, serializer=JSONSerializer)
    rq_worker.work(burst=burst)


@cli.command()
@click.option("--run_id", required=True)
def status(run_id: str):
    """Check the status of a benchmark run"""
    try:
        data = RunStore(_paths()[0]).get_run(run_id)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps({key: data[key] for key in ("run_id", "status", "aggregate", "tasks")}, indent=2))


@cli.command()
@click.option("--run_id", required=True)
@click.option("--format", "output_format", type=click.Choice(["json", "table", "summary"]), default="json")
def results(run_id: str, output_format: str):
    """Get results from a completed run"""
    try:
        data = RunStore(_paths()[0]).get_run(run_id)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    if output_format == "json":
        click.echo(json.dumps(data, indent=2))
    elif output_format == "summary":
        click.echo(f"Run {run_id}: {data['status']}\n{json.dumps(data['aggregate'], indent=2)}")
    else:
        click.echo("TASK ID\tSTATUS\tSCORE\tDURATION\tERROR")
        for item in data["tasks"]:
            value = item["result"] or {}
            click.echo(f"{item['task_id']}\t{item['status']}\t{value.get('score', '')}\t"
                       f"{value.get('duration', '')}\t{value.get('error') or ''}")
        click.echo(f"Accuracy: {data['aggregate']['accuracy']:.1%}")


def main():
    cli()


if __name__ == "__main__":
    main()
