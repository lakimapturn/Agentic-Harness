"""Manifest backed benchmark loader. A custom adapter can implement the same methods."""

import json
import importlib
from pathlib import Path
from typing import Any

import yaml

from ..core.models import Task


class BenchmarkError(ValueError):
    pass


class ConfigBenchmark:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        manifest = self.directory / "benchmark.yaml"
        if not manifest.is_file():
            raise BenchmarkError(f"Benchmark manifest not found: {manifest}")
        self.config = yaml.safe_load(manifest.read_text())
        if not isinstance(self.config, dict):
            raise BenchmarkError(f"Invalid benchmark manifest: {manifest}")
        dataset = (self.directory / self.config["dataset"]).resolve()
        if not dataset.is_relative_to(self.directory):
            raise BenchmarkError("Dataset must be inside the benchmark directory")
        rows = json.loads(dataset.read_text())
        if not isinstance(rows, list):
            raise BenchmarkError("Dataset must contain a list")
        fields = self.config["fields"]
        self.tasks: dict[str, Task] = {}
        for row in rows:
            task = Task(
                id=str(row[fields["id"]]),
                prompt=str(row[fields["prompt"]]),
                image=str(row.get(fields.get("image", "docker_image")) or self.config["default_image"]),
                data=row,
            )
            if task.id in self.tasks:
                raise BenchmarkError(f"Duplicate task ID: {task.id}")
            self.tasks[task.id] = task

    def get_task(self, task_id: str) -> Task:
        try:
            return self.tasks[task_id]
        except KeyError:
            raise BenchmarkError(f"Task not found: {task_id}") from None

    def command(self, stage: str, task: Task, workspace: str = "/workspace") -> list[str]:
        spec = self.config[stage]
        values = {key: value for key, value in task.data.items() if isinstance(value, (str, int, float))}
        values.update({"task_id": task.id, "workspace": workspace})
        return [str(part).format_map(values) for part in spec["command"]]

    def timeout(self, stage: str) -> int:
        return int(self.config.get("timeouts", {}).get(stage, 120))


def resolve_benchmark(name: str, root: Path) -> ConfigBenchmark:
    # Name lookup is intentionally narrow; arbitrary paths can be added by a registry later.
    if not name or name in {".", ".."} or Path(name).name != name:
        raise BenchmarkError(f"Invalid benchmark name: {name}")
    directory = root / name
    manifest = directory / "benchmark.yaml"
    if manifest.is_file():
        config = yaml.safe_load(manifest.read_text())
        adapter = config.get("adapter") if isinstance(config, dict) else None
        if adapter:
            module_name, class_name = adapter.rsplit(".", 1)
            implementation = getattr(importlib.import_module(module_name), class_name)
            return implementation(directory)
    return ConfigBenchmark(directory)
