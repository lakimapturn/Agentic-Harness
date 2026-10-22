"""Copy agent submissions into a trusted grader workspace with path confinement."""

import os
import shutil
from pathlib import Path

from ..environments.types import EnvironmentError


def _copy_submission(source: Path, target: Path, paths: list[str],
                     protected_paths: list[str] | None = None):
    """Copy task files while confining source and destination paths to their workspaces."""
    source_root = source.resolve()
    target_root = target.resolve()

    def safe_relative(name: str) -> Path:
        relative = Path(name)
        if not name or relative.is_absolute() or ".." in relative.parts:
            raise EnvironmentError(f"Invalid submission path: {name}")
        return relative

    protected = [safe_relative(name) for name in (protected_paths or [])]
    if Path(".") in protected:
        raise EnvironmentError("Cannot protect the entire grader workspace")

    def is_protected(relative: Path) -> bool:
        return any(relative == name or name in relative.parents for name in protected)

    def copy_item(relative: Path):
        if is_protected(relative):
            raise EnvironmentError(f"Submission overlaps a protected grader path: {relative}")

        item = source_root / relative
        source_part = source_root
        for part in relative.parts:
            source_part = source_part / part
            if source_part.is_symlink():
                if not source_part.exists():
                    raise EnvironmentError(f"Submission contains a dangling symlink: {relative}")
                if not source_part.resolve().is_relative_to(source_root):
                    raise EnvironmentError(f"Submission escapes agent workspace: {relative}")
        if not item.exists():
            return
        resolved = item.resolve()
        if not resolved.is_relative_to(source_root):
            raise EnvironmentError(f"Submission escapes agent workspace: {relative}")

        destination = target_root / relative
        for depth in range(1, len(relative.parts)):
            ancestor = target_root.joinpath(*relative.parts[:depth])
            if ancestor.is_symlink():
                raise EnvironmentError(f"Submission targets a grader symlink: {relative}")
            if ancestor.exists() and not ancestor.is_dir():
                raise EnvironmentError(f"Submission conflicts with grader file: {relative}")
        if not destination.parent.resolve().is_relative_to(target_root):
            raise EnvironmentError(f"Submission escapes grader workspace: {relative}")
        if destination.is_symlink():
            raise EnvironmentError(f"Submission would overwrite a grader symlink: {relative}")

        if item.is_symlink():
            if is_protected(resolved.relative_to(source_root)):
                raise EnvironmentError(f"Submission symlink targets a protected grader path: {relative}")
            if destination.exists():
                raise EnvironmentError(f"Submission would overwrite grader file: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            link = item.readlink()
            if link.is_absolute():
                link = Path(os.path.relpath(target_root / resolved.relative_to(source_root),
                                            destination.parent))
            destination.symlink_to(link)
        elif item.is_dir():
            if destination.exists() and not destination.is_dir():
                raise EnvironmentError(f"Submission conflicts with grader file: {relative}")
            destination.mkdir(parents=True, exist_ok=True)
            for child in item.iterdir():
                copy_item(relative / child.name)
        elif item.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
        else:
            raise EnvironmentError(f"Unsupported submission file: {relative}")

    for name in paths:
        relative = safe_relative(name)
        if relative == Path("."):
            for item in source_root.iterdir():
                copy_item(Path(item.name))
        else:
            copy_item(relative)


