"""Create trusted grading data for one trial task."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id")
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()

    tasks = {row["case"]: row for row in json.loads(
        (Path(__file__).parent / "dataset.json").read_text()
    )}
    expected = tasks[args.task_id]["expected"]
    private = Path(args.workspace) / "private"
    private.mkdir(parents=True, exist_ok=True)
    (private / "expected.txt").write_text(expected)


if __name__ == "__main__":
    main()
