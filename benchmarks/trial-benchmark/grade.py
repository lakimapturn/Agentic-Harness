"""Grade a trial submission using data kept out of the agent workspace."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id")
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    answer = workspace / "answer.txt"
    expected = (workspace / "private" / "expected.txt").read_text()
    passed = answer.is_file() and answer.read_text() == expected
    print(json.dumps({
        "task_id": args.task_id,
        "passed": passed,
        "score": float(passed),
        "details": {"answer_present": answer.is_file()},
    }))


if __name__ == "__main__":
    main()
