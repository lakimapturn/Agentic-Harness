#!/bin/sh
set -eu
harness run --sync --agent agents/mini-swe-agent --benchmark bash-operations --task_ids bash-001
