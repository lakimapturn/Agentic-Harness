#!/bin/sh
set -eu
harness run --sync --agent agents/mini-swe-agent --benchmark python-tasks --task_ids py-001
