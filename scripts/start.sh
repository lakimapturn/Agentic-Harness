#!/usr/bin/env bash

# Start or manage the local harness tmux session. Credentials are inherited by
# the dedicated tmux server from this shell; they are never command arguments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOCKET="agentic-harness"
SESSION="agentic-harness"
HARNESS="$ROOT/.venv/bin/harness"
PYTHON="$ROOT/.venv/bin/python"
ACTION="${1:-start}"

die() {
    printf 'Error: %s\n' "$1" >&2
    return 1
}

has_session() {
    tmux -L "$SOCKET" has-session -t "$SESSION" 2>/dev/null
}

has_server() {
    tmux -L "$SOCKET" show-environment -g >/dev/null 2>&1
}

server_sessions() {
    tmux -L "$SOCKET" list-sessions -F '#{session_name}' 2>/dev/null || true
}

check_tmux() {
    if ! command -v tmux >/dev/null 2>&1; then
        printf 'Error: tmux is required. Install it with `brew install tmux` (macOS) or your system package manager (Linux).\n' >&2
        return 1
    fi
}

check_start_requirements() {
    if ! command -v docker >/dev/null 2>&1; then
        printf 'Error: Docker is required. Install Docker Desktop or Docker Engine, then rerun make start.\n' >&2
        return 1
    fi
    if ! docker compose version >/dev/null 2>&1; then
        printf 'Error: Docker Compose v2 is required. Install or enable the Docker Compose plugin, then rerun make start.\n' >&2
        return 1
    fi
    if ! docker info >/dev/null 2>&1; then
        printf 'Error: the Docker daemon is not running. Start Docker Desktop or the Docker service, then rerun make start.\n' >&2
        return 1
    fi
    if [[ ! -x "$HARNESS" || ! -x "$PYTHON" ]]; then
        printf 'Error: the project environment is missing. Run `make install` first.\n' >&2
        return 1
    fi
}

check_terminal() {
    if [[ ! -t 0 || ! -t 1 ]]; then
        printf 'Error: make start/attach needs an interactive terminal. Run it from a terminal window.\n' >&2
        return 1
    fi
    if [[ "${TERM:-}" == "dumb" || -z "${TERM:-}" ]]; then
        printf 'Error: this terminal does not advertise screen control. Set TERM=xterm-256color and retry.\n' >&2
        return 1
    fi
}

check_openrouter_credentials() {
    local model="${MSWEA_MODEL_NAME:-}"
    if [[ "$model" == openrouter/* && -z "${OPENROUTER_API_KEY:-}" ]]; then
        printf 'Error: OPENROUTER_API_KEY is required for the selected OpenRouter model. Export it in this shell, then rerun make start.\n' >&2
        return 1
    fi
}

redis_reachable() {
    "$PYTHON" - <<'PY' >/dev/null 2>&1
import os
from redis import Redis
Redis.from_url(os.environ.get("HARNESS_REDIS_URL", "redis://localhost:6379/0"),
               socket_connect_timeout=1, socket_timeout=1).ping()
PY
}

start_redis() {
    local before
    before="$(docker compose ps --status running -q redis 2>/dev/null || true)"
    if ! docker compose up -d redis; then
        printf 'Error: Redis did not start. Check `docker compose logs redis` and try again.\n' >&2
        return 1
    fi
    if [[ -z "$before" ]]; then
        mkdir -p "$ROOT/.harness/startup"
        : > "$ROOT/.harness/startup/redis-started-by-harness"
    fi

    local attempt
    for attempt in {1..40}; do
        if redis_reachable; then
            return 0
        fi
        sleep 0.5
    done
    printf 'Error: Redis started but did not become reachable at HARNESS_REDIS_URL (or localhost:6379). Check the URL and `docker compose logs redis`.\n' >&2
    if [[ -f "$ROOT/.harness/startup/redis-started-by-harness" ]]; then
        docker compose stop redis >/dev/null 2>&1 || true
        rm -f "$ROOT/.harness/startup/redis-started-by-harness"
    fi
    return 1
}

tmux_environment_value() {
    local name="$1" line
    TMUX_ENV_VALUE=""
    line="$(tmux -L "$SOCKET" show-environment -t "$SESSION" "$name" 2>/dev/null || true)"
    case "$line" in
        "$name"=*) TMUX_ENV_VALUE="${line#*=}"; return ;;
        "-$name" ) return ;;
    esac
    # Session variables fall back to the server's global environment.
    line="$(tmux -L "$SOCKET" show-environment -g "$name" 2>/dev/null || true)"
    case "$line" in
        "$name"=*) TMUX_ENV_VALUE="${line#*=}" ;;
    esac
}

session_environment_is_current() {
    local names name current_value stale=0
    names="$( {
        compgen -e
        tmux -L "$SOCKET" show-environment -g 2>/dev/null
    } | awk '
        {
            name = $0
            if (substr(name, 1, 1) == "-") name = substr(name, 2)
            sub(/=.*/, "", name)
            if (name ~ /^(MSWEA_MODEL_NAME|HARNESS_HOME|HARNESS_REDIS_URL|GOOGLE_APPLICATION_CREDENTIALS)$/ ||
                name ~ /_API_KEY$/ || name ~ /_TOKEN$/ || name ~ /_API_BASE$/ ||
                name ~ /_API_VERSION$/ || name ~ /_ENDPOINT$/ || name ~ /_PROJECT$/ ||
                name ~ /_LOCATION$/ || name ~ /^AWS_/ || name ~ /^AZURE_/) print name
        }
    ' | sort -u )"

    while IFS= read -r name; do
        [[ -n "$name" ]] || continue
        tmux_environment_value "$name"
        current_value="${!name-}"
        if [[ "$current_value" != "$TMUX_ENV_VALUE" ]]; then
            stale=1
            break
        fi
    done <<< "$names"
    [[ "$stale" -eq 0 ]]
}

warn_stale_session() {
    printf 'Error: the existing harness session has different model, credential, or runtime environment settings.\n' >&2
    printf 'It was left untouched so active tasks can finish. Run `make stop`, wait for any active task to drain, then export the desired settings and run `make start`.\n' >&2
    printf 'To reconnect with the existing settings, run `make attach` after restoring the original environment.\n' >&2
}

wait_for_window() {
    local window="$1" timeout="$2" started="$SECONDS" panes dead
    while (( SECONDS - started < timeout )); do
        if ! tmux -L "$SOCKET" list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null | grep -Fxq "$window"; then
            return 0
        fi
        panes="$(tmux -L "$SOCKET" list-panes -t "$SESSION:$window" -F '#{pane_dead}' 2>/dev/null || true)"
        dead=1
        while IFS= read -r line; do
            [[ "$line" == 1 ]] || dead=0
        done <<< "$panes"
        [[ -n "$panes" && "$dead" -eq 1 ]] && return 0
        sleep 0.25
    done
    return 1
}

stop_services() {
    local timeout="${HARNESS_STOP_TIMEOUT:-30}"
    if [[ ! "$timeout" =~ ^[0-9]+$ ]] || (( timeout < 1 )); then
        die 'HARNESS_STOP_TIMEOUT must be a positive number of seconds.'
        return 1
    fi

    if ! has_session; then
        printf 'No harness tmux session is running.\n'
        # A session may have been killed while its dedicated tmux server stayed
        # alive. Replace only this private server so new workers inherit the
        # current shell environment.
        if has_server; then
            if [[ -n "$(server_sessions)" ]]; then
                printf 'Error: the dedicated tmux socket contains a different session. It was left untouched.\n' >&2
                return 1
            fi
            tmux -L "$SOCKET" kill-server
        fi
    else
        # Ctrl-C asks the CLI worker pool to signal each RQ worker. RQ performs
        # a warm shutdown and waits for its current workhorse/task to finish.
        tmux -L "$SOCKET" send-keys -t "$SESSION:dispatcher" C-c 2>/dev/null || true
        tmux -L "$SOCKET" send-keys -t "$SESSION:workers" C-c 2>/dev/null || true
        if ! wait_for_window dispatcher "$timeout" || ! wait_for_window workers "$timeout"; then
            printf 'The dispatcher stopped, but workers are still draining. A task may be active; no worker or container was killed.\n' >&2
            printf 'The tmux session and Redis were left running. Re-run `make stop` after the task finishes.\n' >&2
            return 1
        fi
        tmux -L "$SOCKET" kill-session -t "$SESSION" 2>/dev/null || true
        printf 'Harness dispatcher and workers stopped gracefully.\n'
    fi

    if [[ -f "$ROOT/.harness/startup/redis-started-by-harness" ]]; then
        if docker compose stop redis; then
            rm -f "$ROOT/.harness/startup/redis-started-by-harness"
            printf 'Harness-managed Redis stopped; its Compose volume remains intact.\n'
        else
            printf 'The harness session stopped, but Redis did not stop. Run `docker compose stop redis` manually.\n' >&2
            return 1
        fi
    fi
}

attach_session() {
    check_terminal || return 1
    check_tmux || return 1
    if ! has_session; then
        printf 'No harness session exists. Run `make start` to create one.\n' >&2
        return 1
    fi
    if ! session_environment_is_current; then
        warn_stale_session
        return 1
    fi
    tmux -L "$SOCKET" select-window -t "$SESSION:cli" 2>/dev/null || true
    if ! tmux -L "$SOCKET" attach-session -t "$SESSION"; then
        printf 'Error: tmux could not attach. Check terminal support; try TERM=xterm-256color.\n' >&2
        return 1
    fi
}

start_session() {
    check_terminal || return 1
    check_tmux || return 1
    if has_session; then
        attach_session
        return $?
    fi
    check_start_requirements || return 1

    # A dead session can leave the private server with old credentials.
    if has_server; then
        if [[ -n "$(server_sessions)" ]]; then
            printf 'Error: the dedicated tmux socket contains a different session. It was left untouched.\n' >&2
            return 1
        fi
        tmux -L "$SOCKET" kill-server
    fi

    check_openrouter_credentials || return 1
    start_redis || return 1

    local root_q harness_q shell_q
    printf -v root_q '%q' "$ROOT"
    printf -v harness_q '%q' "$HARNESS"
    shell_q="${SHELL:-/bin/bash}"
    printf -v shell_q '%q' "$shell_q"

    # The first tmux server on this private socket inherits this process's
    # environment. Child windows and all three workers inherit the same values.
    if ! tmux -L "$SOCKET" new-session -d -s "$SESSION" -n dispatcher -c "$ROOT" \
            "exec $harness_q dispatcher"; then
        if has_session; then
            attach_session
            return $?
        fi
        printf 'Error: could not create the tmux session. Check tmux permissions and try again.\n' >&2
        return 1
    fi
    tmux -L "$SOCKET" set-option -t "$SESSION" remain-on-exit on
    tmux -L "$SOCKET" new-window -t "$SESSION" -n workers -c "$ROOT" \
        "exec $harness_q worker --concurrency 3"
    tmux -L "$SOCKET" new-window -t "$SESSION" -n cli -c "$ROOT" \
        "source $root_q/.venv/bin/activate && exec $shell_q -i"
    tmux -L "$SOCKET" select-window -t "$SESSION:cli"

    printf 'Harness started in tmux session %s. Detach with Ctrl-b, then d.\n' "$SESSION"
    printf 'Opening the CLI window. Use `harness run`, `status`, and `results` there.\n'
    if ! tmux -L "$SOCKET" attach-session -t "$SESSION"; then
        printf 'Error: tmux could not attach. Check terminal support; try TERM=xterm-256color.\n' >&2
        return 1
    fi
}

case "$ACTION" in
    start)
        start_session
        ;;
    attach)
        check_tmux || exit 1
        attach_session
        ;;
    stop)
        if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
            die 'Docker Compose is required to stop harness-managed Redis.'
            exit 1
        fi
        if ! command -v tmux >/dev/null 2>&1; then
            die 'tmux is required to stop harness-managed services.'
            exit 1
        fi
        stop_services
        ;;
    *)
        die "Unknown action '$ACTION'. Use start, attach, or stop."
        exit 2
        ;;
esac
