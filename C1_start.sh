#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
HOST="${HOST:-}"
LOG_DIR="$ROOT_DIR/outputs"
PID_FILE="$LOG_DIR/server.pid"
PORT_FILE="$LOG_DIR/server.port"
LOG_FILE="$LOG_DIR/server.log"

mkdir -p "$LOG_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

CONFIG_HOST="$("$PYTHON_BIN" - <<'PY'
from morality_happiness.settings import server_defaults

print(server_defaults().get("host", "127.0.0.1"))
PY
)"
CONFIG_PORT="$("$PYTHON_BIN" - <<'PY'
from morality_happiness.settings import server_defaults

print(server_defaults().get("port", 8000))
PY
)"

HOST="${HOST:-$CONFIG_HOST}"

is_port_free() {
  local port="$1"
  "$PYTHON_BIN" - "$HOST" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        sys.exit(1)
PY
}

is_project_pid() {
  local pid="${1:-}"
  local cmd
  cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  [[ "$cmd" == *"$ROOT_DIR"* || "$cmd" == *"morality_happiness.server"* ]]
}

server_pid_on_port() {
  local port="$1"
  command -v lsof >/dev/null 2>&1 || return 1
  while IFS= read -r pid; do
    if [[ "$pid" =~ ^[0-9]+$ ]] && is_project_pid "$pid"; then
      echo "$pid"
      return 0
    fi
  done < <(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  return 1
}

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    existing_cmd="$(ps -p "$existing_pid" -o command= 2>/dev/null || true)"
    if [[ "$existing_cmd" == *"morality_happiness.server"* ]]; then
      running_port="$(cat "$PORT_FILE" 2>/dev/null || echo "${PORT:-8000}")"
      display_host="$HOST"
      if [[ "$display_host" == "0.0.0.0" ]]; then
        display_host="127.0.0.1"
      fi
      echo "Morality Happiness server is already running."
      echo "PID: $existing_pid"
      echo "URL: http://$display_host:$running_port/"
      echo "Log: $LOG_FILE"
      exit 0
    fi
  fi
  rm -f "$PID_FILE" "$PORT_FILE"
fi

if [[ -n "${PORT:-}" ]]; then
  SELECTED_PORT="$PORT"
  if ! is_port_free "$SELECTED_PORT"; then
    echo "Requested port $SELECTED_PORT is already in use." >&2
    exit 1
  fi
else
  SELECTED_PORT=""
  for candidate in "$CONFIG_PORT" 8000 8010 8011 8012 8013 8014 8015 8016 8017 8018 8019 8020; do
    if is_port_free "$candidate"; then
      SELECTED_PORT="$candidate"
      break
    fi
  done

  if [[ -z "$SELECTED_PORT" ]]; then
    echo "No free port found in 8000, 8010-8020. Set PORT manually, e.g. PORT=8030 ./start.sh" >&2
    exit 1
  fi
fi

if [[ ! -f "$ROOT_DIR/frontend/dist/index.html" ]]; then
  echo "Warning: frontend/dist/index.html not found. The server will still start, but the built dashboard is missing." >&2
  echo "Run: cd frontend && npm install && npm run build" >&2
fi

display_host="$HOST"
if [[ "$display_host" == "0.0.0.0" ]]; then
  display_host="127.0.0.1"
fi

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  echo "Starting Morality Happiness server in foreground."
  echo "URL: http://$display_host:$SELECTED_PORT/"
  exec "$PYTHON_BIN" -m morality_happiness.server \
    --host "$HOST" \
    --port "$SELECTED_PORT"
fi

SCREEN_NAME="morality_happiness_${SELECTED_PORT}"
if command -v screen >/dev/null 2>&1; then
  screen -S "$SCREEN_NAME" -X quit >/dev/null 2>&1 || true
  screen -dmS "$SCREEN_NAME" bash -lc 'cd "$1" && exec "$2" -m morality_happiness.server --host "$3" --port "$4" > "$5" 2>&1' _ "$ROOT_DIR" "$PYTHON_BIN" "$HOST" "$SELECTED_PORT" "$LOG_FILE"
  server_pid=""
else
  nohup "$PYTHON_BIN" -m morality_happiness.server \
    --host "$HOST" \
    --port "$SELECTED_PORT" \
    >"$LOG_FILE" 2>&1 &
  server_pid="$!"
  echo "$server_pid" > "$PID_FILE"
fi
echo "$SELECTED_PORT" > "$PORT_FILE"

sleep 1

if [[ -n "$server_pid" ]] && ! kill -0 "$server_pid" 2>/dev/null; then
  echo "Server failed to start. Recent log:" >&2
  tail -40 "$LOG_FILE" >&2 || true
  rm -f "$PID_FILE" "$PORT_FILE"
  exit 1
fi

ready=0
for _ in {1..24}; do
  if [[ -n "$server_pid" ]] && ! kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  if "$PYTHON_BIN" - "http://$display_host:$SELECTED_PORT/api/defaults" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=1.0) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
  then
    ready=1
    break
  fi
  sleep 0.25
done

if [[ "$ready" != "1" ]]; then
  echo "Server process started but did not become ready. Recent log:" >&2
  tail -40 "$LOG_FILE" >&2 || true
  rm -f "$PID_FILE" "$PORT_FILE"
  exit 1
fi

actual_pid="$(server_pid_on_port "$SELECTED_PORT" || true)"
if [[ -z "$actual_pid" ]]; then
  actual_pid="$server_pid"
fi
if [[ -z "$actual_pid" ]]; then
  echo "Server became ready, but the project PID could not be detected." >&2
  tail -40 "$LOG_FILE" >&2 || true
  rm -f "$PID_FILE" "$PORT_FILE"
  exit 1
fi
echo "$actual_pid" > "$PID_FILE"

echo "Morality Happiness server started."
echo "PID: $actual_pid"
echo "URL: http://$display_host:$SELECTED_PORT/"
echo "Log: $LOG_FILE"
echo "Stop: kill \$(cat \"$PID_FILE\")"
