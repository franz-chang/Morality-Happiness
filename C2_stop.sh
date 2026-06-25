#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" && -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

LOG_DIR="$ROOT_DIR/outputs"
PID_FILE="$LOG_DIR/server.pid"
PORT_FILE="$LOG_DIR/server.port"

PIDS=""
PORTS=""

add_pid() {
  local pid="${1:-}"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && is_project_pid "$pid" && ! contains_item "$PIDS" "$pid"; then
    PIDS="${PIDS}${pid}
"
  fi
}

add_port() {
  local port="${1:-}"
  if [[ "$port" =~ ^[0-9]+$ ]] && ! contains_item "$PORTS" "$port"; then
    PORTS="${PORTS}${port}
"
  fi
}

contains_item() {
  local values="$1"
  local needle="$2"
  while IFS= read -r item; do
    [[ "$item" == "$needle" ]] && return 0
  done <<< "$values"
  return 1
}

is_project_pid() {
  local pid="${1:-}"
  local cmd
  cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  [[ "$cmd" == *"$ROOT_DIR"* || "$cmd" == *"morality_happiness.server"* ]]
}

add_project_pids_on_port() {
  local port="${1:-}"
  [[ "$port" =~ ^[0-9]+$ ]] || return 0
  command -v lsof >/dev/null 2>&1 || return 0
  while IFS= read -r pid; do
    add_pid "$pid"
  done < <(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
}

add_ports_for_pid() {
  local pid="${1:-}"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  command -v lsof >/dev/null 2>&1 || return 0
  while IFS= read -r port; do
    add_port "$port"
  done < <(
    lsof -Pan -p "$pid" -iTCP -sTCP:LISTEN 2>/dev/null \
      | awk 'NR > 1 { split($9, parts, ":"); print parts[length(parts)] }'
  )
}

if [[ -f "$PID_FILE" ]]; then
  add_pid "$(cat "$PID_FILE" 2>/dev/null || true)"
fi

if [[ -f "$PORT_FILE" ]]; then
  add_project_pids_on_port "$(cat "$PORT_FILE" 2>/dev/null || true)"
fi

if command -v pgrep >/dev/null 2>&1; then
  while IFS= read -r pid; do
    add_pid "$pid"
  done < <(pgrep -f "morality_happiness.server" || true)
fi

if command -v ps >/dev/null 2>&1; then
  while IFS= read -r line; do
    pid="${line%% *}"
    cmd="${line#* }"
    if [[ "$cmd" == *"$ROOT_DIR"* && ( "$cmd" == *"morality_happiness.server"* || "$cmd" == *"vite"* ) ]]; then
      add_pid "$pid"
    fi
  done < <(ps -axo pid=,command=)
fi

while IFS= read -r pid; do
  [[ -z "$pid" ]] && continue
  add_ports_for_pid "$pid"
done <<< "$PIDS"

if [[ -z "$PIDS" ]]; then
  echo "No Morality Happiness processes found."
else
  echo "Stopping Morality Happiness processes:"
  echo "$PIDS" | sed '/^$/d; s/^/  PID /'
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    kill "$pid" 2>/dev/null || true
  done <<< "$PIDS"

  sleep 1

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
      echo "Force stopping PID $pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done <<< "$PIDS"
fi

rm -f "$PID_FILE" "$PORT_FILE"

echo "Cleaned project PID/port files."
if [[ -n "$PORTS" ]]; then
  echo "Released project ports:"
  echo "$PORTS" | sed '/^$/d; s/^/  /'
else
  echo "No project-owned listening ports found."
fi
