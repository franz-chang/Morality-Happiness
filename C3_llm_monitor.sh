#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" && -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-}"
INTERVAL="${INTERVAL:-1}"
LIMIT="${LIMIT:-200}"
CLEAR_ON_START="${CLEAR_ON_START:-0}"
LOG_DIR="$ROOT_DIR/outputs"
PORT_FILE="$LOG_DIR/server.port"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

api_ready() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
import urllib.request

url = sys.argv[1].rstrip("/") + "/api/defaults"
try:
    with urllib.request.urlopen(url, timeout=1.5) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
}

detect_port() {
  if [[ -n "$PORT" ]]; then
    echo "$PORT"
    return 0
  fi

  if [[ -f "$PORT_FILE" ]]; then
    local recorded_port
    recorded_port="$(cat "$PORT_FILE" 2>/dev/null || true)"
    if [[ "$recorded_port" =~ ^[0-9]+$ ]] && api_ready "http://$HOST:$recorded_port"; then
      echo "$recorded_port"
      return 0
    fi
  fi

  local candidate
  for candidate in 8000 8010 8011 8012 8013 8014 8015 8016 8017 8018 8019 8020; do
    if api_ready "http://$HOST:$candidate"; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

SELECTED_PORT="$(detect_port || true)"
if [[ -z "$SELECTED_PORT" ]]; then
  echo "Morality Happiness server not found. Start it first with: bash C1_start.sh" >&2
  exit 1
fi

BASE_URL="http://$HOST:$SELECTED_PORT"

if [[ "$CLEAR_ON_START" == "1" ]]; then
  "$PYTHON_BIN" - "$BASE_URL/api/llm-traces/clear" <<'PY'
import sys
import urllib.request

request = urllib.request.Request(sys.argv[1], data=b"{}", method="POST")
try:
    urllib.request.urlopen(request, timeout=2).read()
except Exception:
    pass
PY
fi

echo "LLM monitor attached: $BASE_URL"
echo "Only LLM input and output will be printed. Press Ctrl+C to stop."

"$PYTHON_BIN" - "$BASE_URL" "$LIMIT" "$INTERVAL" <<'PY'
import json
import sys
import time
import urllib.parse
import urllib.request

base_url = sys.argv[1].rstrip("/")
limit = max(1, min(200, int(float(sys.argv[2]))))
interval = max(0.2, float(sys.argv[3]))
seen: set[str] = set()


def trace_key(item: dict) -> str:
    return "\0".join(
        str(item.get(key, ""))
        for key in ("timestamp", "provider", "model", "stage", "prompt", "response", "error")
    )


def fetch_items() -> list[dict]:
    url = f"{base_url}/api/llm-traces?{urllib.parse.urlencode({'limit': limit})}"
    with urllib.request.urlopen(url, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    items = payload.get("items", [])
    return items if isinstance(items, list) else []


try:
    while True:
        try:
            items = fetch_items()
        except Exception as exc:
            print(f"\n[monitor] trace API unavailable: {exc}", flush=True)
            time.sleep(interval)
            continue

        for item in items:
            key = trace_key(item)
            if key in seen:
                continue
            seen.add(key)
            print("\n========== LLM INPUT ==========")
            print(str(item.get("prompt", "")))
            print("========== LLM OUTPUT =========")
            print(str(item.get("response", "")))
            print("========== END ================", flush=True)

        time.sleep(interval)
except KeyboardInterrupt:
    print("\nLLM monitor stopped.")
PY
