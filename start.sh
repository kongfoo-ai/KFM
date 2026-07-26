#!/usr/bin/env bash
set -euo pipefail

export GPU_MEMORY_UTILIZATION=0.5 

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PID_FILE="${PID_FILE:-$ROOT_DIR/api.pid}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/api.log}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "api.py already running (pid=$old_pid). Use ./stop.sh first."
    exit 1
  fi
  rm -f "$PID_FILE"
fi

# Match an already-running api.py even without pidfile
existing="$(pgrep -f "[p]ython([0-9.]*)?[[:space:]].*api\.py" || true)"
if [[ -n "$existing" ]]; then
  echo "api.py already running (pid(s): $existing). Use ./stop.sh first."
  exit 1
fi

nohup "$PYTHON_BIN" api.py >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
echo "started api.py pid=$(cat "$PID_FILE") log=$LOG_FILE"
tail -f api.log
