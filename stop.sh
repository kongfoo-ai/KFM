#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${PID_FILE:-$ROOT_DIR/api.pid}"

pids=()

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${pid}" ]]; then
    pids+=("$pid")
  fi
fi

# Scan for live python api.py processes
while IFS= read -r pid; do
  [[ -n "$pid" ]] && pids+=("$pid")
done < <(pgrep -f "[p]ython([0-9.]*)?[[:space:]].*api\.py" || true)

# Deduplicate
if ((${#pids[@]})); then
  mapfile -t pids < <(printf '%s\n' "${pids[@]}" | awk 'NF && !seen[$0]++')
fi

if ((${#pids[@]} == 0)); then
  echo "no api.py process found"
  rm -f "$PID_FILE"
  exit 0
fi

echo "stopping api.py pid(s): ${pids[*]}"
kill "${pids[@]}" 2>/dev/null || true

# Wait briefly, then force-kill leftovers
for _ in 1 2 3 4 5; do
  alive=()
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done
  ((${#alive[@]} == 0)) && break
  sleep 1
done

for pid in "${pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "force kill pid=$pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
done

rm -f "$PID_FILE"
echo "stopped"
