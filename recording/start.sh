#!/usr/bin/env bash
# usage: start.sh <run_id> [notes...]   (on the box)
set -euo pipefail
RID="${1:?usage: start.sh <run_id> [notes]}"; shift || true
NOTES="${*:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS_ROOT="${RUNS_ROOT:-/workspace/runs}"
PY="${PY:-/venv/main/bin/python}"
CPUS="${LOGGER_CPUS:-188,190}"      # two NUMA-0 cores at the top of the range; override if needed
RD="$RUNS_ROOT/$RID"
if [ -f "$RUNS_ROOT/logger.pid" ] && kill -0 "$(cat "$RUNS_ROOT/logger.pid")" 2>/dev/null; then
  echo "logger already running (pid $(cat "$RUNS_ROOT/logger.pid"), run $(cat "$RUNS_ROOT/CURRENT")); run stop.sh first" >&2; exit 1
fi
mkdir -p "$RD"
echo "$RID" > "$RUNS_ROOT/CURRENT"
setsid nohup taskset -c "$CPUS" "$PY" -u "$HERE/logger.py" run "$RID" --notes "$NOTES" \
  > "$RD/logger.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$RUNS_ROOT/logger.pid"; echo "$PID" > "$RD/logger.pid"
sleep 3
if kill -0 "$PID" 2>/dev/null; then echo "started run=$RID pid=$PID cpus=$CPUS"; head -5 "$RD/logger.log"
else echo "logger died:"; cat "$RD/logger.log"; exit 1; fi
