#!/usr/bin/env bash
# usage: stop.sh   -> SIGTERM the logger, wait for its final flush
RUNS_ROOT="${RUNS_ROOT:-/workspace/runs}"
PF="$RUNS_ROOT/logger.pid"
[ -f "$PF" ] || { echo "no pid file"; exit 1; }
PID=$(cat "$PF"); RID=$(cat "$RUNS_ROOT/CURRENT" 2>/dev/null)
kill -TERM "$PID" 2>/dev/null || { echo "pid $PID not running"; rm -f "$PF"; exit 0; }
for i in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
if kill -0 "$PID" 2>/dev/null; then echo "still alive after 30s; sending KILL"; kill -KILL "$PID"; fi
rm -f "$PF"
echo "stopped run=$RID"; tail -8 "$RUNS_ROOT/$RID/logger.log"; du -sh "$RUNS_ROOT/$RID"
