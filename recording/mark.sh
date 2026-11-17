#!/usr/bin/env bash
# usage: mark.sh <phase> [note]   -> appends {ts_ns, mono_ns, phase} to $RUNS_ROOT/$(CURRENT)/events.jsonl
HERE="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$HERE/logger.py" mark "${1:?usage: mark.sh <phase> [note]}" ${2:+--note "$2"}
