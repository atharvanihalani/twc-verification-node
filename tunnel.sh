#!/bin/bash
# Keep a reverse SSH tunnel up so a GPU box can reach the verification node running on this Mac.
# The box's 127.0.0.1:18999 -> this Mac's localhost:8000. Reconnects whenever the link drops;
# ServerAlive* makes ssh notice a stalled link within ~30 s instead of hanging forever.
#
#   ./tunnel.sh <ssh-port> root@<box-ip>
PORT=${1:?usage: tunnel.sh <ssh-port> <user@host>}
HOST=${2:?usage: tunnel.sh <ssh-port> <user@host>}
while true; do
  # A dead session can leave the box still holding port 18999, which blocks the new tunnel.
  # Free it first (only kills whatever is listening on 18999, i.e. our own stale tunnel end).
  ssh -o ConnectTimeout=10 -p "$PORT" "$HOST" \
    'pid=$(ss -ltnp 2>/dev/null | grep ":18999 " | grep -o "pid=[0-9]*" | head -1 | cut -d= -f2); [ -n "$pid" ] && kill "$pid"' \
    >/dev/null 2>&1
  ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
      -R 18999:localhost:8000 -p "$PORT" "$HOST" 2>&1 | grep -v -i "vast.ai\|have fun\|agents\|TCP_NODELAY"
  echo "[tunnel] down at $(date +%T), reconnecting in 3 s"
  sleep 3
done
