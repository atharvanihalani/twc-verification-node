"""Replay feed: plays a recorded run into a node's /api/collect, as if it were collector.py.

It reads a file made by make_replay.py, stamps every row with the current time, and POSTs
one second of rows per batch, on the same schedule and in the same body shape as the real
collector. The node can't tell the difference (v0 has no signatures or nonces).
At the end of the file it loops; energy keeps counting up across loops, like a real counter.

    python3 replay_feed.py --node http://127.0.0.1:8000            # password from TWC_PASSWORD
    python3 replay_feed.py --node https://twc-verification-node.fly.dev --file replays/c1-tp8-c64.jsonl.gz

The node starts this itself from Debug → Live telemetry → "Play a recorded trace".
To fake extra boxes for testing (e.g. a 4-GPU H100 box starting 3 min into the recording):

    python3 replay_feed.py --host fake-h100 --gpus 4 --gpu-name "NVIDIA H100 NVL" --offset 180

Like collector.py, it says hello (what it is) before its first batch.
Stdlib only.
"""
import argparse
import base64
import gzip
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def log(msg):
    print(f"[replay] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default="http://127.0.0.1:8000")
    ap.add_argument("--file", default=str(HERE / "replays" / "c1-tp8-c64.jsonl.gz"))
    ap.add_argument("--host", help="name shown on the node (default: from the file)")
    ap.add_argument("--exit-with-parent", action="store_true",
                    help="stop when the process that started us goes away (used by the node)")
    ap.add_argument("--gpus", type=int, help="only play the first N GPUs (pretend to be a smaller box)")
    ap.add_argument("--gpu-name", help="report this GPU model instead of the recorded one")
    ap.add_argument("--offset", type=float, default=0, help="start this many seconds into the recording")
    a = ap.parse_args()

    with gzip.open(a.file, "rt") as f:
        header = json.loads(f.readline())
        ticks = [json.loads(line) for line in f]
    host = a.host or f"replay · {header['name']}"
    gpus, dt = header["gpus"], header["dt"]
    n_gpu = min(a.gpus or len(gpus), len(gpus))
    gpus = {str(g): a.gpu_name or gpus[str(g)] for g in range(n_gpu)}
    ticks = [{"p": t["p"][:n_gpu], **({"s": t["s"][:n_gpu]} if "s" in t else {})} for t in ticks]
    per_batch = round(1 / dt)                     # 10 ticks = 1 s per POST, like the collector

    # energy: shift each loop up by what one pass through the file consumed, so the counter never goes backwards
    e = [[s.get("energy_mj") for s in t["s"]] for t in ticks if "s" in t]
    first = [next((r[g] for r in e if r[g] is not None), 0) for g in range(len(gpus))]
    last = [next((r[g] for r in reversed(e) if r[g] is not None), 0) for g in range(len(gpus))]
    # (last - first) spans len(e) - 1 seconds; one pass lasts len(e) seconds, so scale up by one step
    # (otherwise the counter would stand still for one second at every loop seam)
    per_loop = [max(0, round((l - f) * len(e) / max(1, len(e) - 1))) for f, l in zip(first, last)]

    url = a.node.rstrip("/") + "/api/collect"
    headers = {"Content-Type": "application/json"}
    pw = os.environ.get("TWC_PASSWORD", "")
    if pw:
        headers["Authorization"] = "Basic " + base64.b64encode(f":{pw}".encode()).decode()
    parent = os.getppid()
    hello = {"host": host, "collector": "replay_feed.py", "replay": {"label": header["label"], "about": header["about"]},
             "source": header["source"], "offset_s": a.offset, "gpus": [{"index": int(g), "name": n} for g, n in gpus.items()],
             "rates_hz": {"power": round(1 / dt), "slow": 1}}
    need_hello = True
    log(f"{header['source']}: {len(ticks)} ticks x {len(gpus)} GPUs, looping; sending to {url} as '{host}'")

    k = int(a.offset / dt) // per_batch * per_batch   # whole seconds, so the 1 Hz rows stay on the batch's first tick
    t0, sent, refused = time.time() - k * dt, 0, 0
    next_send = time.monotonic() + 1
    while True:
        rows = []
        for _ in range(per_batch):
            loop, i = divmod(k, len(ticks))
            tick, t = ticks[i], round(t0 + k * dt, 3)
            for g, p in enumerate(tick["p"]):
                row = {"t": t, "gpu": g, "power_w": p}
                if "s" in tick:
                    row.update(tick["s"][g])
                    if "energy_mj" in row:
                        row["energy_mj"] += loop * per_loop[g]
                rows.append(row)
            k += 1
        # wait until the newest row's time has passed, so we never send samples "from the future"
        time.sleep(max(0, next_send - time.monotonic()))
        next_send += 1
        body = json.dumps({"host": host, "gpus": gpus, "samples": rows}).encode()
        try:
            if need_hello:   # introduce ourselves first; the node asks again if it forgets us (restart)
                urllib.request.urlopen(urllib.request.Request(a.node.rstrip("/") + "/api/hello",
                                                              json.dumps(hello).encode(), headers), timeout=5).read()
                need_hello = False
                log("said hello")
            r = json.loads(urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=5).read() or b"{}")
            need_hello = bool(r.get("need_hello"))
            sent += 1
            if sent == 1 or sent % 60 == 0:
                log(f"sent batch {sent} (tick {k % len(ticks)}/{len(ticks)}, loop {k // len(ticks)})")
        except urllib.error.HTTPError as ex:   # e.g. 429: the node already shows 4 boxes
            refused += 1
            if refused % 30 == 1:
                log(f"send refused: HTTP {ex.code} {ex.read().decode(errors='ignore')[:200]}")
        except Exception as ex:   # node restarting etc.: drop this second, keep the clock running
            log(f"send failed: {ex}")
        if a.exit_with_parent and os.getppid() != parent:
            log("parent process gone, exiting")
            return


if __name__ == "__main__":
    main()
