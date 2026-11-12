#!/usr/bin/env python3
"""Prover CLI (v0): replays recorded GPU telemetry to the verification node.

  prover.py keygen                        make keys/prover.key, print the public key to pin
  prover.py run --trace train-ddp --declare a:inference:0-3 --declare b:inference:4-7
  prover.py run --trace serve-tp4x2 --record rec.jsonl      save the signed windows
  prover.py run --attack replay --replay-file rec.jsonl     send old windows instead (moment 2)
  prover.py reset                          clear the node's history
  prover.py traces                         list trace presets

Every message after `hello` carries prev_hash (hash chain) and is Ed25519-signed with
the prover key. Each window carries the nonce the node just pushed.
"""
import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import twc_common as tc  # noqa: E402

tc.load_env_file(str(HERE / ".env"))

GTE = HERE.parents[2] / "gpu-telemetry-exploration" / "data"          # own 10 Hz H100 traces (DCGM)
SRV = HERE.parent / "telemetry-logs" / "serving-traces-vllm" / "runs"   # 20 Hz H200 serving runs (NVML)
TRACES = {
    "idle":         ("csv", GTE / "b1_telemetry.csv", None, "H100 x8, idle with model loaded (B1)"),
    "infer-tp8":    ("csv", GTE / "i3_telemetry.csv", None, "H100 x8, vLLM TP8 inference, light load (I3)"),
    "train-ddp":    ("csv", GTE / "t1_telemetry.csv", None, "H100 x8, DDP training 3.37B (T1)"),
    "train-fsdp":   ("csv", GTE / "t6_telemetry.csv", None, "H100 x8, FSDP training (T6), low power: rule 3 misses it"),
    "serve-tp4x2":  ("run", SRV / "c2-tp4x2-dspark-20261104T170433Z", "conc_64", "H200 x8, two TP4 vLLM replicas, 64 concurrent (C2)"),
    "serve-tp8":    ("run", SRV / "c1-tp8-dspark-20261103T232811Z", "conc_64", "H200 x8, one TP8 vLLM replica, 64 concurrent (C1): rule 3 false positive"),
}
DT = 0.1  # replay on a 10 Hz grid


def load_trace(name):
    """Return (power [N,G] W, energy [N,G] mJ cumulative) on a 10 Hz grid."""
    kind, path, phase, _ = TRACES[name]
    cache = HERE / ".cache" / f"{name}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["p"], z["e"]
    import pandas as pd
    if kind == "csv":
        df = pd.read_csv(path, usecols=["timestamp", "phase", "gpu", "power_w", "energy_mj"])
        df = df[df.phase == "steady"]
        p = df.pivot_table(index="timestamp", columns="gpu", values="power_w").sort_index()
        e = df.pivot_table(index="timestamp", columns="gpu", values="energy_mj").reindex(p.index).ffill().bfill()
    else:
        ev = [json.loads(l) for l in open(path / "events.jsonl")]
        starts = [x["ts_ns"] for x in ev]
        i = next(k for k, x in enumerate(ev) if x["phase"] == phase)
        t0, t1 = starts[i], starts[i + 1]
        df = pd.read_parquet(path / "gpu_power", columns=["device_index", "ts_ns", "power_instant_mw", "energy_mj"])
        df = df[(df.ts_ns >= t0) & (df.ts_ns < t1)]
        df["bin"] = (df.ts_ns // int(DT * 1e9))
        p = df.pivot_table(index="bin", columns="device_index", values="power_instant_mw") / 1000.0
        e = df.pivot_table(index="bin", columns="device_index", values="energy_mj", aggfunc="max")
        e = e.reindex(p.index).ffill().bfill()
        p = p.interpolate(limit_direction="both")
    p, e = p.to_numpy(float), e.to_numpy(float)
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, p=p, e=e)
    return p, e


def parse_decl(s):
    job, typ, rng = s.split(":")
    if typ not in ("inference", "training", "idle"):
        raise SystemExit(f"bad job type {typ!r}: use inference | training | idle")
    gpus = []
    for part in rng.split(","):
        a, _, b = part.partition("-")
        gpus += list(range(int(a), int(b or a) + 1))
    return {"job_id": job, "type": typ, "gpus": gpus}


def auth_header():
    pw = os.environ.get("SITE_PASSWORD")
    if not pw:
        return {}
    tok = base64.b64encode(f":{pw}".encode()).decode()
    return {"Authorization": f"Basic {tok}"}


def load_key(path):
    if not Path(path).exists():
        raise SystemExit(f"no key at {path}: run `prover.py keygen` first")
    return tc.key_from_seed_b64(Path(path).read_text().strip())


class Replayer:
    """Steps through a trace at wall-clock speed; loops at the end; emits a cumulative energy counter."""
    def __init__(self, p, e, speed):
        self.p, self.e, self.speed = p, e, speed
        self.i, self.t = 0, time.time()
        self.energy = np.zeros(p.shape[1])

    def step(self):
        i, n = self.i, len(self.p)
        j = (i + 1) % n
        dE = self.e[j] - self.e[i] if j else self.p[i] * DT * 1000  # on wrap, fall back to P*dt
        self.energy += np.maximum(dE, 0)
        self.i, self.t = j, self.t + DT
        return self.t, self.p[j]


async def run(args):
    import websockets
    key = load_key(args.key)
    decls = [parse_decl(d) for d in args.declare]
    replayed = [json.loads(l) for l in open(args.replay_file)] if args.attack == "replay" else None
    if args.attack == "replay" and not replayed:
        raise SystemExit("replay file is empty")
    p, e = load_trace(args.trace)
    if args.skip:
        k = int(args.skip / DT)
        p, e = p[k:], e[k:]
    G = p.shape[1]
    print(f"[prover] trace {args.trace}: {G} GPUs, {len(p) * DT:.0f} s (loops) · declared: "
          + ("; ".join(f"{d['job_id']}={d['type']}@{d['gpus']}" for d in decls) or "nothing"))
    rec = open(args.record, "w") if args.record else None
    url = args.node.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + "/ingest"

    async with websockets.connect(url, additional_headers=auth_header(), max_size=2**24) as ws:
        hello = tc.sign(key, {"type": "hello", "prover_id": args.prover_id, "pub": tc.pub_b64(key),
                              "ts_ms": int(time.time() * 1000)})
        await ws.send(json.dumps(hello))
        welcome = json.loads(await ws.recv())
        if welcome.get("type") != "welcome":
            raise SystemExit(f"[prover] node refused: {welcome}")
        print(f"[prover] connected; node key {welcome['node_pub'][:16]}…, window {welcome['window_s']} s")
        last_hash = tc.msg_hash(hello)

        async def send(body):
            nonlocal last_hash
            msg = tc.sign(key, {**body, "prev_hash": last_hash})
            await ws.send(json.dumps(msg))
            last_hash = tc.msg_hash(msg)
            return msg

        if decls:
            await send({"type": "declare", "declarations": decls, "ts_ms": int(time.time() * 1000)})

        rp = Replayer(p, e, args.speed)
        pending = {g: [] for g in range(G)}
        t_start = time.time()

        async def sampler():
            while True:
                t, row = rp.step()
                for g in range(G):
                    pending[g].append([round(t, 2), round(float(row[g]), 1)])
                await asyncio.sleep(DT / args.speed)

        samp = asyncio.create_task(sampler())
        sent = 0
        try:
            while args.duration <= 0 or time.time() - t_start < args.duration:
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
                except asyncio.TimeoutError:
                    continue
                if m.get("type") != "nonce":
                    print(f"[prover] node says: {m}")
                    continue
                if replayed is not None:
                    old = replayed[sent % len(replayed)]
                    await ws.send(json.dumps(old))   # verbatim: valid signature, old nonce, old chain
                    print(f"[prover] REPLAYED old window {old['window_id']} for fresh nonce {m['window_id']}")
                else:
                    samples = {str(g): v for g, v in pending.items()}
                    for g in pending:
                        pending[g] = []
                    energy = {str(g): [round(rp.t, 2), round(float(rp.energy[g]), 1)] for g in range(G)}
                    msg = await send({"type": "window", "window_id": m["window_id"], "nonce": m["nonce"],
                                      "samples": samples, "energy": energy})
                    if rec:
                        rec.write(json.dumps(msg) + "\n"); rec.flush()
                    n = sum(len(v) for v in samples.values())
                    print(f"[prover] window {m['window_id']}: {n} samples, signed, nonce {m['nonce'][:8]}…")
                sent += 1
        finally:
            samp.cancel()
            if rec:
                rec.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node", default=os.environ.get("NODE_URL", "http://localhost:8000"))
    ap.add_argument("--key", default=str(HERE / "keys" / "prover.key"))
    # --node / --key also work after the subcommand (SUPPRESS keeps the top-level value otherwise)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--node", default=argparse.SUPPRESS)
    common.add_argument("--key", default=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("keygen", parents=[common])
    sub.add_parser("reset", parents=[common])
    sub.add_parser("traces", parents=[common])
    r = sub.add_parser("run", parents=[common])
    r.add_argument("--trace", default="serve-tp4x2", choices=list(TRACES))
    r.add_argument("--declare", action="append", default=[], metavar="JOB:TYPE:GPUS",
                   help="e.g. a:inference:0-3 (repeatable). TYPE = inference | training | idle")
    r.add_argument("--prover-id", default="demo-prover")
    r.add_argument("--duration", type=float, default=0, help="seconds; 0 = forever")
    r.add_argument("--speed", type=float, default=1.0)
    r.add_argument("--skip", type=float, default=0, help="skip the first N seconds of the trace")
    r.add_argument("--record", help="save signed windows (jsonl) for a later replay attack")
    r.add_argument("--attack", choices=["replay"])
    r.add_argument("--replay-file")
    args = ap.parse_args()

    if args.cmd == "keygen":
        path = Path(args.key)
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            raise SystemExit(f"{path} exists; refusing to overwrite")
        k = tc.new_key()
        path.write_text(tc.seed_b64(k) + "\n")
        os.chmod(path, 0o600)
        print(f"wrote {path}\nPROVER_PUBKEY={tc.pub_b64(k)}")
    elif args.cmd == "traces":
        for k, (_, path, _, desc) in TRACES.items():
            print(f"{k:13} {'ok ' if path.exists() else 'MISSING'} {desc}")
    elif args.cmd == "reset":
        import httpx
        print(httpx.post(args.node.rstrip("/") + "/api/reset", headers=auth_header()).json())
    else:
        try:
            asyncio.run(run(args))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
