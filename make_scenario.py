#!/usr/bin/env python3
"""make_scenario.py: build the Replay demo's scripted scenario (replays/scenario-1.json).

A pretend 16-GPU cluster (node-a, node-b, 8 GPUs each) for 36 minutes, stitched from our own
recorded 10 Hz H100 traces (gpu-telemetry-exploration/data, see prover.py TRACES):

    job 101  inference    node-a  13:30 -> 14:15 (declared)   telemetry: vLLM TP8, light load (I3)
    job 102  pretraining  node-b  13:30 -> 14:00 (ran, ended) telemetry: DDP training (T1)

Three staged violations (the telemetry lies about nothing; the *declarations* are incomplete):
    13:52-13:58  job 102's training quietly spills onto node-a (the inference pod's GPUs).
                 node-a gets the SAME DDP samples as node-b (+ small sensor noise): one job, two
                 partitions -> training_shaped on 101, (formerly) cross_partition_sync 101~102
    14:01-14:04  node-b keeps training after job 102 ended -> undeclared_load
    14:04-       node-b idle again

The rules are the real ones in node/rules/, evaluated every 10 s on the trailing 60 s, offline.
The page just plays the result back (client-side clock), so the Fly trial's 5-min machine stop
can't reset it. Run:  .venv/bin/python make_scenario.py
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from node import rules
from prover import load_trace

HERE = Path(__file__).resolve().parent
OUT = HERE / "replays" / "scenario-1.json"
DT = 0.1                      # 10 Hz
DUR_S = 36 * 60
EVAL_EVERY_S = 10
START = datetime(2026, 11, 6, 13, 30, 0)
M = 60                        # seconds per minute, for readability below

NODES = {"node-a": [f"a{i}" for i in range(8)], "node-b": [f"b{i}" for i in range(8)]}
GPUS = NODES["node-a"] + NODES["node-b"]
JOBS = [  # declared_end = what Slurm was told (time limit); end = when it actually ended (None = running)
    {"id": "101", "name": "serve-pod", "type": "inference", "partition": "serve", "node": "node-a",
     "gpus": NODES["node-a"], "start": 0, "declared_end": 45 * M, "end": None},
    {"id": "102", "name": "pretrain-run", "type": "pretraining", "partition": "train", "node": "node-b",
     "gpus": NODES["node-b"], "start": 5, "declared_end": 30 * M, "end": 30 * M},
]
SPILL = (22 * M, 28 * M)       # job 102's training also on node-a
UNDECLARED = (31 * M, 34 * M)  # node-b busy with no job
STORY = [
    {"from": SPILL[0], "to": SPILL[1], "text": "Pretraining spills onto the inference pod's GPUs"},
    {"from": UNDECLARED[0], "to": UNDECLARED[1], "text": "node-b keeps training after job 102 ended"},
]
RULES = [  # what the page shows
    {"key": "undeclared_load", "name": "undeclared_load", "text": rules.RULE_TEXT["undeclared_load"]},
    {"key": "training_shaped", "name": "training_shaped", "text": rules.RULE_TEXT["training_shaped"]},
    {"key": "boundary_traffic", "name": "boundary_traffic", "text": rules.RULE_TEXT["boundary_traffic"]},
]


def tiled(name, n, offset=0):
    p, _ = load_trace(name)
    idx = (np.arange(n) + offset) % len(p)
    return p[idx]


def build_power():
    n = int(DUR_S / DT)
    rng = np.random.default_rng(1)
    infer, train, idle = tiled("infer-tp8", n), tiled("train-ddp", n), tiled("idle", n)
    t = np.arange(n) * DT
    a, b = infer.copy(), idle.copy()
    job102 = (t >= JOBS[1]["start"]) & (t < JOBS[1]["end"])
    b[job102] = train[job102]
    spill = (t >= SPILL[0]) & (t < SPILL[1])
    a[spill] = train[spill] + rng.normal(0, 1.5, (spill.sum(), 8))   # same job, so the same samples (+ a little sensor noise;
                                                                     # 6 W of noise already drowns the lockstep)
    und = (t >= UNDECLARED[0]) & (t < UNDECLARED[1])
    b[und] = train[und]
    return t, np.hstack([a, b])                                      # [n, 16] in GPUS order


def declarations_at(ts):
    """Jobs whose declared run overlaps the trailing window [ts - 60 s, ts]. (Using only the jobs alive
    at ts made undeclared_load fire for the minute after a job ended, on its own last 60 s of work.)"""
    lo = ts - rules.WINDOW_S
    return [{"job_id": j["id"], "type": j["type"], "gpus": j["gpus"]}
            for j in JOBS if j["start"] <= ts and (j["end"] if j["end"] is not None else j["declared_end"]) > lo]


def evaluate_all(t, P):
    evals = []
    win = int(rules.WINDOW_S / DT)
    for ts in range(EVAL_EVERY_S, DUR_S + 1, EVAL_EVERY_S):
        k = int(round(ts / DT))
        lo = max(0, k - win)
        buffers = {g: list(zip(t[lo:k], P[lo:k, j])) for j, g in enumerate(GPUS)}
        r = rules.evaluate(buffers, declarations_at(ts), GPUS, t[k - 1])
        hits = r["hits"]
        evals.append({"t": ts, "status": r["status"], "hits": hits, "features": r["features"]})
    return evals


def intervals(evals):
    """Merge consecutive evaluations with the same hit into violation intervals (for the overlays)."""
    open_, out = {}, []
    for e in evals:
        keys = set()
        for h in e["hits"]:
            if h["rule"] == "undeclared_load":
                key = ("undeclared_load", "node-a" if str(h["gpus"][0]).startswith("a") else "node-b")
            elif h["rule"] == "training_shaped":
                key = ("training_shaped", h["jobs"][0])
            else:
                key = (h["rule"], "~".join(h["jobs"]))
            keys.add(key)
            if key not in open_:
                open_[key] = {"rule": key[0], "subject": key[1], "start": e["t"], "end": e["t"], "gpus": set()}
            open_[key]["end"] = e["t"]
            if key[0] == "undeclared_load":
                open_[key]["gpus"].add(h["detail"].split()[1])
        for key in [k for k in open_ if k not in keys]:
            out.append(open_.pop(key))
    out += open_.values()
    for v in out:
        if v["rule"] == "undeclared_load":
            v["gpus"] = sorted(v["gpus"])
        elif v["rule"] == "training_shaped":
            v["jobs"] = [v["subject"]]; v["gpus"] = next(j["gpus"] for j in JOBS if j["id"] == v["subject"])
        else:
            v["jobs"] = v["subject"].split("~"); v["gpus"] = []
        v["start"] -= EVAL_EVERY_S   # the hit covers the eval period leading up to it
    return sorted(out, key=lambda v: v["start"])


def sacct_text(at_s):
    """What `sacct` would have printed at this moment (the declarations, as reported)."""
    ts = lambda s: (START + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%S")
    rows = []
    for j in JOBS:
        if j["start"] > at_s:
            continue
        done = j["end"] is not None and j["end"] <= at_s
        rows.append(f"{j['id']:<6} {j['name']:>14} {j['partition']:>10} {('COMPLETED' if done else 'RUNNING'):>10} "
                    f"{ts(j['start']):>19} {(ts(j['end']) if done else 'Unknown'):>19} {j['node']:>8} "
                    f"{'cpu=32,gres/gpu=8':>22} {'declared:' + j['type']:>24}")
    head = ("$ sacct -X --format=JobID,JobName%14,Partition,State,Start,End,NodeList,AllocTRES%22,Comment%24\n"
            f"{'JobID':<6} {'JobName':>14} {'Partition':>10} {'State':>10} {'Start':>19} {'End':>19} {'NodeList':>8} "
            f"{'AllocTRES':>22} {'Comment':>24}\n"
            f"{'-' * 6} {'-' * 14} {'-' * 10} {'-' * 10} {'-' * 19} {'-' * 19} {'-' * 8} {'-' * 22} {'-' * 24}\n")
    return head + "\n".join(rows)


def main():
    t, P = build_power()
    evals = evaluate_all(t, P)
    viol = intervals(evals)
    changes = sorted({0} | {j["start"] for j in JOBS} | {j["end"] for j in JOBS if j["end"] is not None})
    out = {
        "name": "scenario-1",
        "about": "16 recorded H100 GPUs on 2 pretend nodes, 36 min; declarations from Slurm; 3 staged violations",
        "start": START.isoformat(), "duration_s": DUR_S, "eval_every_s": EVAL_EVERY_S,
        "nodes": [{"name": n, "gpus": g} for n, g in NODES.items()],
        "jobs": JOBS, "rules": RULES, "story": STORY, "violations": viol,
        "evals": [{"t": e["t"], "hits": e["hits"], "jobs": e["features"].get("jobs", {}),
                   "cross": e["features"].get("cross_corr", {}), "warming": e["status"] == "warming_up"} for e in evals],
        "sacct": [{"at": s, "text": sacct_text(s)} for s in changes],
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB): {len(evals)} evaluations, {len(viol)} violations")
    for v in viol:
        print(f"  {v['rule']:<22} {v['subject']:<10} {v['start'] / 60:5.1f} -> {v['end'] / 60:5.1f} min")


if __name__ == "__main__":
    main()
