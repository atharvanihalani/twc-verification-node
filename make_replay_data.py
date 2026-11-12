"""Turn the 2026-11-11 recordings into what the Replay demo page plays.

    .venv/bin/python make_replay_data.py            # writes node/static/replay-data/
    .venv/bin/python make_replay_data.py --table    # only print what the rules say about each recording

Input: run directories written by the serving-traces logger (parquet: power at 20 Hz, GPU counters at 10 Hz;
events.jsonl), in ../telemetry-logs/serving-traces-vllm/runs/.
Output:
    <id>.json     one recording, whole length: every GPU's readings once a second, as integers
                  (power W; tensor cores and HBM bandwidth in 0.1 %; NVLink and PCIe in 0.001 GB/s)
    <id>.hz10.json the same five GPU readings ten times a second, for the last-minute strips; no network row
    index.json    the recordings (what ran, phases), and the step-2 scenarios with the verdicts of the real
                  rules (node/rules), computed here over the 20 Hz power and the once-a-second counters

Units follow collector.py: NVML's per-second PCIe and NVLink numbers are divided by 1000 and called GB/s.
The counters are averaged over each second, which is what a collector polling once a second would read.
"""
import datetime, json, re, sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "node"))
import rules  # noqa: E402

RUNS = HERE.parent / "telemetry-logs" / "serving-traces-vllm" / "runs"
OUT = HERE / "node" / "static" / "replay-data"
SCALE = {"power": 1, "tensor": 10, "hbm_bw": 10, "nvlink": 1000, "pcie_tx": 1000, "net_tx": 1000}

# id -> run directory. Node names are the rental's own: both nodes of one rental, 8 x H200 each.
RECS = {
    "p1b": "p1b-base-b4-20261111T1151Z",
    "p2": "p2-selective-b4-20261111T0954Z",
    "p3": "p3-base-b1-20261111T0954Z",
    "i1": "i1-tp8-20261111T1021Z",
    "i2": "i2-ep8-20261111T1037Z",
    "i3p": "i3-pd-2node-20261111T1123Z-prefill",
    "i3d": "i3-pd-2node-20261111T1123Z-decode",
}
FINE_HZ = 10   # the rate of the last-minute strips: the counters' own polling rate
AGENTX_FROM = {"i3p": "i3d"}
PHASE_NAMES = {"idle": "idle", "train_start": "training starts", "train_stop": "training stopped", "idle_after": "idle",
               "server_start": "server loading the model", "server_ready": "server ready, no load", "pd_smoke_ok": "server ready, no load",
               "cooldown": "load stopped", "server_stopped": "server stopped"}


def load(rid):
    """One recording: raw 20 Hz power and 1 s means of everything, on a clock that starts at the first mark."""
    R = RUNS / RECS[rid]
    man = json.load(open(R / "run.json"))
    ev = [json.loads(line) for line in open(R / "events.jsonl")]
    t0, t_end = ev[0]["ts_ns"], ev[-1]["ts_ns"]
    n, G = int((t_end - t0) / 1e9), len(man["gpus"])
    p = ds.dataset(R / "gpu_power").to_table(columns=["ts_ns", "device_index", "power_instant_mw"])
    s = ds.dataset(R / "gpu_state").to_table(columns=["ts_ns", "device_index", "gpm_any_tensor_util", "gpm_dram_bw_util",
                                                      "gpm_nvlink_total_tx_per_sec", "gpm_nvlink_total_rx_per_sec", "gpm_pcie_tx_per_sec"])
    col = lambda t, c: t.column(c).to_numpy(zero_copy_only=False)
    pt, pg, pw = (col(p, "ts_ns") - t0) / 1e9, col(p, "device_index"), col(p, "power_instant_mw") / 1000.0
    stt, sg = (col(s, "ts_ns") - t0) / 1e9, col(s, "device_index")
    raw = {"tensor": col(s, "gpm_any_tensor_util"), "hbm_bw": col(s, "gpm_dram_bw_util"),
           "nvlink": (col(s, "gpm_nvlink_total_tx_per_sec") + col(s, "gpm_nvlink_total_rx_per_sec")) / 1000.0,
           "pcie_tx": col(s, "gpm_pcie_tx_per_sec") / 1000.0}

    def per_bin(t, g, v, hz=1):
        """The mean of each GPU's readings in each 1/hz-second bin. A bin with no reading repeats the one before."""
        N, out = n * hz, np.zeros((G, n * hz))
        for k in range(G):
            m = (g == k) & (t >= 0) & (t < n) & ~np.isnan(v)
            b = np.minimum((t[m] * hz).astype(int), N - 1)
            cnt, tot = np.bincount(b, minlength=N), np.bincount(b, weights=v[m], minlength=N)
            last = np.maximum.accumulate(np.where(cnt > 0, np.arange(N), 0))
            out[k] = (tot / np.maximum(cnt, 1))[last]
        return out

    series = {"power": per_bin(pt, pg, pw), **{k: per_bin(stt, sg, v) for k, v in raw.items()}}
    # the same five readings ten times a second: the counters as polled, power as the mean of 2 polls
    # (the driver's power value only changes about ten times a second anyway)
    series5 = {"power": per_bin(pt, pg, pw, FINE_HZ), **{k: per_bin(stt, sg, v, FINE_HZ) for k, v in raw.items()}}
    # What the node sent to other nodes over its high-speed (RDMA) ports, all ports added up, GB/s, once a second.
    # From the side logger: cumulative counters in units of 4 bytes. One row for the whole node, not one per GPU.
    net, last = np.zeros(n), None
    for line in open(R / "extras" / "rdma.jsonl"):
        d = json.loads(line)
        if "v" not in d:
            continue
        cur = (d["ts_ns"], sum(v.get("port_xmit_data", 0) for v in d["v"].values()) * 4)
        if last and cur[0] > last[0] and 0 <= (i := int((cur[0] - t0) / 1e9)) < n:
            net[i] = (cur[1] - last[1]) / ((cur[0] - last[0]) / 1e9) / 1e9
        last = cur
    series["net_tx"] = net[None, :]
    phases = []
    for e in ev[:-1]:
        name = PHASE_NAMES.get(e["phase"]) or ("load generator starting, no requests yet" if e["phase"].startswith("conc_") else e["phase"])
        phases.append([round((e["ts_ns"] - t0) / 1e9), name, e["phase"]])
    log = R / "train.log"
    if log.exists():   # checkpoint saves and the first step, from TorchTitan's own log (the box's clock is UTC)
        stamp = lambda line: datetime.datetime.strptime(line.split("[titan] ")[1][:23] + "+0000", "%Y-%m-%d %H:%M:%S,%f%z").timestamp() - t0 / 1e9
        for line in open(log, errors="replace"):
            if "Saving the checkpointer" in line:
                phases.append([round(stamp(line)), "saving a checkpoint (90 GB)", "save"])
            elif "Finished saving the checkpoint" in line or re.search(r"step: +1 ", line):
                phases.append([round(stamp(line)), "training steps", "steps"])
    mark = {e["phase"]: (e["ts_ns"] - t0) / 1e9 for e in ev}
    # The load generator's own phases, from its log (time of day, UTC): a cache warm-up of long prompts with one token
    # out, then the measured load. A two-node run has one log, in the node that drove it: AGENTX_FROM points there.
    day = datetime.datetime.strptime(man["start_utc"][:10] + "+0000", "%Y-%m-%d%z").timestamp()
    for log in sorted((RUNS / RECS[AGENTX_FROM.get(rid, rid)] / "agentx").glob("c*.log")):
        c = log.stem[1:]
        for line in open(log, errors="replace"):
            m = re.match(r"(\d\d):(\d\d):(\d\d\.\d+) +\w+ +Phase (warmup|profiling) \(\w+\) (started|sending complete)", line)
            if not m:
                continue
            t = day + int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) - t0 / 1e9
            if m[5] == "started":
                mark[("warm_" if m[4] == "warmup" else "load_") + c] = t
                phases.append([round(t), "warm-up: filling the cache (long prompts in, one token out)" if m[4] == "warmup" else f"load: {c} sessions at once", m[4]])
            elif m[4] == "profiling":
                mark["load_end_" + c] = t
                phases.append([round(t), "load generator finishing", "drain"])
    phases.sort(key=lambda q: q[0])
    return {"id": rid, "run": RECS[rid], "n": n, "G": G, "t0_ns": t0, "start_utc": man["start_utc"], "gpu": man["gpus"][0]["name"],
            "host": man["hostname"], "driver": man["driver_version"], "series": series, "series5": series5, "phases": phases,
            "mark": mark, "power20": (pt, pg, pw)}


def evaluate(nodes, jobs, dur, step=5.0):
    """Run the real rules over a scenario. nodes: [(recording, offset_s)], scenario time 0 = recording time offset.
    jobs: [{id, type, node, g: [first, last]}]. Returns one verdict every `step` seconds."""
    power, readings, models = {}, {"pcie_sent": {}, "tensor_util": {}, "net_sent": {}}, {}
    for ni, (rec, off) in enumerate(nodes):
        pt, pg, pw = rec["power20"]
        sec = np.arange(int(off), min(int(off + dur) + 1, rec["n"]))
        readings["net_sent"][str(ni)] = (sec - off + 0.5, rec["series"]["net_tx"][0][sec])
        for g in range(rec["G"]):
            key = f"{ni}:{g}"
            m = (pg == g) & (pt >= off) & (pt <= off + dur)
            power[key] = (pt[m] - off, pw[m])
            sec = np.arange(int(off), min(int(off + dur) + 1, rec["n"]))
            readings["pcie_sent"][key] = (sec - off + 0.5, rec["series"]["pcie_tx"][g][sec])
            readings["tensor_util"][key] = (sec - off + 0.5, rec["series"]["tensor"][g][sec])
            models[key] = rec["gpu"]
    gpus = sorted(power)
    decl = [{"job_id": str(j["id"]), "type": j["type"], "gpus": [f"{j['node']}:{g}" for g in range(j["g"][0], j["g"][1] + 1)]} for j in jobs]
    cut = lambda tv, lo, hi: [(float(a), float(b)) for a, b in zip(*tv) if lo < a <= hi]
    out, t = [], step
    while t <= dur:
        r = rules.evaluate({g: cut(power[g], t - 61, t) for g in gpus}, decl, gpus, t,
                           {k: {g: cut(q, t - 61, t + 1) for g, q in v.items()} for k, v in readings.items()}, models)
        out.append((t, r))
        t += step
    return out


def spans(evals, jobs, step=5.0):
    """Verdicts -> one entry per stretch a rule stayed broken: {rule, a, b, gpus, jobs, d: [[t, text], ...]}."""
    job_gpus = {str(j["id"]): [f"{j['node']}:{g}" for g in range(j["g"][0], j["g"][1] + 1)] for j in jobs}
    open_, done = {}, []
    for t, r in evals:
        now = {}
        for h in r["hits"]:
            if h["rule"] == "undeclared_load":   # one hit per GPU: show them as one
                q = now.setdefault("undeclared_load", {"rule": h["rule"], "gpus": [], "jobs": [], "w": []})
                q["gpus"] += h["gpus"]; q["w"].append(float(h["detail"].split(" at ")[1].split(" W")[0]))
            else:
                ids = [str(x) for x in h["jobs"]]
                gp = h["gpus"] or [g for i in ids for g in job_gpus[i]]
                text = re.sub(r"^(\d+)", r"Job \1", re.sub(r"^(\d+) and (\d+) are", r"\1 and job \2 are", h["detail"]))
                now[h["rule"] + "|" + ",".join(ids)] = {"rule": h["rule"], "gpus": gp, "jobs": [int(i) for i in ids], "text": text}
        for key, q in now.items():
            if "w" in q:
                q["text"] = f"{len(q['gpus'])} GPUs busy with no declared job (avg {np.mean(q.pop('w')):.0f} W)"
            v = open_.get(key)
            if not v:
                v = open_[key] = {"rule": q["rule"], "a": t, "gpus": [], "jobs": q["jobs"], "d": []}
            v["gpus"] = sorted(set(v["gpus"]) | set(q["gpus"])); v["b"] = t + step
            if not v["d"] or v["d"][-1][1] != q["text"]:
                v["d"].append([t, q["text"]])
        for key in [k for k in open_ if k not in now]:
            done.append(open_.pop(key))
    last = evals[-1][0]
    for v in open_.values():
        v["b"] = 1e9 if v["b"] > last else v["b"]   # still broken when the clip ends
        done.append(v)
    return sorted(done, key=lambda v: v["a"])


def summary(evals):
    n = len(evals); bad = {}
    for t, r in evals:
        for h in r["hits"]:
            bad.setdefault(h["rule"], set()).add(t)
    judged = sum(r["status"] != "warming_up" for _, r in evals)
    return f"{judged} checks; " + (", ".join(f"{k} in {len(v)}" for k, v in bad.items()) or "all rules hold")


def table(L):
    """What the rules say about each recording alone, one job over all 8 GPUs, by phase and declared type."""
    spansof = {"p1b": [("steps", "train_start", "train_stop")], "p2": [("steps", "train_start", "train_stop")], "p3": [("steps", "train_start", "train_stop")],
               "i1": [("load 16", "load_16", "load_end_16"), ("warm 16", "warm_16", "load_16"), ("load 64", "load_64", "load_end_64"), ("warm 64", "warm_64", "load_64"), ("loading", "server_start", "server_ready")],
               "i2": [("load 16", "load_16", "load_end_16"), ("warm 16", "warm_16", "load_16"), ("loading", "server_start", "server_ready")],
               "i3p": [("load 16", "load_16", "load_end_16"), ("warm 16", "warm_16", "load_16")], "i3d": [("load 16", "load_16", "load_end_16"), ("warm 16", "warm_16", "load_16")]}
    for rid, rec in L.items():
        for name, a, b in spansof[rid]:
            off, dur = rec["mark"][a], rec["mark"][b] - rec["mark"][a]
            for typ in ("inference", "pretraining", None):
                jobs = [{"id": 1, "type": typ, "node": 0, "g": [0, rec["G"] - 1]}] if typ else []
                print(f"{rid:4} {name:8} declared {str(typ):11} {summary(evaluate([(rec, off)], jobs, dur, 10.0))}")
    off = L["i3p"]["mark"]["load_16"]; dur = L["i3p"]["mark"]["load_end_16"] - off
    jobs = [{"id": 1, "type": "inference", "node": 0, "g": [0, 7]}, {"id": 2, "type": "inference", "node": 1, "g": [0, 7]}]
    ev = evaluate([(L["i3p"], off), (L["i3d"], L["i3d"]["mark"]["load_16"])], jobs, dur, 10.0)
    print("i3 two nodes, two jobs declared inference:", summary(ev))
    f = [r["features"]["jobs"] for _, r in ev if r["status"] != "warming_up"]
    for j in ("1", "2"):
        v = [q[j].get("pcie_sent", 0) for q in f]
        print(f"   job {j} PCIe sent per GPU (60 s median): min {min(v):.2f}, median {np.median(v):.2f}, max {max(v):.2f} GB/s")


def outcome(sc, ev):
    """What happened in a scenario, in sentences, worked out from the verdicts so it stays true if the rules change:
    caught / missed (the rule the scenario is built to break), false alarm (training_shaped on a job that really is
    inference), also (anything else that broke)."""
    mmss = lambda t: f"{int(t) // 60}:{int(t) % 60:02d}"
    truth, expect, out = {j["id"]: j["truth"] for j in sc["jobs"]}, sc["expect"], []
    liar = lambda v: any(truth[i] != "inference" for i in v["jobs"])
    hit = [v for v in sc["viol"] if v["rule"] == expect and (expect != "training_shaped" or liar(v))]
    if expect and hit:
        again = len(hit) - 1
        out.append({"kind": "caught", "text": f"{expect} breaks {mmss(hit[0]['a'])} in" + (f", then lets go and breaks again {again} time{'s' if again > 1 else ''}" if again else "") + "."})
    elif expect:
        text = f"{expect} never breaks."
        if expect == "boundary_traffic":
            top = max(q.get("net_sent", 0) for _, r in ev if r["status"] != "warming_up" for q in r["features"]["jobs"].values() if q["type"] == "inference")
            text += f" The most either job sends to other nodes is {top:.4f} GB/s (60 s mean); the limit is {rules.boundary_traffic.NET_SENT_GB_S}."
        out.append({"kind": "missed", "text": text})
    false = [v for v in sc["viol"] if v["rule"] == "training_shaped" and not liar(v)]
    for job in sorted({i for v in false for i in v["jobs"]}):
        mine = [v for v in false if job in v["jobs"]]
        total = sum(min(v["b"], sc["dur"]) - v["a"] for v in mine)
        out.append({"kind": "false alarm", "text": f"training_shaped breaks on job {job}, which really is inference, for {total:.0f} s in all ({mine[0]['d'][0][1].split(': ', 1)[1]})."})
    for v in sc["viol"]:
        if v not in hit and v not in false and not any(q["kind"] == "also" and q["rule"] == v["rule"] for q in out):
            out.append({"kind": "also", "rule": v["rule"], "text": f"{v['rule']} breaks {mmss(v['a'])} in ({v['d'][-1][1]})."})
    return out or [{"kind": "holds", "text": "All three rules hold from start to end."}]


def scenarios(L):
    """Step 2: recordings laid side by side, with declarations, judged by the real rules. Each clip is 10 minutes.
    Serving clips start when the measured load does (after the load generator's cache warm-up); training clips
    30 s before the launch.
    The first three scenarios each use a different serving and training recording, so they do not look alike:
    tensor-parallel at 16 with the baseline training run, expert-parallel at 16 with middle activation
    checkpointing, tensor-parallel at 64 with batch 1."""
    serve = lambda rid="i1", c=16: (rid, L[rid]["mark"][f"load_{c}"])
    train = lambda rid="p1b": (rid, L[rid]["mark"]["train_start"] - 30)
    # truth: what the job really runs. expect: the rule the scenario is built to break, if any.
    J = lambda i, name, typ, node, truth=None: {"id": i, "name": name, "type": typ, "truth": truth or typ, "node": node, "g": [0, 7], "start": 0, "limit": 720}
    S = [
        {"id": "honest", "expect": None, "name": "Honest cluster", "blurb": "Serving and pretraining, both declared.",
         "played": "tensor-parallel serving at 16 sessions; training, batch 4", "nodes": [serve(), train()],
         "jobs": [J(4101, "deepseek-serve", "inference", 0), J(4102, "llama3-pretrain", "pretraining", 1)]},
        {"id": "disguised", "expect": "training_shaped", "name": "Training declared as inference", "blurb": "A job says inference and trains.",
         "played": "expert-parallel serving at 16 sessions; training with middle activation checkpointing", "nodes": [serve("i2"), train("p2")],
         "jobs": [J(4201, "deepseek-serve", "inference", 0), J(4202, "batch-inference", "inference", 1, "pretraining")]},
        {"id": "undeclared", "expect": "undeclared_load", "name": "Undeclared load", "blurb": "Eight GPUs get busy with no job on them.",
         "played": "tensor-parallel serving at 64 sessions; training, batch 1", "nodes": [serve("i1", 64), train("p3")],
         "jobs": [J(4301, "deepseek-serve", "inference", 0)]},
        {"id": "handoff", "expect": "boundary_traffic", "name": "One job split in two", "blurb": "Two inference jobs on two nodes are one server: prefill on one, decode on the other.",
         "played": "the two-node prefill and decode split at 16 sessions, recorded on both nodes at once",
         "nodes": [("i3p", L["i3p"]["mark"]["load_16"]), ("i3d", L["i3d"]["mark"]["load_16"])],
         "jobs": [J(4401, "serve-1", "inference", 0), J(4402, "serve-2", "inference", 1)]},
    ]
    for sc in S:
        sc["dur"] = 600
        ev = evaluate([(L[r], off) for r, off in sc["nodes"]], sc["jobs"], sc["dur"])
        sc["viol"] = spans(ev, sc["jobs"])
        sc["result"] = outcome(sc, ev)
        sc["nodes"] = [{"name": f"node-{'ab'[i]}", "gpus": L[r]["G"], "rec": r, "off": round(off)} for i, (r, off) in enumerate(sc["nodes"])]
        print(f"scenario {sc['id']}: {summary(ev)}")
        for q in sc["result"]:
            print(f"   [{q['kind']}] {q['text']}")
        for v in sc["viol"]:
            print(f"   {v['rule']} from {v['a']:.0f} s to {'the end' if v['b'] > 1e8 else f'{v['b']:.0f} s'}: {v['d'][-1][1]}")
    return S


def main():
    L = {rid: load(rid) for rid in RECS}
    if "--table" in sys.argv:
        return table(L)
    OUT.mkdir(parents=True, exist_ok=True)
    index = {"made_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"), "scale": SCALE, "recs": {}}
    for rid, rec in L.items():
        body = {"id": rid, "n": rec["n"], "s": {k: [[int(round(x * SCALE[k])) for x in row] for row in v] for k, v in rec["series"].items()}}
        text = json.dumps(body, separators=(",", ":"))
        (OUT / f"{rid}.json").write_text(text)
        fine = json.dumps({"id": rid, "n": rec["n"], "hz": FINE_HZ, "s": {k: [[int(round(x * SCALE[k])) for x in row] for row in v] for k, v in rec["series5"].items()}}, separators=(",", ":"))
        (OUT / f"{rid}.hz{FINE_HZ}.json").write_text(fine)
        index["recs"][rid] = {"run": rec["run"], "n": rec["n"], "gpus": rec["G"], "gpu": rec["gpu"].replace("NVIDIA ", ""), "host": rec["host"],
                              "start_utc": rec["start_utc"], "driver": rec["driver"], "phases": [q[:2] for q in rec["phases"]],
                              "mark": {k: round(v) for k, v in rec["mark"].items() if not k.startswith("_")}}
        print(f"{rid}: {rec['n']} s, {len(text) // 1024} KB at 1 Hz, {len(fine) // 1024} KB at {FINE_HZ} Hz")
    index["scenarios"] = scenarios(L)
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
