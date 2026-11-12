"""Run the real rules over a recording, every 10 s, as if one job covered all its GPUs.

    .venv/bin/python recordings/run_rules.py <recording.jsonl> <phases.jsonl> <declared type> [from-phase to-phase]
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "node"))
import rules

rec, phases_path, declared = sys.argv[1:4]
ph = {}
for l in open(phases_path):
    p = json.loads(l); ph[p["phase"]] = p["t"]
power, readings, models = defaultdict(list), {"pcie_sent": defaultdict(list), "tensor_util": defaultdict(list)}, {}
for line in open(rec):
    d = json.loads(line)
    if not (d["path"].endswith("/collect") and d["body"]): continue
    for g, name in d["body"].get("gpus", {}).items(): models[f"box:{g}"] = name
    for s in d["body"]["samples"]:
        g = f"box:{s['gpu']}"
        power[g].append((s["t"], s["power_w"]))
        if s.get("pcie_tx_gb_s") is not None: readings["pcie_sent"][g].append((s["t"], s["pcie_tx_gb_s"]))
        if s.get("tensor_util_pct") is not None: readings["tensor_util"][g].append((s["t"], s["tensor_util_pct"]))
gpus = sorted(power)
t0 = ph[sys.argv[4]] if len(sys.argv) > 4 else min(p[0][0] for p in power.values())
t1 = ph[sys.argv[5]] if len(sys.argv) > 5 else max(p[-1][0] for p in power.values())
decl = [{"job_id": "1", "type": declared, "gpus": gpus}] if declared != "none" else []
out, n, first = Counter(), 0, {}
t = t0 + 60
while t <= t1:
    lo = t - 61
    r = rules.evaluate({g: [x for x in power[g] if lo < x[0] <= t] for g in gpus}, decl, gpus, t,
                       {k: {g: [x for x in v[g] if lo < x[0] <= t + 1] for g in gpus} for k, v in readings.items()}, models)
    n += 1; out[r["status"]] += 1
    for h in r["hits"]:
        for key in ([f"{h['rule']}/{f}" for f in h["flags"]] if h.get("flags") else [h["rule"]]):
            out[key] += 1; first.setdefault(key, (round(t - t0), h["detail"]))
    t += 10
print(f"{n} evaluations, declared {declared}: " + ", ".join(f"{k} {v}" for k, v in out.most_common()))
for k, (sec, detail) in first.items(): print(f"   first {k} at +{sec} s: {detail}")
