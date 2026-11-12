"""Numbers the rules care about, from a recording made with the collector sending to a local file.

    .venv/bin/python recordings/analyse.py <recording.jsonl> <phases.jsonl> <from-phase> <to-phase> [seconds to skip at the start]

Prints mean power, the spread of the 5 s average power per 60 s window (rule 2's power-spread flag, removed 2026-11-11), and the
once-a-second counters (tensor-core utilisation for rule 2, PCIe sent for rule 3).
"""
import json, sys
import numpy as np, pandas as pd

rec, phases_path, p_from, p_to = sys.argv[1:5]
skip = float(sys.argv[5]) if len(sys.argv) > 5 else 0
rows = []
for line in open(rec):
    d = json.loads(line)
    if d["path"].endswith("/collect") and d["body"]:
        rows += d["body"]["samples"]
df = pd.DataFrame(rows)
ph = {}
for l in open(phases_path):
    p = json.loads(l); ph[p["phase"]] = p["t"]            # the last mark of each name wins
start, stop = ph[p_from] + skip, ph.get(p_to, df.t.max())
w = df[(df.t > start) & (df.t < stop - 1)].copy()
w["s"] = w.t.astype(int)
p = w.groupby(["s", "gpu"]).power_w.mean().unstack()      # 1 s means per GPU
job = p.mean(axis=1)
print(f"{p_from} -> {p_to}, skipping {skip:.0f} s: {stop - start:.0f} s analysed, {len(w):,} samples")
print(f"power: {job.mean():.0f} W per GPU on average (per GPU {p.mean().min():.0f}..{p.mean().max():.0f}); 1 s job mean ranged {job.min():.0f}..{job.max():.0f} W")

def spreads(series, win=60, roll=5):
    r = series.rolling(roll).mean().dropna().to_numpy()
    return np.array([r[i:i + win].max() - r[i:i + win].min() for i in range(0, len(r) - win + 1, 10)])   # 60 s windows every 10 s, as the rules run
if len(job) >= 70:
    jb = spreads(job)
    hot = np.array([job.rolling(5).mean().dropna().to_numpy()[i:i + 60].mean() for i in range(0, len(job) - 4 - 60 + 1, 10)])
    print(f"power spread per 60 s window: median {np.median(jb):.0f} W, min {jb.min():.0f}, max {jb.max():.0f} ({len(jb)} windows); "
          f"under 100 W in {100 * np.mean(jb < 100):.0f}% of them; windows averaging over 400 W: {100 * np.mean(hot > 400):.0f}%")
slow = w.dropna(subset=["tensor_util_pct"]) if "tensor_util_pct" in w else w.iloc[0:0]
for col, name in [("tensor_util_pct", "tensor-core utilisation %"), ("dram_bw_util_pct", "memory bandwidth %"), ("sm_util_pct", "SM utilisation %"),
                  ("nvlink_tx_gb_s", "NVLink sent GB/s"), ("pcie_tx_gb_s", "PCIe sent GB/s"), ("pcie_rx_gb_s", "PCIe received GB/s")]:
    if col in slow and len(slow):
        q = slow[col].quantile([0.1, 0.5, 0.9, 0.99]).round(2).tolist()
        j = slow.groupby("s")[col].mean()
        m60 = [j.iloc[i:i + 60].median() for i in range(0, len(j) - 59, 10)]
        rng = f"{min(m60):.2f}..{max(m60):.2f}" if m60 else "n/a"
        print(f"{name:27s} 10th/median/90th/99th per reading: {q}; 60 s median of the job: {rng}")
