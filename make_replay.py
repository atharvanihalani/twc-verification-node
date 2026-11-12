"""Turn a slice of a recorded serving run into a replay file for replay_feed.py.

The output is already in collector.py's wire format (same field names and units), so
the node can't tell a replayed row from a live one. Run locally, once per slice:

    .venv/bin/python make_replay.py            # writes replays/c1-tp8-c64.jsonl.gz

File layout (gzipped JSON lines):
    line 1        header: {"name", "label", "gpus": {idx: name}, "dt": 0.1, "ticks": N, "source": ...}
    lines 2..N+1  one 0.1 s tick: {"p": [power_w per GPU], "s": [slow-field dict per GPU]}
                  "s" is only present on every 10th tick (the collector's 1 Hz fields)
Timestamps are left out: the replayer stamps every row with the current time.
"""
import argparse
import glob
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "telemetry-logs/serving-traces-vllm/runs/c1-tp8-dspark-20261103T232811Z"

# c1 conc_64: AgentX profiling phase started 2026-11-04 00:16:17.780 UTC (agentx/c64.log line 197).
# For its first ~5 min only ~10 of 64 sessions are active (the post-warmup ramp, see RUNLOG.md);
# from +5 min to the end (+15 min) it's steady at ~45-52 running requests, ~590-630 W per GPU.
PROFILING_START = 1793751377.780
DEFAULT_START, DEFAULT_LEN = 300, 600   # seconds after profiling start, slice length

# logger (gpu_state) column -> collector field, scale
STATE_MAP = {
    "clk_sm_mhz": ("sm_clock_mhz", 1), "clk_event_reasons": ("throttle_reasons", 1),
    "temp_gpu_c": ("gpu_temp_c", 1), "temp_hbm_c": ("hbm_temp_c", 1), "mem_used_bytes": ("mem_used_gb", 1e-9),
    "gpm_sm_util": ("sm_util_pct", 1), "gpm_sm_occupancy": ("sm_occupancy_pct", 1),
    "gpm_any_tensor_util": ("tensor_util_pct", 1), "gpm_dram_bw_util": ("dram_bw_util_pct", 1),
    "gpm_pcie_tx_per_sec": ("pcie_tx_gb_s", 1e-3), "gpm_pcie_rx_per_sec": ("pcie_rx_gb_s", 1e-3),   # MB/s -> GB/s
    "gpm_nvlink_total_tx_per_sec": ("nvlink_tx_gb_s", 1e-3), "gpm_nvlink_total_rx_per_sec": ("nvlink_rx_gb_s", 1e-3),
}
INT_FIELDS = {"sm_clock_mhz", "throttle_reasons", "gpu_temp_c", "hbm_temp_c"}


def load(stream, cols, lo, hi):
    parts = []
    for f in sorted(glob.glob(str(RUN / stream / "*.parquet"))):
        t = pq.read_table(f, columns=cols).to_pandas()
        t = t[(t.ts_ns >= lo * 1e9) & (t.ts_ns < hi * 1e9)]
        if len(t):
            parts.append(t)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=DEFAULT_START, help="seconds after c64 profiling start")
    ap.add_argument("--len", type=float, default=DEFAULT_LEN, help="slice length, seconds")
    ap.add_argument("--out", default=str(HERE / "replays" / "c1-tp8-c64.jsonl.gz"))
    a = ap.parse_args()

    lo, hi = PROFILING_START + a.start, PROFILING_START + a.start + a.len
    pad = 5   # read a little either side so nearest-match has neighbours at the edges
    pw = load("gpu_power", ["ts_ns", "device_index", "power_instant_mw", "energy_mj"], lo - pad, hi + pad)
    st = load("gpu_state", ["ts_ns", "device_index", *STATE_MAP], lo - pad, hi + pad)
    gpus = sorted(pw.device_index.unique().tolist())
    names = {str(g["index"]): g["name"] for g in json.loads((RUN / "run.json").read_text()).get("gpus", [])} \
        if (RUN / "run.json").exists() else {}

    n = int(round(a.len * 10))
    grid = pd.DataFrame({"t": lo + np.arange(n) * 0.1})
    power, slow = {}, {}
    for g in gpus:
        p = pw[pw.device_index == g].assign(t=lambda d: d.ts_ns / 1e9).sort_values("t")
        # power: the 20 Hz reading nearest each 10 Hz tick
        power[g] = pd.merge_asof(grid, p[["t", "power_instant_mw"]], on="t", direction="nearest").power_instant_mw.to_numpy() / 1000
        # energy is read round-robin (~2.5 Hz/GPU), so take the latest reading at or before the tick
        e = p.dropna(subset=["energy_mj"])[["t", "energy_mj"]]
        s = st[st.device_index == g].assign(t=lambda d: d.ts_ns / 1e9).sort_values("t")
        sg = grid.iloc[::10]
        sg = pd.merge_asof(sg, e, on="t", direction="nearest")
        slow[g] = pd.merge_asof(sg, s.drop(columns=["ts_ns", "device_index"]), on="t", direction="nearest")

    out = Path(a.out)
    out.parent.mkdir(exist_ok=True)
    header = {
        "name": "c1-tp8-c64",
        "label": "recorded 2026-11-03",                    # the pill next to the host name
        "about": "c1 · vLLM TP8 · 64 concurrent",         # shown in the status line
        "gpus": {str(g): names.get(str(g), "NVIDIA H200") for g in gpus},
        "dt": 0.1, "ticks": n,
        "source": f"{RUN.name} conc_64 profiling +{a.start:.0f}s..+{a.start + a.len:.0f}s",
    }
    with gzip.open(out, "wt") as f:
        f.write(json.dumps(header) + "\n")
        for k in range(n):
            tick = {"p": [round(float(power[g][k]), 1) for g in gpus]}
            if k % 10 == 0:
                rows = []
                for g in gpus:
                    r = slow[g].iloc[k // 10]
                    row = {"energy_mj": int(r.energy_mj)} if not math.isnan(r.energy_mj) else {}
                    for col, (key, scale) in STATE_MAP.items():
                        v = r[col]
                        if v is None or (isinstance(v, float) and not math.isfinite(v)):
                            continue   # same rule as the collector: never send NaN/inf
                        v = float(v) * scale
                        row[key] = int(round(v)) if key in INT_FIELDS else round(v, 2 if key == "mem_used_gb" else 3)
                    rows.append(row)
                tick["s"] = rows
            f.write(json.dumps(tick, separators=(",", ":")) + "\n")
    p = np.array([power[g] for g in gpus])
    print(f"wrote {out} ({out.stat().st_size // 1024} KB): {n} ticks x {len(gpus)} GPUs, "
          f"power mean {p.mean():.0f} W (min {p.min():.0f}, max {p.max():.0f})")


if __name__ == "__main__":
    main()
