#!/usr/bin/env python3
"""Serving-trace telemetry logger for an 8x H200 vLLM box.

Usage (on the box):
    python logger.py run  <run_id> [--notes "..."]     # foreground; SIGTERM = clean flush + exit
    python logger.py mark <phase> [--run-id RID]       # append {ts_ns, mono_ns, phase} to events.jsonl

Streams (one thread each, written as parquet part files every FLUSH_S seconds):
    gpu_power     20 Hz  power_instant_mw, power_avg_mw, energy_mj (+ NVML field timestamps)
    gpu_state     10 Hz  clocks, temps, throttle bitmask, mem used, GPM metrics over each ~100 ms window
    engine_metrics 1 Hz  every vllm:* series from /metrics (long format, per-engine labels kept)
    host           1 Hz  /proc CPU util (overall + per NUMA), mem, vLLM RSS, logger's own CPU/RSS

Timestamps: ts_ns = time.time_ns() (UTC epoch ns), mono_ns = time.monotonic_ns()
(CLOCK_MONOTONIC, comparable across processes on the box, used by `mark`).
"""
import argparse, json, os, re, signal, socket, sys, threading, time, urllib.request

RUNS_ROOT = os.environ.get("RUNS_ROOT", "/workspace/runs")
METRICS_URL = os.environ.get("VLLM_METRICS_URL", "http://127.0.0.1:8001/metrics")
# Comma-separated for several independent servers; each is tagged replica=<i> in the labels.
METRICS_URLS = [u.strip() for u in METRICS_URL.split(",") if u.strip()]
NODE_ID = os.environ.get("NODE_ID", socket.gethostname())
FLUSH_S = float(os.environ.get("FLUSH_S", "15"))
POWER_HZ, STATE_HZ, ENGINE_HZ, HOST_HZ = 20.0, 10.0, 1.0, 1.0

# NVML field ids (nvml.h)
FI_POWER_INSTANT = 186
FI_POWER_AVERAGE = 185
FI_TOTAL_ENERGY = 83      # NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION, mJ
FI_MEMORY_TEMP = 82       # NVML_FI_DEV_MEMORY_TEMP (HBM), degC

GPM_CANDIDATES = [
    "SM_UTIL", "SM_OCCUPANCY", "GRAPHICS_UTIL", "ANY_TENSOR_UTIL", "HMMA_TENSOR_UTIL",
    "IMMA_TENSOR_UTIL", "DFMA_TENSOR_UTIL", "DRAM_BW_UTIL", "FP16_UTIL", "FP32_UTIL",
    "FP64_UTIL", "INTEGER_UTIL", "NVLINK_TOTAL_TX_PER_SEC", "NVLINK_TOTAL_RX_PER_SEC",
    "PCIE_TX_PER_SEC", "PCIE_RX_PER_SEC",
]


def now():
    return time.time_ns(), time.monotonic_ns()


# ---------------------------------------------------------------- writer
class Stream:
    """Column buffer for one stream; flush() writes a parquet part file."""

    def __init__(self, run_dir, name, const):
        self.dir = os.path.join(run_dir, name)
        os.makedirs(self.dir, exist_ok=True)
        self.name, self.const = name, const
        self.cols, self.lock, self.part, self.rows_total = {}, threading.Lock(), 0, 0

    def add(self, row):
        with self.lock:
            n = len(next(iter(self.cols.values()))) if self.cols else 0
            for k, v in row.items():
                if k not in self.cols:
                    self.cols[k] = [None] * n
                self.cols[k].append(v)
            for k in self.cols:          # pad columns missing from this row
                if len(self.cols[k]) < n + 1:
                    self.cols[k].append(None)

    def flush(self):
        with self.lock:
            cols, self.cols = self.cols, {}
        if not cols:
            return 0
        import pyarrow as pa, pyarrow.parquet as pq
        n = len(next(iter(cols.values())))
        data = {k: pa.array([v] * n).dictionary_encode() for k, v in self.const.items()}
        data.update({k: pa.array(v) for k, v in cols.items()})
        self.part += 1
        tmp = os.path.join(self.dir, f".part-{self.part:05d}.tmp")
        pq.write_table(pa.table(data), tmp, compression="zstd")
        os.replace(tmp, os.path.join(self.dir, f"part-{self.part:05d}.parquet"))
        self.rows_total += n
        return n


# ---------------------------------------------------------------- loop helper
def periodic(hz, stop, fn, stats):
    """Absolute-deadline loop: drift-free, records achieved period + work time."""
    period = 1.0 / hz
    nxt = time.monotonic()
    last = None
    while not stop.is_set():
        t0 = time.monotonic()
        if last is not None:
            stats["periods"].append(t0 - last)
        last = t0
        try:
            fn()
        except Exception as e:  # never die on one bad sample
            stats["errors"] += 1
            if stats["errors"] <= 5:
                print(f"[{stats['name']}] error: {type(e).__name__}: {e}", flush=True)
        stats["work"].append(time.monotonic() - t0)
        nxt += period
        delay = nxt - time.monotonic()
        if delay < 0:                      # overran: skip missed slots, don't burst
            stats["overruns"] += 1
            nxt = time.monotonic()
            delay = 0
        stop.wait(delay)


def new_stats(name):
    return {"name": name, "periods": [], "work": [], "errors": 0, "overruns": 0}


def summarize(stats):
    p, w = stats["periods"], stats["work"]
    stats["periods"], stats["work"] = [], []
    if not p:
        return f"{stats['name']}: no samples"
    p = sorted(p); w = sorted(w)
    return (f"{stats['name']}: n={len(p)} period mean={1e3*sum(p)/len(p):.2f}ms "
            f"p99={1e3*p[int(.99*(len(p)-1))]:.2f}ms max={1e3*p[-1]:.2f}ms | "
            f"work mean={1e3*sum(w)/len(w):.2f}ms max={1e3*w[-1]:.2f}ms | "
            f"overruns={stats['overruns']} errors={stats['errors']}")


# ---------------------------------------------------------------- run.json helpers
def find_vllm_serve():
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        argv = [a.decode(errors="replace") for a in argv if a]
        for i, a in enumerate(argv[:3]):
            if a.endswith("vllm") and i + 1 < len(argv) and argv[i + 1] == "serve":
                return int(pid), argv
    return None, None


def pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def gpu_numa(N, h):
    bus = N.nvmlDeviceGetPciInfo(h).busId
    bus = bus.decode() if isinstance(bus, bytes) else bus
    return int(open(f"/sys/bus/pci/devices/{bus.lower()[4:]}/numa_node").read())


def write_run_json(run_dir, run_id, notes, handles, gpm_ok, cpus):
    import pynvml as N
    def s(f, *a):
        try:
            v = f(*a)
            return v.decode() if isinstance(v, bytes) else v
        except Exception as e:
            return f"ERR:{type(e).__name__}"
    gpus = []
    for i, h in enumerate(handles):
        gpus.append({
            "index": i, "name": s(N.nvmlDeviceGetName, h), "uuid": s(N.nvmlDeviceGetUUID, h),
            "serial": s(N.nvmlDeviceGetSerial, h),
            "board_part_number": s(N.nvmlDeviceGetBoardPartNumber, h),
            "vbios": s(N.nvmlDeviceGetVbiosVersion, h),
            "pci_bus_id": s(lambda x: N.nvmlDeviceGetPciInfo(x).busId, h),
            "power_limit_mw": s(N.nvmlDeviceGetPowerManagementLimit, h),
            "enforced_power_limit_mw": s(N.nvmlDeviceGetEnforcedPowerLimit, h),
            "max_clocks_mhz": {"sm": s(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_SM),
                               "mem": s(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_MEM)},
            "persistence_mode": s(N.nvmlDeviceGetPersistenceMode, h),
            "numa_node": s(gpu_numa, N, h),
        })
    pid, argv = find_vllm_serve()
    model = None
    if argv:
        k = argv.index("serve")
        model = argv[k + 1] if k + 1 < len(argv) else None
    ts, mono = now()
    meta = {
        "run_id": run_id, "node_id": NODE_ID, "hostname": socket.gethostname(),
        "start_ts_ns": ts, "start_mono_ns": mono,
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts / 1e9)),
        "notes": notes,
        "driver_version": s(N.nvmlSystemGetDriverVersion),
        "cuda_driver_version": s(N.nvmlSystemGetCudaDriverVersion),
        "nvml_version": s(N.nvmlSystemGetNVMLVersion),
        "versions": {"vllm": pkg_version("vllm"), "torch": pkg_version("torch"),
                     "nvidia-ml-py": pkg_version("nvidia-ml-py"), "python": sys.version.split()[0]},
        "vllm_pid": pid, "vllm_cmdline": argv, "model_path": model,
        "gpus": gpus,
        "logger": {"power_hz": POWER_HZ, "state_hz": STATE_HZ, "engine_hz": ENGINE_HZ,
                   "host_hz": HOST_HZ, "flush_s": FLUSH_S, "gpm_metrics": gpm_ok,
                   "cpu_affinity": sorted(os.sched_getaffinity(0)), "pid": os.getpid(),
                   "metrics_urls": METRICS_URLS},
        "vast_env": {k: v for k, v in os.environ.items()
                     if k.startswith(("VAST_", "CONTAINER_ID", "GPU_COUNT"))},
        "units": {"ts_ns": "UTC epoch ns", "power_*_mw": "mW", "energy_mj": "mJ (monotonic since driver load); read round-robin, one GPU per 20 Hz sweep -> 2.5 Hz/GPU, null otherwise",
                  "temp_*_c": "degC", "clk_*_mhz": "MHz", "mem_used_bytes": "bytes",
                  "gpm_*": "pct for *_UTIL/OCCUPANCY, bytes/s for *_PER_SEC (NVML GPM units)"},
    }
    json.dump(meta, open(os.path.join(run_dir, "run.json"), "w"), indent=2)
    return meta


# ---------------------------------------------------------------- GPM
def gpm_setup(N, handles):
    ids = {n: getattr(N, f"NVML_GPM_METRIC_{n}") for n in GPM_CANDIDATES if hasattr(N, f"NVML_GPM_METRIC_{n}")}
    h = handles[0]
    a, b = N.nvmlGpmSampleAlloc(), N.nvmlGpmSampleAlloc()
    N.nvmlGpmSampleGet(h, a); time.sleep(0.1); N.nvmlGpmSampleGet(h, b)
    names = list(ids)
    res = gpm_compute(N, a, b, [ids[n] for n in names])
    N.nvmlGpmSampleFree(a); N.nvmlGpmSampleFree(b)
    ok = [n for n, (rc, _) in zip(names, res) if rc == 0]
    bad = {n: rc for n, (rc, _) in zip(names, res) if rc != 0}
    return ok, [ids[n] for n in ok], bad


def gpm_compute(N, s1, s2, metric_ids):
    mg = N.c_nvmlGpmMetricsGet_t()
    mg.version = N.NVML_GPM_METRICS_GET_VERSION
    mg.numMetrics = len(metric_ids)
    mg.sample1, mg.sample2 = s1, s2
    for i, m in enumerate(metric_ids):
        mg.metrics[i].metricId = m
    N.nvmlGpmMetricsGet(mg)
    return [(mg.metrics[i].nvmlReturn, mg.metrics[i].value) for i in range(len(metric_ids))]


# ---------------------------------------------------------------- host helpers
def read_cpu_jiffies():
    out = {}
    for line in open("/proc/stat"):
        if not line.startswith("cpu"):
            break
        f = line.split()
        v = list(map(int, f[1:9]))
        idle = v[3] + v[4]
        out[f[0]] = (sum(v), idle)
    return out


def parse_cpulist(s):
    cpus = []
    for part in s.strip().split(","):
        if "-" in part:
            a, b = part.split("-"); cpus.extend(range(int(a), int(b) + 1))
        elif part:
            cpus.append(int(part))
    return cpus


def numa_map():
    m, base = {}, "/sys/devices/system/node"
    try:
        for d in os.listdir(base):
            if d.startswith("node") and d[4:].isdigit():
                for c in parse_cpulist(open(f"{base}/{d}/cpulist").read()):
                    m[f"cpu{c}"] = int(d[4:])
    except OSError:
        pass
    return m


def vllm_pids():
    pids = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            comm = open(f"/proc/{pid}/comm").read().strip()
            if comm.startswith("VLLM") or "vllm" in comm:
                pids.append(int(pid)); continue
            cl = open(f"/proc/{pid}/cmdline", "rb").read()
            if b"vllm" in cl and b"serve" in cl:
                pids.append(int(pid))
        except OSError:
            pass
    return pids


PROM_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)')
LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


# ---------------------------------------------------------------- main run
def cmd_run(args):
    import pynvml as N
    run_dir = os.path.join(RUNS_ROOT, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    N.nvmlInit()
    ng = N.nvmlDeviceGetCount()
    handles = [N.nvmlDeviceGetHandleByIndex(i) for i in range(ng)]

    gpm_names, gpm_ids, gpm_bad = gpm_setup(N, handles)
    print(f"GPM ok: {gpm_names}\nGPM unsupported (rc): {gpm_bad}", flush=True)

    # probe the single-call power fieldset; fall back to separate energy call if 83 fails
    fv = N.nvmlDeviceGetFieldValues(handles[0], [FI_POWER_INSTANT, FI_POWER_AVERAGE, FI_TOTAL_ENERGY, FI_MEMORY_TEMP])
    fi_ok = {fid: fv[k].nvmlReturn == 0 for k, fid in enumerate([FI_POWER_INSTANT, FI_POWER_AVERAGE, FI_TOTAL_ENERGY, FI_MEMORY_TEMP])}
    print(f"field support: {fi_ok}", flush=True)
    energy_via_field = fi_ok[FI_TOTAL_ENERGY]
    hbm_temp_ok = fi_ok[FI_MEMORY_TEMP]

    meta = write_run_json(run_dir, args.run_id, args.notes, handles, gpm_names, None)
    meta["logger"]["gpm_unsupported"] = {k: int(v) for k, v in gpm_bad.items()}
    meta["logger"]["field_support"] = {str(k): v for k, v in fi_ok.items()}
    json.dump(meta, open(os.path.join(run_dir, "run.json"), "w"), indent=2)

    const = {"run_id": args.run_id, "node_id": NODE_ID}
    s_power = Stream(run_dir, "gpu_power", {**const, "scope": "gpu"})
    s_state = Stream(run_dir, "gpu_state", {**const, "scope": "gpu"})
    s_eng = Stream(run_dir, "engine_metrics", const)
    s_host = Stream(run_dir, "host", const)
    streams = [s_power, s_state, s_eng, s_host]
    stop = threading.Event()

    # ---- A. power
    # Energy reads cost ~5.4 ms per GPU (measured on this box; power reads ~0.05 ms), so reading
    # energy for all 8 GPUs every tick would eat ~43 ms of the 50 ms budget. Round-robin instead:
    # each 20 Hz sweep reads the energy counter of ONE GPU (sweep % n) -> 2.5 Hz per GPU.
    pfields = [FI_POWER_INSTANT, FI_POWER_AVERAGE]
    st_power = new_stats("gpu_power"); sweep = [0]

    def power_tick():
        sweep[0] += 1
        t_sweep = time.monotonic_ns()
        rows = []
        for i, h in enumerate(handles):
            ts, mono = now()
            v = N.nvmlDeviceGetFieldValues(h, pfields)
            e = N.nvmlDeviceGetTotalEnergyConsumption(h) if i == sweep[0] % ng else None
            rows.append({"ts_ns": ts, "mono_ns": mono, "device_index": i,
                         "power_instant_mw": v[0].value.uiVal if v[0].nvmlReturn == 0 else None,
                         "power_avg_mw": v[1].value.uiVal if v[1].nvmlReturn == 0 else None,
                         "energy_mj": e,
                         "nvml_ts_us": v[0].timestamp,  # driver-side sample timestamp of power_instant
                         "sweep": sweep[0]})
        dur = (time.monotonic_ns() - t_sweep) // 1000
        for r in rows:
            r["poll_us"] = dur
            s_power.add(r)

    # ---- B. state + GPM
    st_state = new_stats("gpu_state")
    gpm_prev = [None] * ng
    if gpm_ids:
        for i, h in enumerate(handles):
            gpm_prev[i] = N.nvmlGpmSampleAlloc(); N.nvmlGpmSampleGet(h, gpm_prev[i])
    gpm_cur = [N.nvmlGpmSampleAlloc() for _ in range(ng)] if gpm_ids else None
    gpm_cols = [f"gpm_{n.lower()}" for n in gpm_names]
    sfields = [FI_MEMORY_TEMP] if hbm_temp_ok else []
    throttle_fn = getattr(N, "nvmlDeviceGetCurrentClocksEventReasons", None) or N.nvmlDeviceGetCurrentClocksThrottleReasons

    def state_tick():
        t_sweep = time.monotonic_ns()
        rows = []
        for i, h in enumerate(handles):
            ts, mono = now()
            r = {"ts_ns": ts, "mono_ns": mono, "device_index": i,
                 "clk_sm_mhz": N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM),
                 "clk_mem_mhz": N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_MEM),
                 "temp_gpu_c": N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU),
                 "clk_event_reasons": int(throttle_fn(h)),
                 "mem_used_bytes": N.nvmlDeviceGetMemoryInfo(h).used}
            if sfields:
                fv = N.nvmlDeviceGetFieldValues(h, sfields)
                r["temp_hbm_c"] = fv[0].value.uiVal if fv[0].nvmlReturn == 0 else None
            if gpm_ids:
                N.nvmlGpmSampleGet(h, gpm_cur[i])
                win = gpm_compute(N, gpm_prev[i], gpm_cur[i], gpm_ids)
                gpm_prev[i], gpm_cur[i] = gpm_cur[i], gpm_prev[i]
                for c, (rc, val) in zip(gpm_cols, win):
                    r[c] = float(val) if rc == 0 else None
            rows.append(r)
        dur = (time.monotonic_ns() - t_sweep) // 1000
        for r in rows:
            r["poll_us"] = dur
            s_state.add(r)

    # ---- C. engine metrics
    st_eng = new_stats("engine_metrics")

    def engine_tick():
        ts, mono = now()
        for rep, url in enumerate(METRICS_URLS):
            body = urllib.request.urlopen(url, timeout=0.8).read().decode()
            dur = (time.monotonic_ns() - mono) // 1000
            for line in body.splitlines():
                if not line.startswith("vllm:"):
                    continue
                m = PROM_RE.match(line)
                if not m:
                    continue
                name, lab, val = m.groups()
                if name.endswith("_bucket") and not args.engine_buckets:
                    continue
                labels = dict(LABEL_RE.findall(lab or ""))
                labels.pop("model_name", None)
                eng = labels.pop("engine", None)
                if len(METRICS_URLS) > 1:
                    labels["replica"] = str(rep)
                s_eng.add({"ts_ns": ts, "mono_ns": mono,
                           "scope": "gpu" if eng is not None else "node",
                           "device_index": int(eng) if eng is not None else -1,
                           "metric": name[5:],
                           "labels": ",".join(f"{k}={v}" for k, v in sorted(labels.items())),
                           "value": float(val), "scrape_us": dur})

    # ---- E. host
    st_host = new_stats("host")
    numa = numa_map()
    prev = {"cpu": read_cpu_jiffies(), "self": None, "t": time.monotonic(), "pids": [], "pids_t": 0}
    page = os.sysconf("SC_PAGE_SIZE"); hz = os.sysconf("SC_CLK_TCK")

    def host_tick():
        ts, mono = now()
        t = time.monotonic()
        cur = read_cpu_jiffies()
        out = []
        tot = cur["cpu"]; ptot = prev["cpu"]["cpu"]
        dt = tot[0] - ptot[0]
        if dt > 0:
            out.append(("node", -1, "cpu_util_pct", 100.0 * (1 - (tot[1] - ptot[1]) / dt)))
        agg = {}
        for k, (a, idle) in cur.items():
            if k == "cpu" or k not in prev["cpu"] or k not in numa:
                continue
            pa_, pi_ = prev["cpu"][k]
            n = numa[k]
            x = agg.setdefault(n, [0, 0]); x[0] += a - pa_; x[1] += idle - pi_
        for n, (a, idle) in sorted(agg.items()):
            if a > 0:
                out.append(("cpu", n, "cpu_util_pct", 100.0 * (1 - idle / a)))
        prev["cpu"] = cur
        mi = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1); mi[k] = int(v.split()[0]) * 1024
        out.append(("node", -1, "mem_total_bytes", float(mi["MemTotal"])))
        out.append(("node", -1, "mem_used_bytes", float(mi["MemTotal"] - mi["MemAvailable"])))
        for p in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
            if os.path.exists(p):
                out.append(("node", -1, "cgroup_mem_bytes", float(open(p).read()))); break
        if t - prev["pids_t"] > 30:
            prev["pids"], prev["pids_t"] = vllm_pids(), t
        rss = 0; alive = []
        for pid in prev["pids"]:
            try:
                rss += int(open(f"/proc/{pid}/statm").read().split()[1]) * page; alive.append(pid)
            except OSError:
                pass
        prev["pids"] = alive
        out.append(("node", -1, "vllm_rss_bytes", float(rss)))
        out.append(("node", -1, "vllm_nprocs", float(len(alive))))
        f = open("/proc/self/stat").read().rsplit(")", 1)[1].split()
        cpu_s = (int(f[11]) + int(f[12])) / hz
        if prev["self"] is not None:
            out.append(("node", -1, "logger_cpu_pct", 100.0 * (cpu_s - prev["self"]) / (t - prev["t"])))
        prev["self"], prev["t"] = cpu_s, t
        out.append(("node", -1, "logger_rss_bytes", float(int(open("/proc/self/statm").read().split()[1]) * page)))
        la = open("/proc/loadavg").read().split()
        out.append(("node", -1, "loadavg_1m", float(la[0])))
        for scope, idx, metric, val in out:
            s_host.add({"ts_ns": ts, "mono_ns": mono, "scope": scope, "device_index": idx,
                        "metric": metric, "value": val})

    threads = [
        threading.Thread(target=periodic, args=(POWER_HZ, stop, power_tick, st_power), daemon=True),
        threading.Thread(target=periodic, args=(STATE_HZ, stop, state_tick, st_state), daemon=True),
        threading.Thread(target=periodic, args=(ENGINE_HZ, stop, engine_tick, st_eng), daemon=True),
        threading.Thread(target=periodic, args=(HOST_HZ, stop, host_tick, st_host), daemon=True),
    ]

    def on_sig(signum, frame):
        print(f"signal {signum}: stopping", flush=True)
        stop.set()
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)

    for th in threads:
        th.start()
    print(f"logger started run_id={args.run_id} pid={os.getpid()} dir={run_dir}", flush=True)
    stats_all = [st_power, st_state, st_eng, st_host]
    while not stop.wait(FLUSH_S):
        t0 = time.monotonic()
        n = [s.flush() for s in streams]
        print(time.strftime("%H:%M:%S"), f"flushed rows={n} in {1e3*(time.monotonic()-t0):.0f}ms", flush=True)
        for st in stats_all:
            print("   ", summarize(st), flush=True)
    for th in threads:
        th.join(timeout=5)
    n = [s.flush() for s in streams]
    print(f"final flush rows={n}; totals={[s.rows_total for s in streams]}", flush=True)
    for st in stats_all:
        print("   ", summarize(st), flush=True)
    with open(os.path.join(run_dir, "events.jsonl"), "a") as f:
        ts, mono = now()
        f.write(json.dumps({"ts_ns": ts, "mono_ns": mono, "phase": "_logger_stop"}) + "\n")
    N.nvmlShutdown()


def cmd_mark(args):
    rid = args.run_id
    if not rid:
        cur = os.path.join(RUNS_ROOT, "CURRENT")
        rid = open(cur).read().strip() if os.path.exists(cur) else None
    if not rid:
        sys.exit("no --run-id and no RUNS_ROOT/CURRENT")
    ts, mono = now()
    ev = {"ts_ns": ts, "mono_ns": mono, "phase": args.phase}
    if args.note:
        ev["note"] = args.note
    with open(os.path.join(RUNS_ROOT, rid, "events.jsonl"), "a") as f:
        f.write(json.dumps(ev) + "\n")
    print(json.dumps({"run_id": rid, **ev}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("run_id"); r.add_argument("--notes", default="")
    r.add_argument("--engine-buckets", action="store_true", help="also keep histogram _bucket series")
    m = sub.add_parser("mark"); m.add_argument("phase"); m.add_argument("--run-id"); m.add_argument("--note")
    a = ap.parse_args()
    {"run": cmd_run, "mark": cmd_mark}[a.cmd](a)


if __name__ == "__main__":
    main()
