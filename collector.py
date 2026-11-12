#!/usr/bin/env python3
"""collector.py: read GPU telemetry with NVML and push it to the verification node.

It has to run ON the GPU box: NVML only sees the GPUs plugged into the machine it runs on.

Try it in the foreground (Ctrl-C to stop):

    TWC_PASSWORD=... python3 collector.py run --node https://twc-verification-node.fly.dev

Install it so it stays on (survives logout, restarts if it crashes, starts on boot where possible):

    TWC_PASSWORD=... python3 collector.py install --node https://twc-verification-node.fly.dev
    python3 collector.py status
    python3 collector.py uninstall

`install` copies itself to ~/.twc-collector/, saves the settings there, installs nvidia-ml-py
if it's missing, and picks whichever "keep it running" tool this machine has:
systemd (normal Linux machines, needs root), else supervisor (some container images, e.g.
vast), else a plain background restart loop (+ a cron @reboot entry if cron exists).

How it works (two threads):

    sampler (main thread)                     sender (background thread)
    ─────────────────────                     ──────────────────────────
    every 0.1 s: read power, every GPU        every 1 s: take whatever the sampler
    every 1 s:   read the slow fields           has collected, POST it to the node
                 (energy, clocks, temps,        as JSON. If the node can't be
                  memory, GPM utilisation)      reached, print a warning and drop
    put each reading in a shared list           that batch.

Neither thread ever gives up: a failed send is dropped and the next one is tried; a failed
GPU read closes NVML, waits (1 s, 2 s, 4 s … up to 60 s) and reopens it.

Before its first batch the sender POSTs a hello to /api/hello: what this box is (host, OS, CPU,
driver, CUDA, and per GPU its name, UUID, serial, PCI id, memory, power limit). It says hello again
whenever the node answers a batch with "need_hello" (e.g. the node restarted and forgot it).

Split mode (prototyping only): `--split 2+2+4` makes ONE box report as several nodes, so a
single 8-GPU box can stand in for a small multi-node cluster. Still one process and one sampler;
each part gets its own sender and shows up as its own host (<host>-a, <host>-b, ...), with its GPUs
renumbered from 0. The hello stays honest (real hostname + each GPU's physical index). What this
fakes vs real multi-node: SPLIT_CAVEATS.md. `install --split ask` prompts for the split.

v0 deliberately leaves out signatures, nonces, prover identity and job
declarations. This is the plain "telemetry pipe" with nothing else on top.
"""
import argparse
import base64
import json
import math
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import pynvml as N  # NVML = NVIDIA Management Library, the C library behind nvidia-smi
except ImportError:     # `install` can fix this; `run` will say so
    N = None

VERSION = "v0.4"     # v0.2: says hello (box info) before its first batch; v0.3: --split; v0.4: minor number in hello
STARTED_AT = time.time()
FAST_HZ = 10        # power readings per second, per GPU
SLOW_EVERY = 10     # read the slow fields every 10th fast tick (= 1 Hz)
SEND_EVERY_S = 1.0  # how often the sender POSTs a batch
MAX_BUFFER_S = 60   # if sending stalls, keep at most this many seconds of samples
MAX_PARTS = 4       # --split: at most this many pretend nodes per box

HOME = Path(os.environ.get("TWC_COLLECTOR_DIR", Path.home() / ".twc-collector"))
CONFIG = HOME / "config.json"
LOG = HOME / "collector.log"
NAME = "twc-collector"
SYSTEMD_UNIT = Path(f"/etc/systemd/system/{NAME}.service")
SUPERVISOR_CONF = Path(f"/etc/supervisor/conf.d/{NAME}.conf")

# NVML "field" ids (from nvml.h), for values that have no dedicated function.
FI_POWER_INSTANT = 186  # instantaneous power, mW (nvmlDeviceGetPowerUsage is a ~1 s average on newer GPUs)
FI_MEMORY_TEMP = 82     # HBM temperature, °C

# GPM = GPU Performance Monitoring (Hopper and newer): utilisation counters measured
# by the GPU between two snapshots. Our column name -> NVML metric name.
GPM_METRICS = {
    "sm_util_pct": "SM_UTIL",
    "sm_occupancy_pct": "SM_OCCUPANCY",
    "tensor_util_pct": "ANY_TENSOR_UTIL",
    "dram_bw_util_pct": "DRAM_BW_UTIL",
    "pcie_tx_gb_s": "PCIE_TX_PER_SEC",           # NVML reports MB/s; we divide by 1000
    "pcie_rx_gb_s": "PCIE_RX_PER_SEC",
    "nvlink_tx_gb_s": "NVLINK_TOTAL_TX_PER_SEC",
    "nvlink_rx_gb_s": "NVLINK_TOTAL_RX_PER_SEC",
}


def log(msg):
    print(f"[collector] {msg}", flush=True)


# ---------------------------------------------------------------- reading the GPU
def read_field(h, field_id):
    """Read one NVML field value; None if this GPU doesn't support it."""
    v = N.nvmlDeviceGetFieldValues(h, [field_id])[0]
    return v.value.uiVal if v.nvmlReturn == 0 else None


def read_power_w(h):
    mw = read_field(h, FI_POWER_INSTANT)
    if mw is None:
        mw = N.nvmlDeviceGetPowerUsage(h)  # fallback for GPUs without the instant field
    return mw / 1000


class Gpm:
    """GPM gives utilisation *between two snapshots*, so we keep the previous
    snapshot for each GPU and compare it with a new one once per second."""

    def __init__(self, handles):
        self.ids = {col: getattr(N, f"NVML_GPM_METRIC_{name}") for col, name in GPM_METRICS.items()
                    if hasattr(N, f"NVML_GPM_METRIC_{name}")}
        self.prev, self.ok = {}, False
        try:
            for i, h in enumerate(handles):
                self.prev[i] = N.nvmlGpmSampleAlloc()
                N.nvmlGpmSampleGet(h, self.prev[i])
            self.cur = N.nvmlGpmSampleAlloc()
            self.ok = True
        except N.NVMLError:
            pass  # GPU or driver without GPM: we just skip these columns

    def read(self, i, h):
        if not self.ok:
            return {}
        N.nvmlGpmSampleGet(h, self.cur)
        mg = N.c_nvmlGpmMetricsGet_t()
        mg.version = N.NVML_GPM_METRICS_GET_VERSION
        mg.numMetrics = len(self.ids)
        mg.sample1, mg.sample2 = self.prev[i], self.cur
        for k, mid in enumerate(self.ids.values()):
            mg.metrics[k].metricId = mid
        N.nvmlGpmMetricsGet(mg)
        self.prev[i], self.cur = self.cur, self.prev[i]  # new snapshot becomes "previous"
        out = {}
        for k, col in enumerate(self.ids):
            if mg.metrics[k].nvmlReturn == 0 and math.isfinite(mg.metrics[k].value):  # skip NaN/inf
                v = mg.metrics[k].value
                out[col] = round(v / 1000 if col.endswith("_gb_s") else v, 3)
        return out


def read_slow(i, h, gpm):
    throttle = getattr(N, "nvmlDeviceGetCurrentClocksEventReasons", None) or N.nvmlDeviceGetCurrentClocksThrottleReasons
    row = {
        "energy_mj": N.nvmlDeviceGetTotalEnergyConsumption(h),   # cumulative since driver load
        "sm_clock_mhz": N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM),
        "throttle_reasons": int(throttle(h)),                     # bitmask; 0x4 = software power cap
        "gpu_temp_c": N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU),
        "hbm_temp_c": read_field(h, FI_MEMORY_TEMP),
        "mem_used_gb": round(N.nvmlDeviceGetMemoryInfo(h).used / 1e9, 2),
    }
    row.update(gpm.read(i, h))
    return row


def _s(v):
    return v.decode() if isinstance(v, bytes) else v


def _try(f, *a):
    try:
        return _s(f(*a))
    except Exception:
        return None


def _proc_value(path, key):
    try:
        for line in open(path):
            if line.startswith(key):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def box_info(handles, gpm):
    """What this box says about itself in its hello, before its first batch of samples."""
    gpus = []
    for i, h in enumerate(handles):
        mem, lim, pci = _try(N.nvmlDeviceGetMemoryInfo, h), _try(N.nvmlDeviceGetEnforcedPowerLimit, h), _try(N.nvmlDeviceGetPciInfo, h)
        gpus.append({
            "index": i, "name": _try(N.nvmlDeviceGetName, h), "uuid": _try(N.nvmlDeviceGetUUID, h),
            "minor": _try(N.nvmlDeviceGetMinorNumber, h),   # the N in /dev/nvidiaN: what Slurm's gres.conf counts by
            "serial": _try(N.nvmlDeviceGetSerial, h), "pci_bus_id": _s(pci.busId) if pci else None,
            "mem_total_gb": round(mem.total / 1e9, 1) if mem else None,
            "power_limit_w": round(lim / 1000) if lim else None, "vbios": _try(N.nvmlDeviceGetVbiosVersion, h),
        })
    cuda = _try(N.nvmlSystemGetCudaDriverVersion_v2) or _try(N.nvmlSystemGetCudaDriverVersion)
    mem_kb = _proc_value("/proc/meminfo", "MemTotal")
    return {
        "collector": f"collector.py {VERSION}", "started_at": round(STARTED_AT, 3),
        "hostname": socket.gethostname(), "os": platform.platform(), "python": platform.python_version(),
        "cpu": _proc_value("/proc/cpuinfo", "model name") or platform.processor() or None, "cpu_cores": os.cpu_count(),
        "ram_gb": round(int(mem_kb.split()[0]) / 1e6) if mem_kb else None,
        "driver": _try(N.nvmlSystemGetDriverVersion), "cuda": f"{cuda // 1000}.{cuda % 1000 // 10}" if cuda else None,
        "nvml": _try(N.nvmlSystemGetNVMLVersion), "gpm": gpm.ok,
        "rates_hz": {"power": FAST_HZ, "slow": FAST_HZ / SLOW_EVERY}, "gpus": gpus,
    }


class HttpError(Exception):
    def __init__(self, code, text):
        super().__init__(f"HTTP {code}: {text}")
        self.code = code


def post(url, body, headers):
    try:
        return json.loads(urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=5).read() or b"{}")
    except urllib.error.HTTPError as e:   # keep the node's explanation (e.g. "already shows 4 boxes")
        try:
            text = json.loads(e.read()).get("error") or e.reason
        except Exception:
            text = e.reason
        raise HttpError(e.code, text) from None


# ---------------------------------------------------------------- sending
def sender(node, password, host, gpus, info, batch, lock):
    url = node.rstrip("/") + "/api/collect"
    headers = {"Content-Type": "application/json"}
    if password:  # the demo site's password, sent as HTTP basic auth (username ignored)
        headers["Authorization"] = "Basic " + base64.b64encode(f":{password}".encode()).decode()
    hello_v = None   # which version of box_info this node has acknowledged (None = say hello first)
    sent, failing, next_t = 0, 0, time.monotonic()
    while True:
        next_t += SEND_EVERY_S               # fixed schedule: network time doesn't stretch the 1 s cycle
        time.sleep(max(0, next_t - time.monotonic()))
        if next_t < time.monotonic() - 10:   # fell far behind (e.g. long timeouts): resync
            next_t = time.monotonic()
        try:
            with lock:                       # grab everything collected so far, leave an empty list
                rows, batch[:] = batch[:], []
                me = dict(info)
            if not rows:
                continue
            if me.get("_v") is not None and hello_v != me["_v"]:   # introduce ourselves before any samples
                try:
                    hello = {k: v for k, v in me.items() if k != "_v"}
                    post(node.rstrip("/") + "/api/hello", json.dumps({"host": host, **hello}).encode(), headers)
                    log(f"{node}: said hello ({len(hello['gpus'])} GPU(s), driver {hello.get('driver')})")
                except HttpError as e:
                    if e.code != 404:
                        raise
                    log(f"{node}: node has no /api/hello (older version); sending samples without it")
                hello_v = me["_v"]
            body = json.dumps({"host": host, "gpus": dict(gpus), "samples": rows}, allow_nan=False).encode()
            if post(url, body, headers).get("need_hello"):   # node forgot us (restart): hello again next time
                hello_v = None
            sent += 1
            if failing:
                log(f"{node}: reachable again after {failing} failed sends")
                failing = 0
            if sent == 1 or sent % 600 == 0:
                log(f"{node}: sent batch {sent} ({len(rows)} samples, {len(body) // 1024} KB)")
        except Exception as e:               # never let the sender thread die
            failing += 1
            if failing <= 3 or failing % 60 == 0:  # don't flood the log during a long outage
                log(f"{node}: send failed ({failing} in a row), dropped this batch: {e}")


# ---------------------------------------------------------------- sampling
def open_gpus():
    N.nvmlInit()
    handles = [N.nvmlDeviceGetHandleByIndex(i) for i in range(N.nvmlDeviceGetCount())]
    names = {str(i): N.nvmlDeviceGetName(h) for i, h in enumerate(handles)}
    return handles, names, Gpm(handles)


def sample_forever(handles, gpm, batches, lock):
    tick, period = 0, 1 / FAST_HZ
    cap = int(MAX_BUFFER_S * FAST_HZ * max(1, len(handles)))
    next_t = time.monotonic()
    while True:
        t = time.time()
        rows = []
        for i, h in enumerate(handles):
            row = {"t": round(t, 3), "gpu": i, "power_w": round(read_power_w(h), 1)}
            if tick % SLOW_EVERY == 0:
                row.update(read_slow(i, h, gpm))
            rows.append(row)
        with lock:
            for batch, local in batches:     # one buffer per (node URL, part); local = {physical idx: idx in part}
                batch.extend(rows if local is None else
                             [{**r, "gpu": local[r["gpu"]]} for r in rows if r["gpu"] in local])
                if len(batch) > cap:         # that node's sender is stuck: drop the oldest, keep memory bounded
                    del batch[:len(batch) - cap]
        tick += 1
        next_t += period                     # fixed schedule, so small delays don't accumulate
        if next_t < time.monotonic() - 1:    # fell behind (e.g. a slow read): resync
            next_t = time.monotonic()
        time.sleep(max(0, next_t - time.monotonic()))


def parse_split(text):
    """'2+2+4' -> [2, 2, 4]. Raises ValueError with a readable reason."""
    try:
        parts = [int(x) for x in text.replace(",", "+").replace(" ", "+").split("+") if x]
    except ValueError:
        raise ValueError(f"'{text}' isn't in the form 4+4 or 2+2+4")
    if not parts or any(p < 1 for p in parts):
        raise ValueError(f"'{text}': every part needs at least 1 GPU")
    if len(parts) > MAX_PARTS:
        raise ValueError(f"'{text}': at most {MAX_PARTS} parts")
    return parts


def split_parts(host, split):
    """The pretend nodes: [{"host", "part", "phys": [physical GPU indices]}]; one entry with phys=None if no split."""
    if not split:
        return [{"host": host, "part": None, "phys": None}]
    out, start = [], 0
    for k, n in enumerate(parse_split(split)):
        out.append({"host": f"{host}-{chr(97 + k)}", "part": chr(97 + k), "phys": list(range(start, start + n))})
        start += n
    return out


def part_info(full, names, p, split):
    """GPU names + hello for one part: its GPUs renumbered from 0, with the physical index kept (honest hello)."""
    if p["phys"] is None:
        return dict(names), full
    gpus = {str(l): names[str(g)] for l, g in enumerate(p["phys"])}
    info = {**full, "gpus": [{**full["gpus"][g], "index": l, "physical_index": g} for l, g in enumerate(p["phys"])],
            "split": {"physical_host": full["hostname"], "part": p["part"], "of": split,
                      "physical_gpus": p["phys"], "physical_gpu_count": len(full["gpus"])}}
    return gpus, info


def run(node, host, password, split=None):
    if N is None:
        sys.exit("[collector] the Python package nvidia-ml-py is missing. "
                 "Run `python3 collector.py install ...` (it installs it) or `pip install nvidia-ml-py`.")
    nodes = [n.strip() for n in node.split(",") if n.strip()]   # several nodes: each gets every sample
    parts = split_parts(host, split)
    log(f"sending to {', '.join(n + '/api/collect' for n in nodes)} as " +
        ", ".join(f"'{p['host']}'" + (f" (GPUs {p['phys'][0]}-{p['phys'][-1]})" if p["phys"] else "") for p in parts))
    lock, batches = threading.Lock(), []
    for p in parts:
        p["gpus"], p["info"] = {}, {}
        local = None if p["phys"] is None else {g: l for l, g in enumerate(p["phys"])}
        for n in nodes:
            batch = []
            batches.append((batch, local))
            threading.Thread(target=sender, args=(n, password, p["host"], p["gpus"], p["info"], batch, lock), daemon=True).start()
    backoff = 1
    while True:
        try:
            handles, names, gpm = open_gpus()
            need = sum(len(p["phys"] or []) for p in parts)
            if need > len(handles):
                sys.exit(f"[collector] split {split} needs {need} GPUs but this box has {len(handles)}")
            full = box_info(handles, gpm)
            with lock:                       # (re)opened NVML: senders say hello again with the fresh info
                for p in parts:
                    names_p, info_p = part_info(full, names, p, split)
                    info_p = {**info_p, "_v": p["info"].get("_v", 0) + 1}
                    p["gpus"].clear(); p["gpus"].update(names_p)
                    p["info"].clear(); p["info"].update(info_p)
            log(f"{len(handles)} GPU(s): {sorted(set(names.values()))}; GPM {'on' if gpm.ok else 'unavailable'}"
                + (f"; unreported (not in split {split}): GPUs {need}-{len(handles) - 1}" if split and need < len(handles) else ""))
            backoff = 1
            sample_forever(handles, gpm, batches, lock)
        except Exception as e:               # NVML hiccup, GPU reset, driver reload…
            log(f"GPU read failed: {e!r}; reopening NVML in {backoff} s")
            try:
                N.nvmlShutdown()
            except Exception:
                pass
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)


# ---------------------------------------------------------------- install / uninstall / status
def sh(*cmd, check=False):
    return subprocess.run(list(cmd), capture_output=True, text=True, check=check)


def python_with_nvml():
    """Return a Python that can import pynvml, installing nvidia-ml-py if needed."""
    if sh(sys.executable, "-c", "import pynvml").returncode == 0:
        return sys.executable
    log("nvidia-ml-py not found; installing it")
    venv = HOME / "venv"                     # 1st choice: a private venv (doesn't touch system Python)
    if sh(sys.executable, "-m", "venv", "--system-site-packages", str(venv)).returncode == 0:
        py = str(venv / "bin" / "python")
        if sh(py, "-m", "pip", "install", "-q", "nvidia-ml-py").returncode == 0:
            return py
    for extra in ([], ["--break-system-packages"]):  # fallbacks: plain user install
        if sh(sys.executable, "-m", "pip", "install", "-q", "--user", "nvidia-ml-py", *extra).returncode == 0:
            return sys.executable
    sys.exit("[collector] couldn't install nvidia-ml-py: this Python has neither venv support nor pip "
             "(bare Ubuntu cloud images ship without both). Fix: apt-get install -y python3-venv python3-pip, "
             "then rerun. Or use the Walkthrough's setup.py, which doesn't need them.")


def detect_keepalive():
    root = os.geteuid() == 0
    if root and Path("/run/systemd/system").is_dir() and shutil.which("systemctl"):
        return "systemd"                     # systemd is actually running as the init system
    if shutil.which("supervisorctl") and SUPERVISOR_CONF.parent.is_dir() and os.access(SUPERVISOR_CONF.parent, os.W_OK) \
            and sh("supervisorctl", "pid").returncode == 0:
        return "supervisor"                  # a supervisor daemon is running and we can add programs
    return "loop"


def stop_loop():
    pidfile = HOME / "loop.pid"
    if pidfile.exists():
        try:
            os.killpg(int(pidfile.read_text()), signal.SIGTERM)
        except (ProcessLookupError, ValueError, PermissionError):
            pass
        pidfile.unlink()


def crontab_lines():
    r = sh("crontab", "-l") if shutil.which("crontab") else None
    return r.stdout.splitlines() if r and r.returncode == 0 else []


def install(a):
    HOME.mkdir(parents=True, exist_ok=True)
    saved = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    password = os.environ.get("TWC_PASSWORD") or saved.get("password", "")   # a reinstall keeps the saved one
    if not password:
        log("note: TWC_PASSWORD is not set; the node will refuse data if it has a password")
    py = python_with_nvml()
    if sh(py, "-c", "import pynvml; pynvml.nvmlInit()").returncode != 0:
        sys.exit("[collector] NVML can't start here: no NVIDIA driver or no GPUs visible to this machine/container.")
    if a.split:
        n = int(sh(py, "-c", "import pynvml as N; N.nvmlInit(); print(N.nvmlDeviceGetCount())", check=True).stdout)
        a.split = ask_split(n) if a.split == "ask" else check_split(a.split, n)
    CONFIG.write_text(json.dumps({"node": a.node, "host": a.host, "password": password, "split": a.split}, indent=2))
    CONFIG.chmod(0o600)                      # holds the password: owner-only
    me = HOME / "collector.py"
    if Path(__file__).resolve() != me.resolve():
        shutil.copy(__file__, me)
    cmd = f"{py} -u {me} run"
    uninstall(quiet=True)                    # replace any earlier install, whichever kind it was
    how = a.via if a.via != "auto" else detect_keepalive()
    if how == "systemd":
        SYSTEMD_UNIT.write_text(f"""[Unit]
Description=TWC GPU telemetry collector
After=network-online.target
Wants=network-online.target

[Service]
ExecStart={cmd}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
""")
        sh("systemctl", "daemon-reload", check=True)
        sh("systemctl", "enable", "--now", NAME, check=True)
        where = f"logs: journalctl -u {NAME} -f"
    elif how == "supervisor":
        SUPERVISOR_CONF.write_text(f"""; TWC GPU telemetry collector (written by collector.py install)
[program:{NAME}]
command={cmd}
autostart=true
autorestart=true
startretries=999
redirect_stderr=true
stdout_logfile={LOG}
stdout_logfile_maxbytes=5MB
stdout_logfile_backups=1
""")
        sh("supervisorctl", "reread"); sh("supervisorctl", "update")
        sh("supervisorctl", "restart", NAME)  # picks up a new collector.py if one was already running
        where = f"logs: tail -f {LOG}"
    else:
        loop = HOME / "run-loop.sh"
        loop.write_text(f"#!/bin/bash\n# keep collector.py running: restart it 3 s after it exits\n"
                        f"while true; do {cmd} >> {LOG} 2>&1; echo \"[loop] collector exited, restarting\" >> {LOG}; sleep 3; done\n")
        loop.chmod(0o755)
        p = subprocess.Popen(["bash", str(loop)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)  # own session: survives logout
        (HOME / "loop.pid").write_text(str(p.pid))
        if shutil.which("crontab"):           # start again after a reboot
            lines = [l for l in crontab_lines() if NAME not in l] + [f"@reboot bash {loop}  # {NAME}"]
            subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True)
            boot = "starts on boot via cron"
        else:
            boot = "won't restart after a reboot (no cron here): rerun install"
        where = f"logs: tail -f {LOG}; {boot}"
    names = ", ".join(f"'{p['host']}'" + (f" = GPUs {p['phys'][0]}-{p['phys'][-1]}" if p["phys"] else "") for p in split_parts(a.host, a.split))
    log(f"installed via {how}: sending to {a.node} as {names}. {where}")


def check_split(text, n):
    parts = parse_split(text)
    if sum(parts) > n:
        raise ValueError(f"'{text}' adds up to {sum(parts)} GPUs, but this box has {n}")
    return "+".join(map(str, parts))


def ask_split(n):
    if not sys.stdin.isatty():
        sys.exit("[collector] --split ask needs a terminal; pass the split directly instead, e.g. --split 4+4")
    while True:
        try:
            text = input(f"This box has {n} GPUs. Enter your partition split, in the form 4+4 or 2+2+4 "
                         f"(up to {MAX_PARTS} parts, GPUs assigned in order): ").strip()
            split = check_split(text, n)
        except ValueError as e:
            print(f"  {e}; try again")
            continue
        if sum(parse_split(split)) < n:
            print(f"  note: GPUs {sum(parse_split(split))}-{n - 1} won't be reported")
        return split


def uninstall(quiet=False):
    did = []
    if SYSTEMD_UNIT.exists():
        sh("systemctl", "disable", "--now", NAME); SYSTEMD_UNIT.unlink(); sh("systemctl", "daemon-reload")
        did.append("systemd")
    if SUPERVISOR_CONF.exists():
        sh("supervisorctl", "stop", NAME); SUPERVISOR_CONF.unlink()
        sh("supervisorctl", "reread"); sh("supervisorctl", "update")
        did.append("supervisor")
    if (HOME / "loop.pid").exists():
        stop_loop(); did.append("loop")
    if any(NAME in l for l in crontab_lines()):
        subprocess.run(["crontab", "-"], input="\n".join(l for l in crontab_lines() if NAME not in l) + "\n", text=True)
    if not quiet:
        log(f"removed: {', '.join(did) or 'nothing was installed'} (settings left in {HOME})")


def status():
    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    print(f"settings: {CONFIG if cfg else 'none'}  node={cfg.get('node')}  host={cfg.get('host')}"
          + (f"  split={cfg['split']}" if cfg.get("split") else ""))
    if SYSTEMD_UNIT.exists():
        print(sh("systemctl", "status", NAME, "--no-pager", "-n", "5").stdout)
    elif SUPERVISOR_CONF.exists():
        print(sh("supervisorctl", "status", NAME).stdout.strip())
    elif (HOME / "loop.pid").exists():
        pid = int((HOME / "loop.pid").read_text())
        alive = sh("kill", "-0", str(pid)).returncode == 0
        print(f"background loop pid {pid}: {'running' if alive else 'NOT running'}")
    else:
        print("not installed")
    if LOG.exists():
        print("last log lines:\n  " + "\n  ".join(LOG.read_text().splitlines()[-5:]))


# ---------------------------------------------------------------- command line
def main():
    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    ap = argparse.ArgumentParser(description="GPU telemetry collector (see the top of this file)")
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "install", "uninstall", "status", "split"])
    ap.add_argument("value", nargs="?", help="for split: how to report this box from now on, e.g. 4+4, or off for one box")
    ap.add_argument("--node", help="node URL; comma-separate several to send to all of them", default=os.environ.get("TWC_NODE") or cfg.get("node") or "http://localhost:8000")
    ap.add_argument("--host", default=cfg.get("host") or socket.gethostname(), help="name this machine shows up as")
    ap.add_argument("--via", default="auto", choices=["auto", "systemd", "supervisor", "loop"],
                    help="install only: force a keep-alive method")
    ap.add_argument("--split", help="prototyping: report this one box as several nodes, e.g. 4+4 or 2+2+4 "
                    "(install: 'ask' prompts). install without --split turns splitting off")
    a = ap.parse_args()
    if a.cmd == "run":
        split = a.split or cfg.get("split")
        if split == "ask":
            sys.exit("[collector] 'ask' only works with install; give the split, e.g. --split 4+4")
        try:
            parse_split(split) if split else None
        except ValueError as e:
            sys.exit(f"[collector] {e}")
        run(a.node, a.host, os.environ.get("TWC_PASSWORD") or cfg.get("password", ""), split)
    elif a.cmd == "install":
        try:
            install(a)
        except ValueError as e:
            sys.exit(f"[collector] {e}")
    elif a.cmd == "split":
        # Change how an installed collector reports this box (as several nodes, or as one again) with
        # everything else kept: node, name, password, keep-alive method. Does nothing if it's already that way.
        if not cfg:
            sys.exit("[collector] not installed on this box yet; run install first")
        try:
            want = None if (a.value or "off").lower() in ("off", "none", "one") else "+".join(map(str, parse_split(a.value)))
        except ValueError as e:
            sys.exit(f"[collector] {e}")
        if (cfg.get("split") or None) == want:
            log(f"already reporting as {want + ' parts' if want else 'one box'}; nothing to do")
            return
        a.split = want
        try:
            install(a)
        except ValueError as e:
            sys.exit(f"[collector] {e}")
    elif a.cmd == "uninstall":
        uninstall()
    else:
        status()


if __name__ == "__main__":
    main()
