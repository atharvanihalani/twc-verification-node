#!/usr/bin/env python3
"""Side logger for what logger.py does not capture (2026-11-11). Standard library only; run with system python3.

    python3 extras.py <run_dir>        # foreground; SIGTERM stops it

Writes JSON lines into <run_dir>/extras/:
    net.jsonl     1 Hz   per interface: rx_bytes, tx_bytes (/proc/net/dev), cumulative since boot
    rdma.jsonl    1 Hz   per RDMA device: port_xmit_data, port_rcv_data (units of 4 bytes), packets, cumulative
    disk.jsonl    1 Hz   per nvme disk: sectors read / written (512-byte sectors), cumulative
    nvlink.jsonl  every 5 s   raw `nvidia-smi nvlink -gt d` text: per-GPU, per-link data counters (KiB), cumulative
    procs.jsonl   every 5 s   which process holds which GPU (uuid, pid, name, MiB)
Every line carries ts_ns (UTC epoch ns) and mono_ns (CLOCK_MONOTONIC), the same clocks logger.py uses.
"""
import glob, json, os, signal, subprocess, sys, time

rd = os.path.join(sys.argv[1], "extras")
os.makedirs(rd, exist_ok=True)
stop = False


def _stop(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)
files = {n: open(os.path.join(rd, n + ".jsonl"), "a", buffering=1) for n in ("net", "rdma", "disk", "nvlink", "procs")}


def stamp():
    return {"ts_ns": time.time_ns(), "mono_ns": time.monotonic_ns()}


def net():
    out = {}
    with open("/proc/net/dev") as f:
        for line in f.readlines()[2:]:
            name, rest = line.split(":", 1)
            v = rest.split()
            if name.strip() != "lo":
                out[name.strip()] = {"rx_bytes": int(v[0]), "rx_packets": int(v[1]), "tx_bytes": int(v[8]), "tx_packets": int(v[9])}
    return out


def rdma():
    out = {}
    for d in sorted(glob.glob("/sys/class/infiniband/*/ports/1/counters")):
        dev = d.split("/")[4]
        row = {}
        for k in ("port_xmit_data", "port_rcv_data", "port_xmit_packets", "port_rcv_packets"):
            try:
                row[k] = int(open(os.path.join(d, k)).read())
            except (OSError, ValueError):
                pass
        out[dev] = row
    return out


def disk():
    out = {}
    with open("/proc/diskstats") as f:
        for line in f:
            v = line.split()
            if v[2].startswith("nvme") and "p" not in v[2][4:]:
                out[v[2]] = {"sectors_read": int(v[5]), "sectors_written": int(v[9])}
    return out


def smi(*args):
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=4).stdout
    except Exception as e:  # never let a slow nvidia-smi stop the 1 Hz streams
        return f"ERROR {e!r}"


tick = 0
next_t = time.monotonic()
while not stop:
    for name, fn in (("net", net), ("rdma", rdma), ("disk", disk)):
        try:
            files[name].write(json.dumps({**stamp(), "v": fn()}) + "\n")
        except Exception as e:
            files[name].write(json.dumps({**stamp(), "error": repr(e)}) + "\n")
    if tick % 5 == 0:
        files["nvlink"].write(json.dumps({**stamp(), "raw": smi("nvlink", "-gt", "d")}) + "\n")
        files["procs"].write(json.dumps({**stamp(), "raw": smi("--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader")}) + "\n")
    tick += 1
    next_t += 1.0
    time.sleep(max(0, next_t - time.monotonic()))
for f in files.values():
    f.close()
