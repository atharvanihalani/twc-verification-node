#!/usr/bin/env python3
"""slurm_reporter.py: report a Slurm cluster's jobs to the verification node, as declarations.

Runs on any machine that can talk to slurmctld (the controller or a login node: wherever
`sbatch` works). One per cluster; it does NOT need to run on the GPU nodes (collector.py does that).

Every few seconds it asks slurmctld for its whole job table (`scontrol show job -d -o`) and POSTs
the full list to the node's /api/declare, plus the raw text it parsed (for the "as reported" view).
It sends even when there are no jobs: that's the heartbeat, so "no jobs" and "reporter down"
look different on the node.

A job's declaration is its comment:  sbatch --comment=declared:pretraining job.sh
(types for the demo: pretraining, inference). No comment = "not declared".

Read-only: it never changes anything in Slurm. Python 3 standard library only.

    TWC_PASSWORD=... python3 slurm_reporter.py run --node https://twc-verification-node.fly.dev
    TWC_PASSWORD=... python3 slurm_reporter.py install --node https://twc-verification-node.fly.dev
    python3 slurm_reporter.py status | uninstall
    python3 slurm_reporter.py once        # parse the job table and print it; sends nothing

No Slurm on this machine? `install` offers to set up a one-box Slurm for testing (prototyping only:
controller + compute node on this machine, its GPUs as GRES, partitions train + serve). Pass
--install-slurm to skip the question. Real clusters already run Slurm; never use this there.

Times: Slurm prints its controller's local time without a timezone; this converts with the
local timezone of the machine it runs on, so run it on a machine with the controller's TZ.
"""
import argparse
import base64
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

VERSION = "v0.2"     # v0.2: install offers a one-box Slurm when there is none
POLL_S = 5
HOME = Path(os.environ.get("TWC_REPORTER_DIR", Path.home() / ".twc-slurm-reporter"))
CONFIG = HOME / "config.json"
LOG = HOME / "reporter.log"
NAME = "twc-slurm-reporter"
SYSTEMD_UNIT = Path(f"/etc/systemd/system/{NAME}.service")
SUPERVISOR_CONF = Path(f"/etc/supervisor/conf.d/{NAME}.conf")
DECLARED = re.compile(r"declared:([A-Za-z][\w-]*)")
KEY_SPLIT = re.compile(r"\s+(?=[A-Za-z][\w/:]*=)")   # a field starts at "Key=" (keys can hold / and :)
GPU_ALLOC = re.compile(r"Nodes=(\S+) CPU_IDs=\S* Mem=\S* GRES=(\S*)")   # one per node group, with -d


def log(msg):
    print(f"[reporter {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(*cmd, check=False):
    return subprocess.run(list(cmd), capture_output=True, text=True, check=check)


# ---------------------------------------------------------------- reading Slurm
def ts(v):
    """Slurm time ('2026-11-07T01:56:42', 'Unknown', 'N/A', 'None') -> unix time or None."""
    try:
        return time.mktime(datetime.strptime(v, "%Y-%m-%dT%H:%M:%S").timetuple())
    except (TypeError, ValueError):
        return None


def secs(v):
    """Slurm duration ('1-02:03:04', '02:03:04', '03:04', 'UNLIMITED') -> seconds or None."""
    if not v or not v[0].isdigit():
        return None
    days, _, rest = v.rpartition("-")
    parts = [int(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def expand_idx(s):
    """'0-3,6' -> [0, 1, 2, 3, 6]"""
    out = []
    for part in filter(None, s.split(",")):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def expand_nodes(s):
    """Slurm hostlist 'node[1-3,5],gpu7' -> ['node1', 'node2', 'node3', 'node5', 'gpu7'] (no scontrol call)."""
    out = []
    for m in re.finditer(r"([^,\[]+)(?:\[([^\]]+)\])?", s or ""):
        prefix, ranges = m.group(1), m.group(2)
        if not ranges:
            out.append(prefix)
            continue
        for part in ranges.split(","):
            a, _, b = part.partition("-")
            out += [f"{prefix}{str(i).zfill(len(a))}" for i in range(int(a), int(b or a) + 1)]
    return out


def parse_job(line):
    f = {}
    for tok in KEY_SPLIT.split(line.strip()):
        k, _, v = tok.partition("=")
        f.setdefault(k, v.strip())            # the first occurrence wins (-d repeats Nodes= per group)
    gpus = []
    for nodes, gres in GPU_ALLOC.findall(line):
        m = re.search(r"\(IDX:([^)]*)\)", gres)
        if m:
            for node in expand_nodes(nodes):
                gpus.append({"node": node, "idx": expand_idx(m.group(1))})
    req = re.search(r"gpu(?::[\w.-]+)?:(\d+)", f.get("JOB_GRES", "") or f.get("TresPerNode", ""))
    comment = f.get("Comment", "")
    comment = "" if comment == "(null)" else comment
    d = DECLARED.search(comment)
    state = f.get("JobState", "")
    return {
        "id": f.get("JobId"), "name": f.get("JobName"), "user": (f.get("UserId") or "").split("(")[0],
        "partition": f.get("Partition"), "state": state, "reason": f.get("Reason"),
        "comment": comment, "declared": d.group(1).lower() if d else None,
        "submit": ts(f.get("SubmitTime")), "start": ts(f.get("StartTime")), "end": ts(f.get("EndTime")),
        "start_is_estimate": state == "PENDING",   # a pending job's StartTime is the scheduler's guess (or Unknown)
        "time_limit_s": secs(f.get("TimeLimit")), "run_time_s": secs(f.get("RunTime")),
        "nodes": expand_nodes(f.get("NodeList")) if f.get("NodeList") not in (None, "(null)") else [],
        "gpus_requested": int(req.group(1)) * int(f.get("NumNodes", "1").split("-")[0] or 1) if req else 0,
        "gpus": gpus,   # [{node, idx: [gres index = N in /dev/nvidiaN]}], only while the job holds GPUs
    }


def read_jobs():
    r = sh("scontrol", "show", "job", "-d", "-o")
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip() or f"scontrol exited {r.returncode}")
    raw = r.stdout
    lines = [l for l in raw.splitlines() if l.startswith("JobId=")]
    return [parse_job(l) for l in lines], raw


def cluster_name():
    r = sh("scontrol", "show", "config")
    m = re.search(r"^ClusterName\s*=\s*(\S+)", r.stdout, re.M)
    return m.group(1) if m else "slurm"


# ---------------------------------------------------------------- sending
def post(url, body, password):
    headers = {"Content-Type": "application/json"}
    if password:
        headers["Authorization"] = "Basic " + base64.b64encode(f":{password}".encode()).decode()
    try:
        return json.loads(urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=10).read() or b"{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}") from None


def run(node, password):
    if not shutil.which("scontrol"):
        sys.exit("[reporter] scontrol not found: run this where Slurm's client commands work")
    urls = [u.rstrip("/") + "/api/declare" for u in node.split(",")]
    cluster, me, seq, fails = cluster_name(), socket.gethostname(), 0, 0
    log(f"{VERSION}: cluster '{cluster}', reporting to {', '.join(urls)} every {POLL_S} s")
    while True:
        t0 = time.time()
        try:
            jobs, raw = read_jobs()
            err = None
        except Exception as e:                 # still send: an error report is also a heartbeat
            jobs, raw, err = [], "", str(e)
        seq += 1
        body = json.dumps({"reporter": f"slurm_reporter.py {VERSION}", "reporter_host": me, "cluster": cluster,
                           "seq": seq, "sent_at": round(time.time(), 3), "poll_s": POLL_S,
                           "jobs": jobs, "raw": raw, "error": err}).encode()
        for url in urls:
            try:
                post(url, body, password)
                if fails:
                    log(f"reaching {url} again")
                fails = 0
            except Exception as e:
                fails += 1
                if fails <= 3 or fails % 60 == 0:
                    log(f"send to {url} failed ({e}); will keep trying")
        if err and seq % 12 == 1:
            log(f"scontrol failed: {err}")
        time.sleep(max(0.5, POLL_S - (time.time() - t0)))


# ---------------------------------------------------------------- install / uninstall / status
def detect_keepalive():
    if os.geteuid() == 0 and Path("/run/systemd/system").is_dir() and shutil.which("systemctl"):
        return "systemd"
    if shutil.which("supervisorctl") and SUPERVISOR_CONF.parent.is_dir() and os.access(SUPERVISOR_CONF.parent, os.W_OK) \
            and sh("supervisorctl", "pid").returncode == 0:
        return "supervisor"
    return "loop"


def crontab_lines():
    r = sh("crontab", "-l") if shutil.which("crontab") else None
    return r.stdout.splitlines() if r and r.returncode == 0 else []


# ---------------------------------------------------------------- one-box Slurm (prototyping)
# Same as slurm/setup_one_box.sh in the repo. Container-friendly: no systemd, no cgroups
# (proctrack/linuxproc, task/none), no accounting DB (so no sacct; scontrol keeps finished jobs 1 h).
# NOTE: without cgroups the GPU allocation is advisory: a job sees every GPU in nvidia-smi.
SLURM_SETUP = r"""
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq slurm-wlm munge > /dev/null
H=$(hostname); CPUS=$(nproc); MEM=$(( $(free -m | awk '/Mem:/{print $2}') - 8192 ))
mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm /run/munge
[ -s /etc/munge/munge.key ] || dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key 2>/dev/null
chown -R munge:munge /etc/munge /var/lib/munge /var/log/munge /run/munge; chmod 400 /etc/munge/munge.key
pgrep -x munged > /dev/null || su -s /bin/sh munge -c munged
GRES=""; [ "$NGPU" -gt 0 ] && GRES="Gres=gpu:$GTYPE:$NGPU"
cat > /etc/slurm/slurm.conf <<CONF
ClusterName=twc
SlurmctldHost=$H
SlurmUser=root
AuthType=auth/munge
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
DefMemPerCPU=4096
SchedulerParameters=bf_interval=10
GresTypes=gpu
AccountingStorageType=accounting_storage/none
JobAcctGatherType=jobacct_gather/none
JobCompType=jobcomp/filetxt
JobCompLoc=/var/log/slurm/jobcomp.log
MinJobAge=3600
ReturnToService=2
MpiDefault=none
NodeName=$H CPUs=$CPUS RealMemory=$MEM $GRES State=UNKNOWN
PartitionName=train Nodes=$H Default=YES MaxTime=INFINITE State=UP
PartitionName=serve Nodes=$H MaxTime=INFINITE State=UP
CONF
if [ "$NGPU" -gt 0 ]; then echo "NodeName=$H Name=gpu Type=$GTYPE File=/dev/nvidia[0-$((NGPU-1))]" > /etc/slurm/gres.conf; else rm -f /etc/slurm/gres.conf; fi
pkill -x slurmctld || true; pkill -x slurmd || true; sleep 1
slurmctld; slurmd
for i in $(seq 1 20); do sinfo -h -o %T 2>/dev/null | grep -q idle && break; sleep 1; done
sinfo
"""


def gpu_inventory():
    """(count, gres type) from nvidia-smi, e.g. (8, 'h100'); (0, None) without GPUs."""
    r = sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader") if shutil.which("nvidia-smi") else None
    names = [l.strip() for l in r.stdout.splitlines() if l.strip()] if r and r.returncode == 0 else []
    m = re.search(r"\b([A-Z]+\d+[A-Z]*)\b", names[0].replace("NVIDIA", "")) if names else None
    return len(names), (m.group(1).lower() if m else "gpu")


def offer_slurm(yes):
    """No scontrol here: offer to install a one-box Slurm (prototyping). Exits if declined or impossible."""
    n, gtype = gpu_inventory()
    what = f"{n}× {gtype.upper()} as GPUs" if n else "no GPUs"
    if not yes:
        if not sys.stdin.isatty():
            sys.exit("[reporter] Slurm isn't installed here. Rerun with --install-slurm to set up a one-box Slurm for testing.")
        a = input(f"Slurm isn't installed on this machine. Install a one-box Slurm here for testing "
                  f"(controller + node on this box, {what}, partitions train + serve; ~2 min)? [y/N] ").strip().lower()
        if a not in ("y", "yes"):
            sys.exit("[reporter] OK, not installing. Run this where Slurm's client commands (sbatch, scontrol) work.")
    if os.geteuid() != 0 or not shutil.which("apt-get"):
        sys.exit("[reporter] installing Slurm needs root and apt-get (Debian/Ubuntu). Install it by hand, then rerun.")
    log(f"installing Slurm (apt slurm-wlm + munge), {what} ...")
    r = subprocess.run(["bash", "-c", SLURM_SETUP], env={**os.environ, "NGPU": str(n), "GTYPE": gtype},
                       capture_output=True, text=True)
    if r.returncode != 0 or not shutil.which("scontrol"):
        sys.exit(f"[reporter] Slurm setup failed:\n{(r.stderr or r.stdout)[-1500:]}")
    log("Slurm is up:\n" + r.stdout.strip())
    log("note: no cgroups here, so a job's GPU allocation isn't enforced (nvidia-smi in a job sees every GPU). "
        "Slurm won't restart by itself if this machine restarts: rerun install.")


def install(a):
    if not shutil.which("scontrol"):
        offer_slurm(a.install_slurm)
    try:
        n = len(read_jobs()[0])
    except Exception as e:
        sys.exit(f"[reporter] scontrol can't read the job table here: {e}")
    HOME.mkdir(parents=True, exist_ok=True)
    password = os.environ.get("TWC_PASSWORD", "")
    CONFIG.write_text(json.dumps({"node": a.node, "password": password}, indent=2))
    CONFIG.chmod(0o600)                      # holds the password: owner-only
    me = HOME / "slurm_reporter.py"
    if Path(__file__).resolve() != me.resolve():
        shutil.copy(__file__, me)
    cmd = f"{sys.executable} -u {me} run"
    uninstall(quiet=True)
    how = a.via if a.via != "auto" else detect_keepalive()
    if how == "systemd":
        SYSTEMD_UNIT.write_text(f"[Unit]\nDescription=TWC Slurm reporter\nAfter=network-online.target\n\n"
                                f"[Service]\nExecStart={cmd}\nRestart=always\nRestartSec=3\n\n[Install]\nWantedBy=multi-user.target\n")
        sh("systemctl", "daemon-reload", check=True)
        sh("systemctl", "enable", "--now", NAME, check=True)
        where = f"logs: journalctl -u {NAME} -f"
    elif how == "supervisor":
        SUPERVISOR_CONF.write_text(f"[program:{NAME}]\ncommand={cmd}\nautostart=true\nautorestart=true\nstartretries=999\n"
                                   f"redirect_stderr=true\nstdout_logfile={LOG}\nstdout_logfile_maxbytes=5MB\n")
        sh("supervisorctl", "reread"); sh("supervisorctl", "update"); sh("supervisorctl", "restart", NAME)
        where = f"logs: tail -f {LOG}"
    else:
        loop = HOME / "run-loop.sh"
        loop.write_text(f"#!/bin/bash\nwhile true; do {cmd} >> {LOG} 2>&1; echo \"[loop] reporter exited, restarting\" >> {LOG}; sleep 3; done\n")
        loop.chmod(0o755)
        p = subprocess.Popen(["bash", str(loop)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        (HOME / "loop.pid").write_text(str(p.pid))
        if shutil.which("crontab"):
            lines = [l for l in crontab_lines() if NAME not in l] + [f"@reboot bash {loop}  # {NAME}"]
            subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True)
        where = f"logs: tail -f {LOG}"
    log(f"installed via {how}: {n} job(s) in the table now, reporting to {a.node}. {where}")


def uninstall(quiet=False):
    did = []
    if SYSTEMD_UNIT.exists():
        sh("systemctl", "disable", "--now", NAME); SYSTEMD_UNIT.unlink(); sh("systemctl", "daemon-reload"); did.append("systemd")
    if SUPERVISOR_CONF.exists():
        sh("supervisorctl", "stop", NAME); SUPERVISOR_CONF.unlink(); sh("supervisorctl", "reread"); sh("supervisorctl", "update")
        did.append("supervisor")
    pidfile = HOME / "loop.pid"
    if pidfile.exists():
        try:
            os.killpg(int(pidfile.read_text()), signal.SIGTERM)
        except (ProcessLookupError, ValueError, PermissionError):
            pass
        pidfile.unlink(); did.append("loop")
    if any(NAME in l for l in crontab_lines()):
        subprocess.run(["crontab", "-"], input="\n".join(l for l in crontab_lines() if NAME not in l) + "\n", text=True)
    if not quiet:
        log(f"removed: {', '.join(did) or 'nothing was installed'}")


def status():
    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    print(f"settings: {CONFIG if cfg else 'none'}  node={cfg.get('node')}")
    if (HOME / "loop.pid").exists():
        pid = int((HOME / "loop.pid").read_text())
        print(f"background loop pid {pid}: {'running' if sh('kill', '-0', str(pid)).returncode == 0 else 'NOT running'}")
    elif SYSTEMD_UNIT.exists():
        print(sh("systemctl", "status", NAME, "--no-pager", "-n", "5").stdout)
    elif SUPERVISOR_CONF.exists():
        print(sh("supervisorctl", "status", NAME).stdout.strip())
    else:
        print("not installed")
    if LOG.exists():
        print("last log lines:\n  " + "\n  ".join(LOG.read_text().splitlines()[-5:]))


def main():
    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    ap = argparse.ArgumentParser(description="Slurm -> TWC verification node reporter (see the top of this file)")
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "once", "install", "uninstall", "status"])
    ap.add_argument("--node", default=os.environ.get("TWC_NODE") or cfg.get("node") or "http://localhost:8000")
    ap.add_argument("--via", default="auto", choices=["auto", "systemd", "supervisor", "loop"])
    ap.add_argument("--install-slurm", action="store_true", help="install: if Slurm is missing, set up a one-box Slurm without asking")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.node, os.environ.get("TWC_PASSWORD") or cfg.get("password", ""))
    elif a.cmd == "once":
        jobs, _ = read_jobs()
        print(json.dumps(jobs, indent=1))
    elif a.cmd == "install":
        install(a)
    elif a.cmd == "uninstall":
        uninstall()
    else:
        status()


if __name__ == "__main__":
    main()
