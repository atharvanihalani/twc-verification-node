#!/usr/bin/env python3
"""Set up a GPU box for the TWC demo. Run it ON the box.

It asks its two questions first, then installs unattended:
  1. The telemetry collector (fetched from the node, kept running by collector.py install).
  2. An optional Hugging Face login: paste a read-only token (recommended), or use Hugging Face's
     browser login, which grants this box far more than reading models. Both serving models
     download without any login.
  3. vLLM in its own venv, ~/.twc-setup/vllm-venv, unless this box already has the same version.
  4. TorchTitan (the training workload: Llama 3 8B as one job across all GPUs) in
     ~/.twc-setup/torchtitan, with its own venv, a tokenizer and our recipe file. Training data
     streams from Hugging Face at run time, so nothing is downloaded here.
Steps 3 and 4 run at the same time. Their output goes to ~/.twc-setup/logs/.
Everything this script prints also goes to ~/.twc-setup/logs/setup.log.
  5. Last, it starts downloading the box's serving model (252 or 511 GB) in the background and
     exits without waiting for it. ~/.twc-setup/bin/twc-progress shows how far that has got.

--nanochat also installs nanochat, the earlier training workload (~/.twc-setup/nanochat).

    TWC_PASSWORD=... python3 setup.py --node https://twc-verification-node.fly.dev

Asks for the password if TWC_PASSWORD isn't set. Safe to rerun. Standard library only.

It needs only python3 (3.11 or newer), git and an NVIDIA driver. It does NOT need pip or the
python3-venv package: it fetches its own copy of uv (from PyPI, checksum-verified) and uses that to
make every environment. The same script is meant for Hopper and Blackwell: uv picks the PyTorch
build from the installed driver, and each step's last check fails if that build can't use the GPUs.
"""
import argparse
import base64
import getpass
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# One version on every box. MiniMax M3.1 Flash (the H100 model) needs 0.33.0; DeepSeek V4.1 Flash
# (H200 and above) needs 0.31 or newer. Our Nov 3-4 H200 serving runs were on 0.32.0.
# [bench] adds pandas and friends: without it `vllm bench serve --dataset-name burstgpt` (the Walkthrough's
# load) stops with "Please install vllm[bench]".
VLLM_SPEC = "vllm[bench]==0.33.0"
BLACKWELL_MIN_CUDA = (12, 8)  # per vLLM's install docs

HOME = Path.home() / ".twc-setup"
VENV = HOME / "vllm-venv"
PYTHON = "3.12"                           # the Python uv fetches for the vLLM and TorchTitan environments
COLLECTOR_VENV = HOME / "collector-venv"
LOGS = HOME / "logs"
SETUP_LOG = LOGS / "setup.log"            # a copy of everything this script prints
INSTALL_GB = 30                           # both environments plus uv's download cache, with some slack
MODEL_GB = {"H100": 252, "other": 511}    # MiniMax M3.1 Flash on H100; DeepSeek V4.1 Flash elsewhere
MODEL_REPO = {"H100": "MiniMaxAI/MiniMax-M3.1-Flash", "other": "deepseek-ai/DeepSeek-V4.1-Flash-1017"}
DOWNLOAD = HOME / "model-download.json"   # what the background model download is, for `twc-progress`
PROGRESS_CMD = HOME / "bin" / "twc-progress"
# Prefill/decode across the two halves of a box (Walkthrough, scenario C) needs vLLM's hand-off library and a
# small proxy that sends each request to the prefill server and then the decode server. Both ran on H200.
NIXL_SPEC = "nixl==1.5.2"
PD_PROXY = HOME / "pd_proxy.py"
PD_PROXY_URL = "https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
# Runs inside a throwaway environment that has Hugging Face's library. The token arrives in the environment (not
# on the command line, where `ps` would show it), is checked with Hugging Face, and saved by their own login().
HF_TOKEN_LOGIN = """
import os, sys
from huggingface_hub import login, whoami
t = os.environ.pop("TWC_HF_TOKEN")
try:
    me = whoami(token=t)
except Exception as e:
    sys.exit("Hugging Face did not accept that token: " + (str(e).strip().splitlines() or [repr(e)])[0][:200])
role = ((me.get("auth") or {}).get("accessToken") or {}).get("role") or "unknown"
login(token=t)
print("TWC_ROLE=" + role + " TWC_USER=" + str(me.get("name")))
"""
# Settings some images export to steer every uv install on the box, meant for their own system Python. vast's
# vLLM image forces nvidia-nccl-cu13==2.30.7 through UV_OVERRIDE; TorchTitan's PyTorch needs a newer one and then
# fails to import ("undefined symbol: ncclReduceConfig"). Our environments are separate, so we don't apply them.
UV_ENV_IGNORED = ("UV_OVERRIDE", "UV_CONSTRAINT", "UV_BUILD_CONSTRAINT", "UV_INDEX_STRATEGY", "UV_PRERELEASE",
                  "UV_RESOLUTION", "UV_EXCLUDE_NEWER")
SPLIT_CMD = HOME / "bin" / "twc-split"    # `twc-split 4+4` / `twc-split off`: how this box reports itself

TT_REPO = "https://github.com/pytorch/torchtitan"
TT_REV = "5a49006c3d793d428a666a3a9d1ca0878222ce25"  # 2026-11-09: the commit our H200 trial ran
TT_DIR = HOME / "torchtitan"
TT_TORCH_INDEX = "https://download.pytorch.org/whl/nightly/cu132"  # TorchTitan tracks PyTorch nightlies
# The Llama 3 recipes need Flash Attention, which TorchTitan's requirements.txt leaves out.
# Hopper: ran on H200. Blackwell: taken from TorchTitan's scripts/install_rl_env.sh, not run by us.
TT_FLASH_ATTN = {"Hopper": ["flash-attn-3", "--extra-index-url", TT_TORCH_INDEX],
                 "Blackwell": ["flash-attn-4[cu13]>=4.0.0b38"]}
# An ungated copy of the Llama 3.1 tokenizer. The original (meta-llama/Llama-3.1-8B) needs an access request.
TT_TOKENIZER_URL = "https://huggingface.co/unsloth/Meta-Llama-3.1-8B/resolve/main"
TT_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
TT_VERIFY = ("import json, os, shutil, sysconfig, torch, twc_recipes; twc_recipes.llama3_8b_twc(); "
             "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, "
             "'cuda_ok': torch.cuda.is_available(), 'gpus': torch.cuda.device_count(), "
             "'headers': os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')), "
             "'compiler': bool(shutil.which('gcc') or shutil.which('cc'))}))")
# TorchTitan has no command-line flags for its settings: they live in a Python recipe. This one reads
# them from the environment, so the Walkthrough can show them as knobs. Written to TT_DIR/twc_recipes.py.
TT_RECIPE = '''"""TWC recipe: Llama 3 8B as one FSDP job across all GPUs, with knobs read from the environment.

  TWC_BATCH        sequences per GPU per step   (default 1)
  TWC_SEQ_LEN      tokens per sequence          (default 8192; TorchTitan's own default is 131072)
  TWC_REMAT        default | selective | full   (activation checkpointing. default keeps every activation; selective
                                                recomputes only the feed-forward's wide layer; full recomputes
                                                everything: least memory, a bit slower)
  TWC_REMAT_RECOMPUTE  with selective: comma-separated regions to recompute (default feed_forward.w13.linear)
  TWC_CKPT_EVERY   save a checkpoint every N steps (default: off). Each save holds the model and the optimizer,
                   about 96 GB for this model, under <output-dir>/checkpoint; the newest TWC_CKPT_KEEP (default 2,
                   the smallest TorchTitan allows) are kept, so leave about 200 GB free
  TWC_CKPT_ASYNC   disabled | async | async_with_pinned_mem   (default disabled: training waits for each save)
  TWC_STEPS        training steps               (default 1000)
  TWC_LOG_FREQ     print metrics every N steps  (default 10)
  TWC_MEM_CAP_GIB  optional: cap GPU memory per process, to emulate a smaller card (79.6 for an 80 GB H100)

  MODULE=twc_recipes CONFIG=llama3_8b_twc ./run_train.sh --output-dir ~/tt-run
"""
import os

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.distributed.activation_remat import ActivationRematConfig, FullActivationRematConfig
from torchtitan_recipes.data import c4_document_capacity
from torchtitan_recipes.models.llama3 import llama3_8b


def llama3_8b_twc():
    seq_len = int(os.environ.get("TWC_SEQ_LEN", "8192"))
    tokens = seq_len * int(os.environ.get("TWC_BATCH", "1"))
    cfg = llama3_8b(seq_len=seq_len)
    cfg.training.num_tokens_per_microbatch_per_dp_rank = tokens
    cfg.training.steps = int(os.environ.get("TWC_STEPS", "1000"))
    cfg.dataloader.max_num_documents = c4_document_capacity(tokens)
    remat = os.environ.get("TWC_REMAT", "default")
    if remat == "full":
        cfg.activation_remat = FullActivationRematConfig()
    elif remat == "selective":  # save everything except the listed regions; TorchTitan ignores a name that matches nothing
        regions = os.environ.get("TWC_REMAT_RECOMPUTE", "feed_forward.w13.linear")
        cfg.activation_remat = ActivationRematConfig(save_regions=["*"], recompute_regions=regions.split(","))
    elif remat != "default":
        raise ValueError(f"TWC_REMAT must be default, selective or full, not {remat!r}")
    if os.environ.get("TWC_CKPT_EVERY"):
        cfg.checkpointer = CheckpointManager.Config(
            interval=int(os.environ["TWC_CKPT_EVERY"]), keep_latest_k=int(os.environ.get("TWC_CKPT_KEEP", "2")),
            async_mode=os.environ.get("TWC_CKPT_ASYNC", "disabled"))
    cfg.profiler.enable_profiling = False
    cfg.metrics.log_freq = int(os.environ.get("TWC_LOG_FREQ", "10"))
    cap = os.environ.get("TWC_MEM_CAP_GIB")
    if cap and os.environ.get("LOCAL_RANK") is not None:  # torchrun sets LOCAL_RANK; each rank uses that GPU
        import torch
        dev = int(os.environ["LOCAL_RANK"])
        total = torch.cuda.get_device_properties(dev).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) / total), dev)
    return cfg
'''

NANOCHAT_REPO = "https://github.com/karpathy/nanochat"
NANOCHAT_REV = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"  # pinned so everyone gets the same training code
NANOCHAT_DIR = HOME / "nanochat"
TOKENIZER_URL = "https://huggingface.co/karpathy/nanochat-d32/resolve/main"
TOKENIZER_DIR = Path.home() / ".cache" / "nanochat" / "tokenizer"  # where nanochat looks by default
TOKENIZER_FILES = ("tokenizer.pkl", "token_bytes.pt")

NANOCHAT_VERIFY = ("import json, torch; from nanochat.tokenizer import get_tokenizer, get_token_bytes; t = get_tokenizer(); "
                   "print(json.dumps({'vocab': t.get_vocab_size(), 'bos': t.get_bos_token_id(), 'token_bytes': len(get_token_bytes('cpu')), "
                   "'torch': torch.__version__, 'cuda': torch.version.cuda, 'cuda_ok': torch.cuda.is_available(), 'gpus': torch.cuda.device_count()}))")

# Printed as one JSON line by the venv's own python after installing.
VERIFY = ("import json, torch, vllm; print(json.dumps({'vllm': vllm.__version__, 'torch': torch.__version__, "
          "'cuda': torch.version.cuda, 'cuda_ok': torch.cuda.is_available(), 'gpus': torch.cuda.device_count()}))")


_print_lock = threading.Lock()
_setup_log = None  # SETUP_LOG, opened in main()
UV = None  # path to uv, set once in main()


def echo(data):
    """Bytes to the terminal and to setup.log."""
    with _print_lock:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        if _setup_log:
            _setup_log.write(data)
            _setup_log.flush()


def log(msg):
    echo(f"[setup] {msg}\n".encode())


def die(msg):
    log(msg)
    sys.exit(1)


class StepFailed(Exception):
    pass


def run(*cmd, env=None, cwd=None, out=None):
    """Run a command. With `out` (a log file), its output goes there, not to the terminal."""
    cmd = [str(c) for c in cmd]
    if out is None:
        log("$ " + " ".join(cmd))
        # Through a pipe, so the output reaches setup.log as well as the terminal. The command keeps the terminal
        # as its input, so one that asks something (collector.py install --split ask) still can.
        p = subprocess.Popen(cmd, env={**(env or os.environ), "PYTHONUNBUFFERED": "1"}, cwd=cwd,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while chunk := os.read(p.stdout.fileno(), 4096):
            echo(chunk)
        if p.wait() != 0:
            raise StepFailed(f"command failed: {cmd[0]}")
        return
    with open(out, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        code = subprocess.run(cmd, env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
    if code != 0:
        tail = "\n".join(Path(out).read_text(errors="replace").splitlines()[-15:])
        raise StepFailed(f"command failed: {' '.join(cmd[:4])} ...\n--- last lines of {out} ---\n{tail}")


def family(gpu_name):
    n = gpu_name.upper()
    if any(x in n for x in ("H100", "H200", "H800")):
        return "Hopper"
    if any(x in n for x in ("B100", "B200", "B300")):  # also matches GB200 / GB300
        return "Blackwell"
    return "other"


def detect_gpus():
    """Returns (names, families). Exits if this machine has no working NVIDIA driver."""
    if not shutil.which("nvidia-smi"):
        die("nvidia-smi not found: run this on a machine with NVIDIA GPUs.")
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    rows = [[c.strip() for c in line.split(",")] for line in r.stdout.splitlines() if line.strip()]
    if r.returncode != 0 or not rows:
        die(f"nvidia-smi can't see any GPUs: {(r.stderr or r.stdout).strip()}")
    names = [row[0] for row in rows]
    families = {family(n) for n in names}
    log(f"{len(names)} GPU(s): {', '.join(sorted(set(names)))}; driver {rows[0][1]}; "
        f"{'/'.join(sorted(families))}; CPU {platform.machine()}")
    busy = [i for i, row in enumerate(rows) if row[2].isdigit() and int(row[2]) > 1024]
    if busy:
        log(f"WARNING: GPU(s) {', '.join(map(str, busy))} already have memory in use. Something else is running on them, "
            "and the Walkthrough's workloads will not fit beside it.")
        log("         On vast's vLLM image that is its auto-started server: supervisorctl stop vllm; pkill -9 -f \"[V]LLM::\"")
    return names, families


def free_gb(path):
    p = Path(path).expanduser()
    while not p.exists():  # the folder may not exist yet: measure the disk it will land on
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def preflight(names):
    """Fail early, with the fix, on the things that otherwise fail late and cryptically."""
    if sys.version_info < (3, 11):
        log(f"WARNING: this is Python {platform.python_version()}. vLLM needs 3.10 or newer and TorchTitan 3.11 or newer.")
    if not shutil.which("git"):
        log("WARNING: git isn't installed, so the TorchTitan step will fail. On Ubuntu: apt-get install -y git")
    room = free_gb(HOME)
    if room < INSTALL_GB:
        die(f"only {room:.0f} GB free where {HOME} lives; the installs need about {INSTALL_GB} GB. "
            "Rent the box with a bigger disk (most providers fix the size at rent time).")
    model = MODEL_GB["H100" if any("H100" in n.upper() for n in names) else "other"]
    cache = os.environ.get("HF_HOME") or "~/.cache/huggingface"
    room = free_gb(cache)
    if room < model + INSTALL_GB:
        log(f"WARNING: {room:.0f} GB free where models download ({cache}); this box's serving model is {model} GB. "
            "Inference will not fit. Training will. To use a bigger disk: export HF_HOME=/path/on/it")


def ensure_uv():
    """Path to uv. If this box has none, fetch the official wheel from PyPI, check its hash and unpack the binary.

    uv makes the environments and installs into them, so the box needs neither pip nor python3-venv
    (bare Ubuntu cloud images ship without both).
    """
    found = shutil.which("uv")
    if found:
        return found
    uv = HOME / "bin" / "uv"
    if uv.exists():
        return str(uv)
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(platform.machine().lower())
    by_hand = "Install uv by hand (https://docs.astral.sh/uv/getting-started/installation/) and rerun."
    if not arch:
        raise StepFailed(f"don't know which uv build suits a {platform.machine()} CPU. {by_hand}")
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/uv/json", timeout=30) as resp:
            meta = json.load(resp)
        wheels = [f for f in meta["urls"] if f["filename"].endswith(".whl") and "manylinux" in f["filename"]
                  and f"_{arch}." in f["filename"]]
        if not wheels:
            raise StepFailed(f"PyPI has no manylinux {arch} build of uv {meta['info']['version']}. {by_hand}")
        with urllib.request.urlopen(wheels[0]["url"], timeout=120) as resp:
            blob = resp.read()
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        raise StepFailed(f"couldn't download uv from PyPI: {e}. {by_hand}")
    if hashlib.sha256(blob).hexdigest() != wheels[0]["digests"]["sha256"]:
        raise StepFailed(f"the uv download doesn't match PyPI's checksum. {by_hand}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        member = next((n for n in z.namelist() if n.endswith("/scripts/uv")), None)
        if not member:
            raise StepFailed(f"unexpected layout inside {wheels[0]['filename']}. {by_hand}")
        uv.parent.mkdir(parents=True, exist_ok=True)
        uv.write_bytes(z.read(member))
    uv.chmod(0o755)
    log(f"fetched uv {meta['info']['version']} to {uv}")
    return str(uv)


def make_venv(path, out=None, own_python=True):
    """Make a venv with uv. By default it gets uv's own standalone Python, not the system one.

    The system Python on a bare cloud image has no header files (no python3-dev), and PyTorch
    compiles small C extensions at run time that need them: TorchTitan's first training step
    failed on exactly that. uv's Python ships its headers, and is the same version on every box.
    """
    if not (Path(path) / "bin" / "python").exists():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        which = ["--python", PYTHON, "--python-preference", "only-managed"] if own_python else ["--python", sys.executable]
        run(UV, "venv", *which, path, out=out)
    return Path(path) / "bin" / "python"


def uv_install(py, *args, out=None, cwd=None):
    run(UV, "pip", "install", "--python", py, *args, out=out, cwd=cwd)


def fetch_collector(node, password):
    req = urllib.request.Request(node.rstrip("/") + "/collector.py")
    req.add_header("Authorization", "Basic " + base64.b64encode(f":{password}".encode()).decode())
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
    except urllib.error.URLError as e:
        raise StepFailed(f"couldn't download the collector from {node}: {e}")
    if b"def install" not in body:  # a bad password lands on the login page, which is still a 200
        raise StepFailed("the node didn't return the collector script: wrong password?")
    return body


def setup_collector(a):
    log("--- collector ---")
    password = os.environ.get("TWC_PASSWORD") or getpass.getpass("TWC password: ")
    body = fetch_collector(a.node, password)
    # Give the collector a Python that already has the NVIDIA library, so it needn't install anything itself.
    py = make_venv(COLLECTOR_VENV, own_python=False)  # the system Python is fine here: nothing gets compiled
    uv_install(py, "-q", "nvidia-ml-py")
    with tempfile.TemporaryDirectory() as d:  # collector.py install copies itself somewhere permanent
        script = Path(d) / "collector.py"
        script.write_bytes(body)
        cmd = [py, script, "install", "--node", a.node]
        if a.host:
            cmd += ["--host", a.host]
        if a.split:
            cmd += ["--split", a.split]
        run(*cmd, env={**os.environ, "TWC_PASSWORD": password})
    # The Walkthrough's scenarios switch the box between one 8-GPU unit and two 4-GPU ones with this.
    SPLIT_CMD.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_CMD.write_text(f'#!/bin/sh\n# twc-split 4+4 | off : report this box as several nodes, or as one again\n'
                         f'exec "{py}" "{Path.home() / ".twc-collector" / "collector.py"}" split "$@"\n')
    SPLIT_CMD.chmod(0o755)


def hf_token_login():
    """Log in with a token the user pastes. Returns a one-line outcome. The token is never printed or logged."""
    log("Make a token at https://huggingface.co/settings/tokens : choose the type \"Read\", or \"Fine-grained\" with")
    log("only \"Read access to contents of all public gated repos you can access\" ticked. Then paste it here.")
    try:
        token = getpass.getpass("[setup] Token (it won't show as you paste; Enter to skip): ").strip()
    except EOFError:
        token = ""
    if not token:
        return None
    r = subprocess.run([UV, "run", "--no-project", "-q", "--with", "huggingface_hub", "python", "-c", HF_TOKEN_LOGIN],
                       env={**os.environ, "TWC_HF_TOKEN": token}, capture_output=True, text=True)
    del token
    line = next((l for l in r.stdout.splitlines() if l.startswith("TWC_ROLE=")), None)
    if r.returncode != 0 or not line:
        log((r.stderr.strip().splitlines() or ["the login didn't complete"])[-1][:300])
        return "not logged in"
    role, user = (part.partition("=")[2] for part in line.split(" ", 1))
    if role == "write":
        log(f"logged in as {user}. WARNING: that is a write token, which can change your repos. It works, but a "
            "read-only one is all this box needs: you can swap it at huggingface.co/settings/tokens.")
        return "logged in (write token)"
    kind = {"read": "a read-only token", "fineGrained": "a fine-grained token (it can do what you ticked, no more)"}.get(role, "a token")
    log(f"logged in as {user} with {kind}")
    return f"logged in ({'read-only' if role == 'read' else 'fine-grained' if role == 'fineGrained' else 'token'})"


def setup_hf_login(a):
    """Optional. A pasted read-only token, or Hugging Face's own browser login."""
    log("--- Hugging Face login (optional) ---")
    hf = [UV, "tool", "run", "-q", "--from", "huggingface_hub", "hf"]  # runs the hf command without installing it anywhere
    later = f"To log in later: {VENV}/bin/hf auth login --token <a read-only token>"
    if os.environ.get("HF_TOKEN") or subprocess.run(hf + ["auth", "whoami"], capture_output=True).returncode == 0:
        log("already logged in to Hugging Face")
        return "already logged in"
    if not sys.stdin.isatty():
        log(f"not an interactive terminal; skipping. {later}")
        return "skipped"
    log("You can skip this: both serving models download without a login. Logging in raises download limits")
    log("and gives access to gated models. The token stays on this box; the verification node never sees it.")
    log("  1  Paste a read-only token (recommended: it can only read models)")
    log("  2  Log in through a browser. Hugging Face's consent screen then grants this box far more than")
    log("     reading models (managing repos, billing, jobs and so on), and has no narrower setting.")
    try:
        answer = input("[setup] 1, 2, or Enter to skip: ").strip().lower()
    except EOFError:  # Ctrl-D, or the terminal went away
        print()
        answer = ""
    if answer == "1":
        outcome = hf_token_login()
        if outcome:
            return outcome
    elif answer == "2":
        log("On the consent screen, pick \"Do not grant access to any of your organizations\".")
        if subprocess.run(hf + ["auth", "login"]).returncode != 0:  # talks to the terminal directly, so it isn't in setup.log
            log(f"the login didn't complete; carrying on without it. {later}")
            return "not logged in"
        log("logged in through the browser")
        return "logged in (browser)"
    log(f"skipped. {later}")
    return "skipped"


def parse_version(v):
    return tuple(int(x) for x in v.split(".")[:2])


def installed_vllm_version(exe):
    """Version of a vllm already on this box, or None if it won't say."""
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = r.stdout.strip().splitlines()
    return lines[-1].strip() if r.returncode == 0 and lines else None


def can_build_kernels():
    """vLLM compiles its attention kernels on the box at first start. That needs the CUDA compiler and ninja."""
    path = [str(VENV / "bin"), "/usr/local/cuda/bin", os.environ.get("PATH", "")]  # as the Walkthrough sets it
    path += [os.path.join(os.environ[v], "bin") for v in ("CUDA_HOME", "CUDA_PATH") if os.environ.get(v)]
    return [tool for tool in ("nvcc", "ninja") if not shutil.which(tool, path=os.pathsep.join(path))]


def setup_vllm(a, families, out=None):
    existing = shutil.which("vllm")
    if existing and Path(existing).resolve().parent == (VENV / "bin").resolve():
        existing = None  # that's our own venv from an earlier run; fall through and make sure it's the pin
    if existing and not a.force_vllm:
        have = installed_vllm_version(existing)
        want = a.vllm_spec.partition("==")[2]
        if not want or have == want:
            log(f"vLLM: {have or '(version unknown)'} is already installed at {existing}; leaving it alone")
            return "already installed"
        # e.g. vast's image ships its own vLLM. Leave that one alone and install ours beside it.
        log(f"vLLM: {have or '(version unknown)'} is already installed at {existing}, but this demo expects {want}.")
        log(f"vLLM: leaving it alone and installing {want} in {VENV}. To use ours: export PATH={VENV}/bin:$PATH")
    if sys.version_info < (3, 10):
        raise StepFailed(f"vLLM needs Python 3.10 or newer; this is {platform.python_version()}")
    py = make_venv(VENV, out)
    uv_install(py, a.vllm_spec, NIXL_SPEC, "--torch-backend=auto", out=out)
    try:  # vLLM's own example proxy, pinned to the release we install
        download(PD_PROXY_URL, PD_PROXY)
    except StepFailed as e:
        log(f"vLLM: WARNING: couldn't fetch the prefill/decode proxy ({e}); only the Walkthrough's scenario C needs it")

    missing = can_build_kernels()
    if missing:  # e.g. a driver-only image with no CUDA toolkit
        log(f"vLLM: this box has no {' or '.join(missing)}, so vLLM can't build its attention kernels at first start; "
            "fetching prebuilt ones instead")
        try:
            # It installs two wheels with uv or pip. Our venv has no pip, so make sure it can find our uv.
            run(VENV / "bin" / "vllm", "download-kernels", out=out,
                env={**os.environ, "PATH": os.pathsep.join([str(VENV / "bin"), str(Path(UV).parent), os.environ.get("PATH", "")])})
        except StepFailed as e:
            log(f"vLLM: WARNING: couldn't fetch prebuilt kernels ({str(e).splitlines()[0]}). Servers may start without "
                f"them and run slowly. Install the CUDA toolkit, or retry: {VENV}/bin/vllm download-kernels")

    r = subprocess.run([str(py), "-c", VERIFY], capture_output=True, text=True)
    if r.returncode != 0:
        raise StepFailed("vLLM installed but won't import:\n" + r.stderr.strip()[-800:])
    info = json.loads(r.stdout.strip().splitlines()[-1])
    log(f"vLLM: {info['vllm']}, torch {info['torch']}, CUDA {info['cuda']}, sees {info['gpus']} GPU(s)")
    if not info["cuda_ok"]:
        raise StepFailed("this PyTorch build can't use the GPUs (a CPU-only build, or a driver mismatch)")
    if "Blackwell" in families and parse_version(info["cuda"]) < BLACKWELL_MIN_CUDA:
        raise StepFailed(f"Blackwell needs CUDA {'.'.join(map(str, BLACKWELL_MIN_CUDA))} or newer; "
                         f"this PyTorch build uses {info['cuda']}")
    log(f"vLLM: ready. {VENV}/bin/vllm serve <model>")


def download(url, dest):
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
    except urllib.error.URLError as e:
        raise StepFailed(f"couldn't download {url}: {e}")
    tmp.rename(dest)


def setup_torchtitan(families, out=None):
    if not shutil.which("git"):
        raise StepFailed("git isn't installed on this box. On Ubuntu: apt-get install -y git")
    if sys.version_info < (3, 11):
        raise StepFailed(f"TorchTitan needs Python 3.11 or newer; this is {platform.python_version()}")
    if TT_DIR.exists():
        run("git", "-C", TT_DIR, "fetch", "--quiet", "origin", out=out)
    else:
        HOME.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--quiet", TT_REPO, TT_DIR, out=out)
    run("git", "-C", TT_DIR, "checkout", "--quiet", TT_REV, out=out)

    py = make_venv(TT_DIR / ".venv", out)
    uv_install(py, "--pre", "torch", "--index-url", TT_TORCH_INDEX, out=out)
    uv_install(py, "-r", "requirements.txt", out=out, cwd=TT_DIR)
    known = sorted(families & set(TT_FLASH_ATTN))
    for fam in known:
        uv_install(py, *TT_FLASH_ATTN[fam], out=out)
    if not known:
        log("TorchTitan: WARNING: not a Hopper or Blackwell GPU, so Flash Attention wasn't installed; the Llama 3 recipe needs it")

    assets = TT_DIR / "assets" / "hf" / "Llama-3.1-8B"  # where the Llama 3 recipes look
    assets.mkdir(parents=True, exist_ok=True)
    for name in TT_TOKENIZER_FILES:
        if not (assets / name).exists():
            download(f"{TT_TOKENIZER_URL}/{name}", assets / name)
    (TT_DIR / "twc_recipes.py").write_text(TT_RECIPE)

    r = subprocess.run([str(py), "-c", TT_VERIFY], capture_output=True, text=True, cwd=TT_DIR)
    if r.returncode != 0:
        raise StepFailed("TorchTitan installed but the check failed:\n" + r.stderr.strip()[-800:])
    info = json.loads(r.stdout.strip().splitlines()[-1])
    log(f"TorchTitan: torch {info['torch']}, CUDA {info['cuda']}, sees {info['gpus']} GPU(s)")
    if not info["cuda_ok"]:
        raise StepFailed("TorchTitan's PyTorch build can't use the GPUs (a CPU-only build, or a driver mismatch)")
    # Training compiles a little C on its first step. Catch a box that can't, here, not five minutes into a run.
    if not info["compiler"]:
        raise StepFailed("no C compiler on this box, and training needs one. On Ubuntu: apt-get install -y build-essential")
    if not info["headers"]:
        raise StepFailed("this environment's Python has no header files (Python.h), and training needs them. "
                         "On Ubuntu: apt-get install -y python3-dev")
    log(f"TorchTitan: ready. {TT_DIR}")


def setup_nanochat(families, out=None):
    if not shutil.which("git"):
        raise StepFailed("git isn't installed on this box. On Ubuntu: apt-get install -y git")
    if NANOCHAT_DIR.exists():
        run("git", "-C", NANOCHAT_DIR, "fetch", "--quiet", "origin", out=out)
    else:
        HOME.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--quiet", NANOCHAT_REPO, NANOCHAT_DIR, out=out)
    run("git", "-C", NANOCHAT_DIR, "checkout", "--quiet", NANOCHAT_REV, out=out)
    run(UV, "sync", "--extra", "gpu", cwd=NANOCHAT_DIR, out=out)

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        if not (TOKENIZER_DIR / name).exists():
            download(f"{TOKENIZER_URL}/{name}", TOKENIZER_DIR / name)

    r = subprocess.run([str(NANOCHAT_DIR / ".venv" / "bin" / "python"), "-c", NANOCHAT_VERIFY],
                       capture_output=True, text=True, cwd=NANOCHAT_DIR)
    if r.returncode != 0:
        raise StepFailed("nanochat installed but the check failed:\n" + r.stderr.strip()[-800:])
    info = json.loads(r.stdout.strip().splitlines()[-1])
    log(f"nanochat: tokenizer {info['vocab']:,} tokens; torch {info['torch']}, CUDA {info['cuda']}, sees {info['gpus']} GPU(s)")
    if not info["cuda_ok"]:
        raise StepFailed("nanochat's PyTorch build can't use the GPUs (a CPU-only build, or a driver mismatch)")
    if "Blackwell" in families and parse_version(info["cuda"]) < BLACKWELL_MIN_CUDA:
        raise StepFailed(f"Blackwell needs CUDA {'.'.join(map(str, BLACKWELL_MIN_CUDA))} or newer; "
                         f"nanochat's PyTorch build uses {info['cuda']}")
    log(f"nanochat: ready. {NANOCHAT_DIR}")


def hub_dir():
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface").expanduser() / "hub"


def dir_bytes(path):
    r = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True)
    return int(r.stdout.split()[0]) if r.returncode == 0 and r.stdout.split() else 0


def model_bytes(folder):
    """Bytes of one model in the Hugging Face cache: its finished files plus the ones still arriving.

    Finished files are reached through the model's snapshot links, because newer versions of the Hugging Face
    library keep them in a folder shared by all models (hub/blobs), not in the model's own. Files still arriving
    sit in the model's folder as *.incomplete. Counting only the model's folder read 15 GB when 150 GB was on disk.
    """
    folder = Path(folder)
    r = subprocess.run(["du", "-sbL", str(folder / "snapshots")], capture_output=True, text=True)  # -L: follow the links
    done = int(r.stdout.split()[0]) if r.stdout.split() and r.stdout.split()[0].isdigit() else 0
    arriving = 0
    for f in folder.rglob("*.incomplete"):
        try:
            arriving += f.stat().st_size
        except OSError:  # finished and renamed while we were looking
            pass
    return done + arriving


def repo_bytes(repo):
    """Total size of a Hugging Face model repo, or None if the API won't say."""
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=30) as resp:
            return sum(f.get("size") or 0 for f in json.load(resp).get("siblings", [])) or None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def start_model_download(repo, guess_gb):
    """Start downloading the serving model in the background, detached, so it outlives this script and the terminal.

    It goes into the normal Hugging Face cache, which is where `vllm serve <repo>` looks. Returns a one-line status.
    """
    total = repo_bytes(repo) or int(guess_gb * 1e9)
    room = free_gb(hub_dir())
    if room < total / 1e9 + INSTALL_GB:
        return f"not started: {total / 1e9:.0f} GB won't fit in the {room:.0f} GB free where models download"
    if DOWNLOAD.exists():
        old = json.loads(DOWNLOAD.read_text())
        if old.get("repo") == repo and pid_alive(old.get("pid")):
            return "already running from an earlier setup"
    own = VENV / "bin" / "hf"
    hf = [str(own)] if own.exists() else [UV, "tool", "run", "-q", "--from", "huggingface_hub", "hf"]
    out, done = LOGS / "model-download.log", HOME / "model-download.exit"
    done.unlink(missing_ok=True)
    script = f'{" ".join(hf)} download {repo} >> "{out}" 2>&1; echo $? > "{done}"'
    p = subprocess.Popen(["bash", "-c", script], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)  # own session: survives logout
    folder = hub_dir() / ("models--" + repo.replace("/", "--"))  # where the Hugging Face cache keeps this model
    DOWNLOAD.write_text(json.dumps({"repo": repo, "pid": p.pid, "total_bytes": total, "started": time.time(),
                                    "dir": str(folder), "dir_bytes_at_start": model_bytes(folder),
                                    "hub": str(hub_dir()), "hub_bytes_at_start": dir_bytes(hub_dir()),
                                    "log": str(out), "exit_file": str(done)}, indent=1))
    return f"started in the background ({total / 1e9:.0f} GB)"


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def progress(watch=False, stop=False):
    """`twc-progress`: how far the background model download has got. Reads DOWNLOAD; changes nothing unless stop."""
    if not DOWNLOAD.exists():
        print("No model download was started on this box.")
        return 1
    d = json.loads(DOWNLOAD.read_text())
    total, exit_file = d["total_bytes"], Path(d["exit_file"])
    if stop:
        if pid_alive(d["pid"]):
            os.killpg(int(d["pid"]), 15)
            print(f"Stopped the download of {d['repo']}. Rerun setup to resume it; what's on disk is kept.")
        else:
            print("The download isn't running.")
        return 0
    def on_disk():
        # The model's own folder, so a download resumed after a stop counts what was already there.
        # (Records written by an older setup.py have no "dir": fall back to the growth of the whole cache.)
        return model_bytes(d["dir"]) if d.get("dir") else max(0, dir_bytes(d["hub"]) - d["hub_bytes_at_start"])

    while True:
        got = on_disk()
        if exit_file.exists():
            code = exit_file.read_text().strip()
            if code == "0":
                print(f"{d['repo']}: finished ({total / 1e9:.0f} GB, {took(exit_file.stat().st_mtime - d['started'])}). "
                      "Inference commands will start without downloading.")
                return 0
            tail = "\n".join(Path(d["log"]).read_text(errors="replace").replace("\r", "\n").splitlines()[-5:])
            print(f"{d['repo']}: the download stopped with an error (exit code {code}). Last lines of {d['log']}:\n{tail}")
            return 1
        if not pid_alive(d["pid"]):
            print(f"{d['repo']}: the download is no longer running and didn't finish ({got / 1e9:.0f} of {total / 1e9:.0f} GB). "
                  "Rerun setup to resume it.")
            return 1
        time.sleep(5)
        now = on_disk()
        # Counted from what has reached the disk. The downloader holds data in memory and writes it in bursts, so
        # on a slow link a 5 s reading swings between 0 and several times the real speed. After the first half
        # minute, use the average since this run of the download started.
        elapsed = time.time() - d["started"]
        rate = (now - d.get("dir_bytes_at_start", 0)) / elapsed if d.get("dir") and elapsed > 30 else (now - got) / 5
        left = ("finishing up" if now >= total else f"about {took((total - now) / rate)} left" if rate > 0
                else "nothing on disk yet (on a slow link it buffers in memory first)")
        print(f"{d['repo']}: {now / 1e9:.0f} of {total / 1e9:.0f} GB ({min(100, 100 * now / total):.0f}%), "
              f"{rate / 1e6:.0f} MB/s, {left}", flush=True)
        if not watch:
            return 0
        time.sleep(5)


def took(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.1f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def main():
    global UV
    ap = argparse.ArgumentParser(description="Set up a GPU box for the TWC demo (see the top of this file)")
    ap.add_argument("--node", default=os.environ.get("TWC_NODE", "https://twc-verification-node.fly.dev"),
                    help="verification node URL")
    ap.add_argument("--host", help="name this machine shows up as (default: its hostname)")
    ap.add_argument("--split", help="report this box as several nodes, e.g. 4+4, or 'ask'")
    ap.add_argument("--vllm-spec", default=VLLM_SPEC, help=f"what to install (default: {VLLM_SPEC})")
    ap.add_argument("--force-vllm", action="store_true",
                    help="install the pinned vLLM in its own venv even if this box already has that version")
    ap.add_argument("--sequential", action="store_true",
                    help="install one thing at a time and print everything, instead of in parallel with log files")
    ap.add_argument("--skip-collector", action="store_true")
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--skip-hf-login", action="store_true", help="don't offer the optional Hugging Face login")
    ap.add_argument("--skip-torchtitan", action="store_true")
    ap.add_argument("--nanochat", action="store_true", help="also install nanochat, the earlier training workload")
    ap.add_argument("--skip-nanochat", action="store_true", help=argparse.SUPPRESS)  # old flag; nanochat is now opt-in
    ap.add_argument("--no-model-download", action="store_true",
                    help="don't download the serving model in the background (252 GB on H100, 511 GB otherwise)")
    ap.add_argument("--model", help="download this Hugging Face model instead of the box's default serving model")
    ap.add_argument("--progress", action="store_true", help="show how far the background model download has got, then exit")
    ap.add_argument("--watch", action="store_true", help="with --progress: keep updating until it finishes")
    ap.add_argument("--stop", action="store_true", help="with --progress: stop the background download")
    a = ap.parse_args()
    if a.progress:
        sys.exit(progress(watch=a.watch, stop=a.stop))

    global _setup_log
    LOGS.mkdir(parents=True, exist_ok=True)
    _setup_log = open(SETUP_LOG, "ab")  # appended to, so a rerun keeps the earlier run's record
    _setup_log.write(f"\n===== setup.py started {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}: "
                     f"{' '.join(sys.argv[1:]) or '(no options)'} =====\n".encode())
    names, families = detect_gpus()
    preflight(names)
    try:
        UV = ensure_uv()
    except StepFailed as e:
        die(str(e))
    dropped = [v for v in UV_ENV_IGNORED if os.environ.pop(v, None) is not None]
    if dropped:
        log(f"not applying this box's own package-install settings ({', '.join(dropped)}) to our environments")

    # The two steps that ask something come first, so the rest can run unattended.
    asking = []
    if not a.skip_collector:
        asking.append(("collector", lambda: setup_collector(a)))
    if not a.skip_hf_login:
        asking.append(("Hugging Face login", lambda: setup_hf_login(a)))
    # The big downloads. `out` is each one's log file, or None to print to the terminal.
    installs = []
    if not a.skip_vllm:
        installs.append(("vLLM", lambda out: setup_vllm(a, families, out)))
    if not a.skip_torchtitan:
        installs.append(("TorchTitan", lambda out: setup_torchtitan(families, out)))
    if a.nanochat:
        installs.append(("nanochat", lambda out: setup_nanochat(families, out)))

    result = {}  # name -> (outcome, seconds)

    def attempt(name, step):
        start = time.time()
        try:
            outcome = step() or "ok"
        except StepFailed as e:
            log(f"{name} FAILED: {e}")
            outcome = "FAILED"
        except Exception as e:  # a bug in this script shouldn't take the other steps down with it
            log(f"{name} FAILED unexpectedly: {e!r}")
            outcome = "FAILED"
        result[name] = (outcome, time.time() - start)

    for name, step in asking:
        attempt(name, step)

    if a.sequential or len(installs) < 2:
        for name, step in installs:
            log(f"--- {name} ---")
            attempt(name, lambda step=step: step(None))
    elif installs:
        log(f"--- installing {' and '.join(n for n, _ in installs)} at the same time; nothing more to answer ---")
        threads = {}
        for name, step in installs:
            out = LOGS / f"{name.lower()}.log"
            out.write_text("")
            log(f"{name}: started. Follow it with: tail -f {out}")
            threads[name] = threading.Thread(target=attempt, args=(name, lambda step=step, out=out: step(out)))
            threads[name].start()
        start = time.time()
        while any(t.is_alive() for t in threads.values()):
            for t in threads.values():
                t.join(timeout=30 / len(threads))
            running = [n for n, t in threads.items() if t.is_alive()]
            if running:
                log(f"still installing {' and '.join(running)} ({took(time.time() - start)} so far)")

    # Last, once the installs are done: the serving model, by far the biggest download. It starts detached and
    # carries on after this script exits. Skipped when the box isn't being set up for inference.
    downloading = None
    if a.model or not (a.no_model_download or a.skip_vllm):
        kind = "H100" if any("H100" in n.upper() for n in names) else "other"
        repo = a.model or MODEL_REPO[kind]
        try:
            downloading = (repo, start_model_download(repo, MODEL_GB[kind]))
        except Exception as e:  # never let the optional download stop the installs
            downloading = (repo, f"not started: {e!r}")
        log(f"model download ({repo}): {downloading[1]}")
        # `twc-progress` is this script's --progress mode, kept where the Walkthrough can call it.
        if Path(__file__).resolve() != (HOME / "setup.py").resolve():
            shutil.copy(__file__, HOME / "setup.py")
        PROGRESS_CMD.parent.mkdir(parents=True, exist_ok=True)
        PROGRESS_CMD.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{HOME / "setup.py"}" --progress "$@"\n')
        PROGRESS_CMD.chmod(0o755)

    log("--- done ---")
    for name, _ in asking + installs:
        outcome, seconds = result[name]
        log(f"{name}: {outcome} ({took(seconds)})")
    log(f"Setup has finished and this terminal is free. A copy of everything above is in {SETUP_LOG}")
    if downloading and downloading[1].startswith(("started", "already")):
        log(f"Still downloading in the background: {downloading[0]}, the model for the inference workloads.")
        log(f"  Check on it:  {PROGRESS_CMD}          (add --watch to keep it updating)")
        log(f"  Stop it:      {PROGRESS_CMD} --stop")
        log("  It keeps going if you close this terminal. Training doesn't need it: you can start that now.")
    elif downloading:
        log(f"The serving model was NOT downloaded ({downloading[1]}). Inference commands will try to download it themselves.")
    sys.exit(1 if any(o == "FAILED" for o, _ in result.values()) else 0)


if __name__ == "__main__":
    main()
