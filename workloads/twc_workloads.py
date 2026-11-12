#!/usr/bin/env python3
"""twc_workloads.py: a menu of GPU workloads for sanity-checking the TWC live telemetry page.

Runs ON a GPU box (needs PyTorch with CUDA). Plays a schedule of timed phases, one after another,
and logs each phase's start/end to phases.jsonl next to this file (ground truth for later).

    python3 twc_workloads.py "idle:60 serve-decode:180 train:180 membw:120"

Phases (NAME:SECONDS, optional ,key=value tweaks e.g. "duty:120,on=5,off=5"):
    idle           nothing (the baseline)
    serve-decode   load the local vLLM server: short prompts, long answers (decode-heavy), 8 at a time
    serve-prefill  long random prompts (~6k tokens), tiny answers (prefill-heavy), 8 at a time
    serve-light    one request at a time, medium prompts and answers
    matmul         dense bf16 8192x8192 matmuls on every GPU, flat out (compute-bound, "training-like")
    train          fake data-parallel training step on every GPU: 6 matmuls, then an NCCL
                   all-reduce of 256 MB "gradients" (GPUs in lockstep; NVLink traffic on multi-GPU boxes)
    allreduce      only the all-reduce, back to back (NVLink-bound; needs >1 GPU)
    membw          copy 1 GiB buffers back and forth (HBM-bandwidth-bound, light on compute)
    duty           matmul bursts: `on` seconds busy, `off` seconds idle (defaults 5/5)

The serve-* phases talk to vLLM at http://127.0.0.1:18000/v1 (vast's vLLM image); --vllm to change.
The GPU phases need only ~1-2 GB per GPU, so they run fine next to an idle vLLM that holds the rest.
"""
import argparse
import json
import os
import random
import string
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "phases.jsonl"
GPU_MODES = {"matmul", "train", "allreduce", "membw", "duty"}
SERVE = {  # concurrency, prompt words, max output tokens
    "serve-decode": (8, 40, 600),
    "serve-prefill": (8, 4500, 8),
    "serve-light": (1, 400, 300),
}


def log(msg):
    print(f"[workloads {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- GPU phases (one process per GPU)
def gpu_worker(rank, world, mode, secs, opts, port):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist_on = world > 1 and mode in ("train", "allreduce")
    if dist_on:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    n = 8192
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    grads = torch.randn(64 * 1024 * 1024, device=dev) if mode in ("train", "allreduce") else None   # 256 MB fp32
    if mode == "membw":
        src = torch.empty(512 * 1024 * 1024, device=dev, dtype=torch.float16)                      # 1 GiB
        dst = torch.empty_like(src)
    stop = torch.zeros(1, device=dev)
    on, off = float(opts.get("on", 5)), float(opts.get("off", 5))
    t_end = time.time() + secs
    step = 0
    while True:
        if mode == "matmul":
            for _ in range(20):
                a @ b
        elif mode == "train":
            for _ in range(6):
                a @ b
            if dist_on:
                dist.all_reduce(grads)
        elif mode == "allreduce":
            if dist_on:
                dist.all_reduce(grads)
            else:
                time.sleep(0.05)
        elif mode == "membw":
            for _ in range(10):
                dst.copy_(src)
                src.copy_(dst)
        elif mode == "duty":
            phase_t = (secs - (t_end - time.time())) % (on + off)
            if phase_t < on:
                for _ in range(20):
                    a @ b
            else:
                torch.cuda.synchronize(dev)
                time.sleep(min(0.1, on + off - phase_t))
        step += 1
        torch.cuda.synchronize(dev)
        # everyone stops on the same step, or a collective would hang: rank 0 decides, all-reduce tells the rest
        stop.fill_(1.0 if time.time() >= t_end else 0.0)
        if dist_on:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item() > 0:
            break
    if dist_on:
        dist.barrier()
        dist.destroy_process_group()


def run_gpu_phase(mode, secs, opts):
    import torch
    import torch.multiprocessing as mp
    world = torch.cuda.device_count()
    if world == 0:
        raise SystemExit("no CUDA GPUs visible")
    if mode == "allreduce" and world < 2:
        log("allreduce needs 2+ GPUs; idling instead")
    port = random.randint(29600, 29999)
    mp.start_processes(gpu_worker, args=(world, mode, secs, opts, port), nprocs=world, join=True, start_method="spawn")


# ---------------------------------------------------------------- serving phases (HTTP load on vLLM)
def words(n):
    # random letters, so nothing hits vLLM's prefix cache
    return " ".join("".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))) for _ in range(n))


def run_serve_phase(name, secs, opts, base):
    conc, n_words, max_tokens = SERVE[name]
    conc = int(opts.get("conc", conc))
    model = json.loads(urllib.request.urlopen(base + "/models", timeout=10).read())["data"][0]["id"]
    t_end = time.time() + secs
    done, toks, errs = [0], [0], [0]
    lock = threading.Lock()

    def client():
        while time.time() < t_end:
            body = {"model": model, "prompt": words(n_words) + "\n\nSummarise the text above.",
                    "max_tokens": max_tokens, "ignore_eos": True, "temperature": 0.8}
            try:
                r = json.loads(urllib.request.urlopen(urllib.request.Request(
                    base + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300).read())
                with lock:
                    done[0] += 1
                    toks[0] += r.get("usage", {}).get("completion_tokens", 0)
            except Exception as e:
                with lock:
                    errs[0] += 1
                    if errs[0] <= 3:
                        log(f"{name}: request failed: {e}")
                time.sleep(1)

    threads = [threading.Thread(target=client, daemon=True) for _ in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=max(0, t_end - time.time()) + 300)
    return {"model": model, "conc": conc, "requests": done[0], "out_tokens": toks[0], "errors": errs[0]}


# ---------------------------------------------------------------- schedule
def parse(spec):
    phases = []
    for item in spec.split():
        head, *kv = item.split(",")
        name, secs = head.split(":")
        if name not in GPU_MODES | set(SERVE) | {"idle"}:
            raise SystemExit(f"unknown phase {name!r}")
        phases.append((name, float(secs), dict(x.split("=", 1) for x in kv)))
    return phases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedule", help='e.g. "idle:60 serve-decode:180 train:180"')
    ap.add_argument("--vllm", default="http://127.0.0.1:18000/v1")
    ap.add_argument("--loop", action="store_true", help="repeat the schedule forever")
    a = ap.parse_args()
    phases = parse(a.schedule)
    log(f"schedule: {' '.join(f'{n}:{s:.0f}' for n, s, _ in phases)} ({sum(s for _, s, _ in phases) / 60:.1f} min)"
        + (", looping" if a.loop else ""))
    while True:
        for name, secs, opts in phases:
            t0 = time.time()
            log(f"start {name} for {secs:.0f} s {opts or ''}")
            info = {}
            try:
                if name == "idle":
                    time.sleep(secs)
                elif name in GPU_MODES:
                    run_gpu_phase(name, secs, opts)
                else:
                    info = run_serve_phase(name, secs, opts, a.vllm)
            except Exception as e:
                info = {"error": repr(e)}
                log(f"{name} failed: {e!r}")
            rec = {"phase": name, "start": round(t0, 3), "end": round(time.time(), 3), "planned_s": secs,
                   "opts": opts, "host": os.uname().nodename, **info}
            with open(LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"done {name} ({rec['end'] - rec['start']:.0f} s) {info or ''}")
        if not a.loop:
            break


if __name__ == "__main__":
    main()
