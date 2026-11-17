# TWC verification node

A proof of concept for **telemetry-based compute verification**. It checks GPU telemetry alone (power, tensor-core utilisation, network traffic) to see whether what a cluster runs matches what its operator declared.

**Live demo:** https://twc-verification-node.fly.dev. The site is password-protected, so ask the author for access. Read the Overview first, then open the Replay demo.

## How it works

```
GPU boxes ── collector.py ──► telemetry ──┐
                                          ├──► verification node ──► verdict per window
Slurm head ── slurm_reporter.py ─► jobs ──┘     (checks 3 rules)
```

- **Collectors** run on each GPU box. They read NVML and stream per-GPU telemetry to the node.
- **The scheduler reporter** sends the cluster's job table. Each job declares its type in Slurm's comment field (`sbatch --comment=declared:inference`).
- **The verification node** runs every 10 s. It checks the last 60 s of telemetry against the declared jobs and shows where a rule broke.

In the intended design, the node sits on-site in a tamper-evident enclosure. It receives telemetry signed on the GPU and reports only signed verdicts to an outside verifier. This prototype implements the data path and the rules, but not the signing (see Limits).

## The rules

| Rule | A job or GPU breaks it when… |
|---|---|
| 1. No undeclared load | A GPU that no declared job covers averages over 200 W (300 W on B200/B300) |
| 2. No hidden training | A job declared `inference` has tensor-core utilisation over 40% |
| 3. No cross-partition traffic | A job declared `inference` sends over 0.001 GB/s to other boxes over the RDMA ports |

The full spec is [`node/rules/rules.md`](node/rules/rules.md). It is detailed enough to reimplement the rules from. [`test_spec.py`](node/rules/test_spec.py) is such a reimplementation, and it is checked against the real code.

## What it was tested on

Recordings from 8×H200 nodes on two rentals: two nodes on 2026-11-11 and one node on 2026-11-16.

- **Inference:** vLLM serving DeepSeek V4.1 Flash, driven by AgentX agentic traces. It covers tensor-parallel, expert-parallel and a prefill/decode split across the two nodes.
- **Pretraining:** TorchTitan on Llama 3 8B, with different batch sizes and activation-checkpointing settings.

The thresholds separated every recorded workload, with wide margins. None of these workloads were trying to evade the rules.

## Limits

- **Telemetry is unsigned.** The design assumes the GPU signs telemetry on-die, which current firmware doesn't support. Until then, the operator could spoof the telemetry.
- **The rules are not red-teamed.** The thresholds are placeholders fit to a handful of runs. Low-power training (an FSDP run at about 187 W) passes under rule 1's floor.
- **Multi-node is partly simulated.** Most composed scenarios line up recordings made at different times. `collector.py --split` can divide one box into pretend nodes (see [SPLIT_CAVEATS.md](SPLIT_CAVEATS.md)).
- **The live collector doesn't send network counters yet,** so rule 3 only judges recordings.
- **Node state lives in memory** on a single machine.

## Repository layout

| Path | What |
|---|---|
| `node/app.py` | The verification node (FastAPI): ingest, rule loop, APIs, pages |
| `node/rules/` | One file per rule, plus the spec and its test |
| `node/pages/`, `node/static/` | The Overview, Replay, Walkthrough and Debug pages; replay data |
| `collector.py` | Telemetry collector for GPU boxes (dependency: `nvidia-ml-py`) |
| `slurm_reporter.py` | Sends the Slurm job table to the node |
| `setup.py` | Sets up a rented GPU box: collector, vLLM, TorchTitan |
| `make_replay_data.py` | Turns raw recordings into the Replay page's data and runs the rules over them |
| `prover.py`, `ivo.py`, `twc_common.py` | An earlier prototype of the signed data path (Ed25519, nonces, hash chain) |

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-node.txt
SITE_PASSWORD=pick-one .venv/bin/uvicorn node.app:app --port 8000
.venv/bin/python node/rules/test_spec.py
```

To stream from a real GPU box, open **Debug → Connect a box** on a running node. It shows a one-line install command with the node's URL filled in.

Open work is in [TODOS.md](TODOS.md).
