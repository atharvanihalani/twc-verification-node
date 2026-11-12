# Demo classifier v0

A prover streams signed GPU power telemetry to a verification node. The node checks
signatures, nonces and a hash chain, runs four conformance rules against the prover's
declarations, and publishes one signed verdict per window to a simulated IVO view.

Live: https://twc-verification-node.fly.dev (`/` Overview, `/replay`, `/walkthrough`, `/debug`; the old `/node` and `/ivo` pages are parked in `parked-pages/`). Password only, no username
(`SITE_PASSWORD` in `.env`). Browsers get a login page that sets a 30-day cookie; the CLIs send
it as HTTP basic auth and the node ignores the username.

Design and demo script: [DEMO_PLAN.md](DEMO_PLAN.md).

## Layout

| Path | What |
|---|---|
| `node/app.py` | Verification node (FastAPI): websocket ingest, nonce pump, checks, password login, APIs |
| `node/rules/` | The three rules, one file each (`undeclared_load.py`, `training_shaped.py`, `boundary_traffic.py`), a small orchestrator (`__init__.py`), the plain-language spec (`rules.md`) and a check that the two agree (`test_spec.py`) |
| `node/static/` | `node.html` (raw traces, rules, log) and `ivo.html` (verdicts only, signatures checked in the browser) |
| `prover.py` | Prover CLI: replays recorded traces, signs windows, can record or replay (attack) |
| `ivo.py` | IVO CLI: fetches verdicts and verifies every node signature against a pinned key |
| `twc_common.py` | Canonical JSON, hashing, Ed25519 |
| `make_replay.py` | Turns a slice of a recorded serving run into a replay file, already in the collector's wire format |
| `make_replay_data.py` | Turns the 2026-11-11 recordings (`../telemetry-logs/serving-traces-vllm/runs/`) into `node/static/replay-data/` for the Replay demo page, and runs the real rules over them. `--table` prints what the rules say about each recording |
| `replay_feed.py` | Plays a replay file into `/api/collect` as if it were `collector.py` (stdlib only, loops) |
| `replays/` | Replay files, baked into the Docker image. `c1-tp8-c64.jsonl.gz` (314 KB) = c1 TP8, conc 64, profiling +5..+15 min |
| `.env` | Secrets: site password, node key, pinned prover pubkey (mirrored to `fly secrets`). Not for sharing. |
| `keys/prover.key` | Prover's Ed25519 seed (the "CPU key"; moment 4 = someone else has this file) |

## Commands

```bash
.venv/bin/python prover.py traces                 # list replay presets
.venv/bin/uvicorn node.app:app --port 8000        # local node (reads .env, so auth is on)
.venv/bin/python prover.py run --trace serve-tp4x2 --declare a:inference:0-3 --declare b:inference:4-7
.venv/bin/python ivo.py --pin "$(grep NODE_PUBKEY .env | cut -d= -f2)"
```

Add `--node https://twc-verification-node.fly.dev` to any `prover.py` or `ivo.py` command (before
or after the subcommand) to target the deployed node. `--speed 4` replays 4x faster; `prover.py reset` clears the node.

Deploy: `fly deploy --ha=false --remote-only`. Use `--ha=false` because state lives in
memory, so the app must run exactly one machine.

## Live collector (verified 2026-11-04 on a vast H200 NVL, live on Fly)

`collector.py` runs **on the GPU box** (NVML only sees local GPUs) and pushes raw samples to
one or more nodes. Its only dependency is `nvidia-ml-py`.
- **Rates:** power at 10 Hz. Energy, SM clock, throttle reasons, GPU/HBM temperature, memory
  used and GPM utilisation at 1 Hz.
- **Transport:** every second it POSTs JSON to `/api/collect`. There are no signatures, nonces,
  identity or declarations yet, by design.
- **Resilience:** a failed send is dropped and retried next second. A failed GPU read closes
  NVML and reopens it with backoff (1 s, 2 s, 4 s … 60 s).

**Install on any box** (the node serves the script; the Debug → Connect a box page shows this
line with the right URL filled in):

```bash
read -rsp "TWC password: " P; echo; curl -fsSu ":$P" https://twc-verification-node.fly.dev/collector.py -o /tmp/collector.py && TWC_PASSWORD="$P" python3 /tmp/collector.py install --node https://twc-verification-node.fly.dev
```

What `install` does:
- Copies itself to `~/.twc-collector/` and saves settings there (`config.json`, mode 600).
- Installs `nvidia-ml-py` in a private venv if it's missing.
- Keeps the collector running with the first of these that the machine has:
  1. systemd (needs root);
  2. supervisor (vast images);
  3. a background restart loop plus a cron `@reboot` entry.
- `collector.py status` and `collector.py uninstall` do what they say.
- `--host NAME` sets the display name.
- `--node a,b` sends to several nodes, for example Fly plus a local dev node.

**Local dev node:** run `.venv/bin/uvicorn node.app:app --port 8000 --reload --reload-dir node`
on your Mac. Then run `./tunnel.sh <ssh-port> <user@box>` so the box's `127.0.0.1:18999` reaches
it. The tunnel reconnects by itself and clears a stale port on the box first. The vast SSH
connection still stalls every so often; Fly doesn't need the tunnel at all.

Verified:
- Recovery from a fake "GPU is lost" error.
- supervisor and loop modes, each surviving a killed collector.
- The one-liner on a cleared box.
- Fly and the local node receiving simultaneously.

systemd mode is written but untested: there's no systemd box yet.

## Demo moments (verified 2026-11-04, local and on Fly)

| # | Command (all add `--speed 4`) | Result |
|---|---|---|
| 1 | `run --trace serve-tp4x2 --declare a:inference:0-3 --declare b:inference:4-7 --record rec.jsonl` | compliant |
| 2 | `run --attack replay --replay-file rec.jsonl` | every window rejected: hash chain broken + nonce mismatch |
| 3 | `run --trace train-ddp --declare a:inference:0-3 --declare b:inference:4-7` | flagged: cross_partition_sync, training_shaped |
| 4 | Same as 1, but narrated as "actually training, forging with the stolen key" | compliant, wrongly (the point) |

Also checked:

- Undeclared GPUs flag `undeclared_load`.
- An unpinned prover key is refused at hello.
- A counter reporting half the real energy flags `energy_mismatch`.
- No password: pages redirect to `/login`, `/api/*` returns 401, the websocket is refused (403).

## Rule performance on the recorded traces

These are offline sliding-window results: 60 s windows every 5 s over the whole trace. Each
trace comes from one box (n = 1).

| Trace | Declared as | Flagged | Which rule |
|---|---|---|---|
| serve-tp4x2 (H200, 2x TP4 vLLM, honest) | two inference jobs | 23% | training_shaped (false positive) |
| serve-tp8 (H200, TP8 vLLM, honest) | one inference job | 34% | training_shaped (false positive) |
| infer-tp8 (H100, TP8 vLLM light load, honest) | one inference job | 0% | |
| idle (honest) | idle | 0% | |
| train-ddp (H100 DDP) | two inference jobs | 100% | cross_partition_sync + training_shaped |
| train-ddp | one inference job | 100% | training_shaped |
| train-fsdp (H100 FSDP, ~187 W) | one or two inference jobs | **0% (missed)** | under the 200 W busy gate |

What this says:

- **`cross_partition_sync` is the strong rule.** Independent TP4 replicas correlate at about r = 0.0
  (90th percentile 0.05), while DDP training correlates at r ≥ 0.72. Separation is clean.
- **`training_shaped` can't tell tensor-parallel serving from DDP training on power alone.**
  Both keep GPUs in lockstep at 600–700 W. This matches the earlier I3 false positive in
  gpu-telemetry-exploration.
- **Low-power training (FSDP here) slips under the busy gate entirely.**
- **Correlations use first differences, not levels.** Two independent replicas behind one load
  generator ramp up together, so their power levels correlate at about r = 0.8 during the ramp.
  Differencing removes that shared trend and fixed the false flags at the start of a load phase.

## Known limits of v0

- The prover key is a file on the prover's machine, which gives neither non-extractable keys
  nor sensor-direct reads (moment 4).
- Telemetry is replayed, not live NVML. A `--trace nvml` source for the real-box step is not
  written yet.
- Node state is in memory. A Fly restart forgets declarations, buffers and verdicts. The node
  key and pinned prover key survive, because they are Fly secrets.
- Declarations are signed by the prover but not cross-checked against Slurm.

## Replay feed (built 2026-11-05, local only so far)

A recorded run played into the live path, so the Debug pages have an 8-GPU feed without renting a box.
- **Data:** c1 (one vLLM TP8 server, 8× H200), concurrency 64. It uses the 10 minutes from 5 to 15 min
  into the AgentX profiling phase: that's the steady part (~45–52 running requests, ~590–630 W per GPU).
  The first ~5 min are a ramp with only ~10 sessions active (see RUNLOG.md).
- **Path:** `replay_feed.py` stamps every row with the current time and POSTs 1 s batches to
  `/api/collect`, in the same body shape as the collector. v0 has no signatures or nonces, so the
  node can't tell it from a real box; the UI tags it "recorded" instead.
- **Controls:** Debug → Live telemetry or Raw data, when nothing is sending: "Play a recorded trace".
  The node then starts `replay_feed.py` as a child process (`POST /api/replay/start`). "Stop" on the
  replay's header (`POST /api/replay/stop`) kills it and drops its data, so the page goes back to empty.
  This applies to everyone viewing the site. The child exits if the node process goes away (e.g. uvicorn --reload).
- **Looping:** at the end of the file it starts over, and energy keeps counting up across loops.
- **Rebuild the file:** `.venv/bin/python make_replay.py [--start S --len L]`, with S in seconds after the profiling start.
- **From the Mac to any node:** `TWC_PASSWORD=… python3 replay_feed.py --node https://twc-verification-node.fly.dev`

Also on 2026-11-05 (local): when nothing is sending, Live telemetry and Raw data show their normal
layout with empty axes or cells, labelled "receiving no data right now", instead of a blank page. Raw
data gets a box picker when more than one box is sending.

## Multiple boxes + hello (built 2026-11-05, local only so far)

Up to 8 boxes can report to one node at once (was 4 until 2026-11-06), e.g. two unrelated vast boxes each running collector.py.
- **Live telemetry:** one shared set of graphs. Each box gets a slot (= its colour): a thick line is the
  mean across that box's GPUs, faint thin lines are its individual GPUs. One row per box above the
  graphs (name, "8× H200"-style label, status); click a row to hide or show that box (kept in localStorage).
  Each box is drawn against its own newest sample, so clock skew between boxes doesn't shift the lines;
  a silent box's lines end where its data stopped.
- **Cap:** a 5th box gets HTTP 429 with the reason, unless a box has been silent for over 60 s, in which case
  the newcomer takes that slot. A box silent for 5 min drops off by itself.
- **Hello:** before its first batch, collector.py (v0.2) POSTs `/api/hello` with what it is: hostname, OS,
  CPU, RAM, driver, CUDA, NVML version, GPM on/off, sample rates, and per GPU its name, UUID, serial,
  PCI bus id, memory, power limit and VBIOS. The node records the sender's IP too. If the node has
  forgotten the box (restart), `/api/collect` answers `need_hello` and the box introduces itself again.
  An older collector without hello still works; its label comes from the GPU names in its batches.
- **Raw data:** picker shows labels; "What this box said about itself (hello)"; a "Connection log" of
  hellos (with everything sent), refusals, evictions and expiries (`GET /api/live/events`).
- **Fake boxes for testing:** `replay_feed.py --host NAME --gpus N --gpu-name "NVIDIA H100 NVL" --offset S`.
  The collector itself can be tested on a Mac with a fake `pynvml` module on PYTHONPATH.

## Split one box into pretend nodes (built 2026-11-06, local only; NOT tested on a real GPU box yet)

For prototyping multi-node scenarios without a multi-node cluster. `collector.py` v0.3:
`--split 2+2+4` (or `install --split ask`, which prompts and validates: up to 4 parts, sum <= GPU count).
- **One process, one sampler**; one sender per part. Parts show up as `<host>-a`, `-b`, ... with their
  GPUs renumbered from 0 (physical GPU 4 = `<host>-b` GPU 0 in a 4+4 split).
- **Honest hello:** each part's hello keeps the real `hostname`, adds `split` (`physical_host`, `part`,
  `of`, `physical_gpus`) and a `physical_index` per GPU. The label reads "4× H200 · part b of 4+4 on one box".
  Rules must not read the split fields.
- **Off again:** `install` without `--split` writes `split: null` to the config.
- **Double-count guard (node):** if two live hosts report the same GPU UUID, both rows show it in red
  and the connection log gets a `conflict` event (`/api/live` field `double_counted`).
- **Connect a box** has a second one-liner (the normal one plus `--split ask`).
- What this fakes vs real multi-node: [SPLIT_CAVEATS.md](SPLIT_CAVEATS.md).
- Tested on the Mac with a fake `pynvml` (8 GPUs, GPU i = 100+10i W): renumbering, honest hello, the guard,
  the prompt (rejects 5 parts, sums over 8, non-numbers; warns about unreported GPUs).

## Replay demo page (built 2026-11-06, local only)

### Real recordings on the page: choices made when wiring them (2026-11-11)

What the page shows is not always exactly what was recorded. These are the places where a choice was made.

- **Ten a second in the last-minute strips, once a second elsewhere.** The recordings hold power at 20 Hz (the value changes about 10 times a second) and GPU counters at 10 Hz. The last-minute strips play the counters as polled and power as the mean of 2 polls. Whole clip, the lanes and the network row use the mean of each second. The rules are run on the 20 Hz power and the once-a-second counters, as the live node would.
- **Serving clips start at the measured load, not at the mark the recording script wrote.** After each concurrency mark, AgentX spends about 4 minutes starting up with no requests, then 2 to 7 minutes filling the cache (long prompts in, one token out, about 640 W flat), then 15 minutes of measured load. In step 1 a serving pick plays that concurrency's own cache warm-up and measured load, start to end (17 to 22 minutes), and nothing else; the step-2 scenarios and "Both, side by side" play 10 minutes of the measured load. The converter reads the three stretches from the AgentX log (`warm_N`, `load_N`, `load_end_N` in `index.json`).
- **Training plays the whole 14-minute recording**: 2 minutes idle, cold start, steps with checkpoint saves, stop, 1 minute idle. The baseline is the repeat run (p1b), because the first run (p1) shared its node with the model download.
- **Concurrency 16 and 64 (tensor-parallel) are one recording:** one server start, the load stepped inside the run. Each pick shows only its own warm-up and load, so the two look different (his ask, 2026-11-11).
- **"Whole clip" in the Telemetry pane** shows the pick's whole clip at once, as about 400 points a line (each the mean of its stretch). The 12 to 13 minutes of server loading are in the data but on no view.
- **Composed views.** "Both, side by side" and the step-2 scenarios "Honest cluster", "Training declared as inference" and "Undeclared load" put a serving recording next to a training recording. Each of the three uses a different pair, so they do not look alike (his ask, 2026-11-11): tensor-parallel at 16 sessions with the baseline training run; expert-parallel at 16 with middle activation checkpointing; tensor-parallel at 64 with batch 1. The line under the scenario pills says which. The two were made at different times on the two nodes of one rental and are lined up to start together. Only the prefill/decode split was recorded on both nodes at once.
- **Declarations are added afterwards.** No scheduler ran on the box. Each scenario's jobs are written in `make_replay_data.py` and declared from the first second of the clip.
- **Two scenarios changed shape from the mock-up.** "Undeclared load" is all 8 GPUs of a node (the training job used all 8), not 4. "Training declared as inference" trains from the start; the mock-up had it serve first and switch.
- **"One job split in two".** With PCIe sent per GPU alone the rule missed it: the real split peaks at 0.24 GB/s against the 0.35 limit. Since Fly v33 rule 3 also reads what a node sends to other nodes over its high-speed network ports, from the side logger's port counters, and catches the split in every check. The live collector does not report that counter yet, so a connected box is still judged on PCIe alone. The verdict lines under each scenario are computed from the rule results, so they change if the rules do.
- **Rule 2 is tensor cores only** since 2026-11-11. The power-spread test was removed after it fired on honest serving in these recordings.
- **Units.** NVML's PCIe and NVLink rates are divided by 1000 and called GB/s, as `collector.py` does.
- **Left out of the page:** PCIe received, the server's own metrics, and most of the side logs (per-link NVLink, disk, which process holds which GPU). Network sent between nodes is in the Readings menu, off by default. They are in the run directories.
- **Still made up:** "Add your GPUs" and the your-box view it opens.
- **One box, one night.** Every number comes from one rental of two 8 × H200 nodes on 2026-11-11.

**Since 2026-11-11 `/replay` is `node/pages/replay.html` playing real recordings from `node/static/replay-data/` (see `make_replay_data.py` and TODOS.md). What follows describes the earlier live page, now parked at `parked-pages/replay-live.html`.**

`/replay` (nav: Replay demo, key 2) replaces the old signed-prover page, which is parked at `parked-pages/node.html` (not online; `/node` redirects to `/replay`).
- **Timeline:** declared jobs from Slurm drawn on GPU rows grouped by node, hatched blue for inference and green for pretraining.
  Red cross-hatch marks a rule broken on those GPUs over that time. Each `boundary_traffic` hit gets a red gutter lane with dots on its member jobs.
  Everything after "now" is veiled. **Fixed now** pins "now" at ~89% of the width so the axis compresses; **Expand** spans all declared jobs.
- **Rules:** one row per rule (`undeclared_load`, `training_shaped`, `boundary_traffic`), each with live numbers and a holding / broken pill. `training_shaped` lists its two flags (tensor cores, power spread) beneath it.
  The dropdown "Slurm table as reported" shows the `sacct` text as of now.
- **Keys:** `e` view · `space` pause · `←/→` ±1 min · `r` restart. Plays at ×12 and loops (36 min → 3 min).
- **Data:** `make_scenario.py` → `replays/scenario-1.json`, served at `/api/scenario` and baked into the image. It stitches recorded H100 traces
  (I3 vLLM TP8 light load = inference pod on node-a; T1 DDP = pretraining on node-b) into 36 min with 3 staged lies:
  pretraining spills onto node-a 13:52–13:58, and node-b keeps training after job 102 ends (14:01–14:04).
  It then runs the real `node/rules.py` every 10 s over the trailing 60 s. The page only plays the result back, so the Fly trial's 5-min machine stop can't reset it.
- **Findings while building:**
  - 6 W of added sensor noise per GPU drops DDP lockstep from ~0.6 to ~0.3, below the 0.5 threshold. The scenario uses 1.5 W.
  - The honest TP8 inference pod has lockstep ~0.5–0.6 too; only the 400 W gate keeps it from flagging.
  - Declarations have to be matched against the whole trailing window. Using only the jobs alive at the eval time made `undeclared_load` fire for the minute after a job ended.

## Slurm declarations (2026-11-06)

Jobs declare their type in Slurm's comment field: `sbatch --comment=declared:pretraining job.sh` (or `inference`).

- `slurm_reporter.py` runs once per cluster, wherever `sbatch` works. Every 5 s it reads the whole job table (`scontrol show job -d -o`, read-only) and POSTs it to `/api/declare`. It sends even with no jobs: that's the heartbeat. Install with the "Connect a scheduler" one-liner on Debug → Connect a box.
- The node records each job's first-seen time (commitment timing) and every version of its comment. Users CAN rewrite a running job's comment with `scontrol update`, verified on RunPod 2026-11-06.
- `/api/slurm` joins each job's GPUs to streaming GPUs: Slurm (node, gres index = N in /dev/nvidiaN) → collector hello (hostname, `minor`, collector v0.4+). Shown on Debug → Slurm jobs as "k/n streaming".
- `slurm/setup_one_box.sh`: one-box Slurm for a GPU container (apt slurm-wlm, controller and node on the same box, no cgroups, no slurmdbd so no `sacct`). On this setup the GPU allocation is NOT enforced: `nvidia-smi` inside a job sees all GPUs.
- `slurm_reporter.py install` v0.2: when there's no `scontrol`, it asks whether to set up a one-box Slurm (same config as `slurm/setup_one_box.sh`, embedded in the script). `--install-slurm` skips the question. Needs root + apt. Prototyping only.
- The Replay demo page (`/replay`) now runs in MODE "live". It draws declared jobs from `/api/slurm` on the streaming GPUs, with split parts labelled by physical GPU numbers. It shows a skeleton when nothing is connected. The "Start at first job" toggle (key `a`) moves the left edge to the earliest running job. The rule rows say "not running": rules aren't evaluated on live data yet. Setting `MODE = "recorded"` brings back the scripted scenario.
- **Rules on live data (2026-11-06).** `rules_loop` in node/app.py runs `rules.evaluate` every 10 s.
  - **Inputs:** the trailing 60 s of 10 Hz power from every GPU that sent data in the last ~15 s, keyed `<host>:<gpu>`. The window ends at the newest timestamp ALL GPUs have reached.
  - **Declarations:** the Slurm jobs whose run overlaps that window. GPUs are matched through `gpu_lookup`. A running job without `declared:<type>` covers nothing, so its load counts as undeclared.
  - **Output:** hits merge into violation intervals (rule + subject, start = first hit − 10 s). `/api/slurm` → `rules` exposes them. The Replay page draws them as red hatch, rule rows and gutter lanes.
  - **Tested on the RunPod 8×H100 pod:**
    - honest pretrain + membw "inference": everything holds, cross r = 0.05;
    - rogue matmul outside Slurm + an undeclared job: undeclared_load on all 8;
    - a declared-pretraining job secretly driving all 8 GPUs (duty 4 s on / 1 s off) beside a sleeping inference job: training_shaped (lockstep 0.53) + cross_partition_sync (r = 0.52), both barely over the 0.5 thresholds.
  - **Known:** intervals trail the load (detection on a 60 s mean); fake `train` steps are invisible at 10 Hz (lockstep 0.06).
