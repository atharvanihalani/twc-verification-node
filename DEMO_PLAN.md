# Demo classifier: script + dummy rules

Written 2026-11-04. Sprint goal (1 week): a blog post with (1) a CLI demo classifier, (2) real telemetry logs from training + inference, (3) reverse-engineering notes on signing uncertainties. This note covers (1).

## Setup

- **Prover:** a remote GPU box (vast.ai), or a replay of recorded traces during development.
- **Verification node:** a cloud server. It gets raw signed telemetry and runs the conformance checks.
- **IVO:** simulated on the same cloud server, as a second UI view that only sees signed verdicts.
- (RATS names, for reference: prover = Attester, verification node = Verifier, IVO = Relying Party.)

Two UI views:

| View | Sees |
|---|---|
| Verification node | Raw traces, declarations, which rule fired |
| IVO | Per-window verdicts only (compliant / flagged + rule name), signed by the node |

The gap between the two views is the confidentiality claim from Draft 3, made visible.

## Demo script: four moments

Each moment shows which layer catches which attack. The last one is deliberately *not* caught.

| # | Moment | What the prover does | What catches it | Shown in the IVO view |
|---|---|---|---|---|
| 1 | Honest run | Declares inference on GPUs 0–7, runs inference | Nothing to catch | All green |
| 2 | Replay | Records a benign inference trace earlier, plays it back while actually training | **Nonces.** The replayed windows can't contain nonces the node only just sent | Flagged: stale / bad nonce |
| 3 | Mislabelled partition | Declares GPUs 0–3 and 4–7 as two separate jobs, actually runs one 8-GPU training job | **Conformance check.** Power on 0–3 and 4–7 moves in lockstep across a declared boundary | Flagged: cross-partition synchrony |
| 4 | Stolen key | Extracts the CPU signing key and signs forged "inference" telemetry with fresh nonces | **Nothing.** Every check passes | Green, wrongly |

Moment 4 motivates the rest of the project: a CPU key gives neither (i) non-extractable keys nor (ii) a signer that reads the sensor itself. On-die signing would give both. Say this out loud in the post rather than hiding it.

### Development without GPUs

Moments 2 and 3 can be built against recorded traces first: replay a training trace while declaring inference. Recorded traces to use:

- Serving: `../telemetry-logs/serving-traces-vllm/runs/` (4 runs, 8x H200, 20 Hz power)
- Pretraining: `../../archived/harvest/corpus_public/normalized/samples.parquet` (pythia / neox / smollm3, wandb cadence, so much coarser than 20 Hz)

Only the final live demo needs a rented GPU box.

## Dummy classifier: three rules

These are rules, not a model. Each one should be stateable in one sentence an auditor could check by hand.

1. **Undeclared load.** A GPU is drawing well above idle while no declared job covers it. Probably the most robust and most useful check.
2. **Declared idle but busy.** A declared-idle window shows sustained load.
3. **Declared inference but training-shaped.** A declared-inference window shows a steady, repeating power cycle at training-step cadence, and/or lockstep across GPUs.

Rule 3 is also what catches moment 3 when it is applied across a declared partition boundary.

Thresholds still to set from the recorded data:

- Idle power: about 122 W per H200 on the smoke run, n = 1 box.
- Autocorrelation peak height and period band for rule 3.
- Cross-GPU correlation cutoff for lockstep.

## Protocol shape (v0)

1. The prover opens one long-lived connection out to the node.
2. The node pushes a fresh nonce every window (e.g. 5–10 s).
3. The prover samples at 1–20 Hz into a hash chain and signs each window's chain head plus the nonce (Ed25519, CPU key).
4. Sign the **cumulative energy counter**, not just instantaneous power, so a dropped window shows up as an unexplained energy jump.
5. The signer sits behind an interface shaped like SPDM `GET_MEASUREMENTS` (nonce in, signed blob out), so an on-die signer can replace it later.

Fine to redo this once v0 works; start minimal.

## Known constraints

- vast.ai boxes are containers with no Slurm, so declarations come from a small fake-scheduler CLI. The Lucid vcf cluster has real `sacct`, but check outbound internet from a compute node first (we're unprivileged there).
- Reading the NVML energy counter costs about 5.4 ms per GPU per read (n = 1 box). Round-robin it, as `logger.py` already does.
- The prover's and node's clocks will disagree. Rely on nonces for freshness, not timestamps.
- Measure signing throughput per window.
