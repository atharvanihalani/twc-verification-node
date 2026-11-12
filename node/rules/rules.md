# The rules

What the code in this folder does, in plain language. Someone who reads only this file should be able to write code that gives the same verdicts; `test_spec.py` is such a second implementation and is checked against the real one. If the code and this file disagree, one of them has a bug.

| Part | File |
|---|---|
| Shared steps | `__init__.py` (the window and the verdict), `../app.py` (what goes in, and violation periods) |
| Rule 1 | `undeclared_load.py` |
| Rule 2 | `training_shaped.py` |
| Rule 3 | `boundary_traffic.py` |

Every comparison is strict: "over 200 W" means greater than 200. All thresholds are placeholders from a handful of runs on 8× H200 boxes.

## Shared steps

Run every 10 s.

**Inputs**

- Power readings: for each GPU, (time, watts) pairs, about ten a second. Times are Unix seconds from the box's clock.
- Once-a-second readings: for each GPU, (time, value) pairs for two counters, about one a second each. **PCIe sent** is the GB/s the GPU sends out over PCIe. **Tensor-core utilisation** is the percentage of time its tensor cores were busy. A GPU may have none of either.
- Network sent, optional: for each box, (time, value) pairs, about one a second: the GB/s the box sends to other boxes over its high-speed network ports (the RDMA ports), all ports added up. A GPU named `<box>:<index>` is on that box; a GPU whose name has no colon is a box by itself. A box may have none.
- Each GPU's model name, as its driver reports it (for example `NVIDIA B200`). May be missing.
- Jobs from the scheduler: an ID, a declared type (`inference` or `pretraining`, possibly missing), the GPUs it was given, a start time and, once over, an end time.

**1. Streaming GPUs and the end time.** A GPU is streaming if its newest power reading is less than 15 s older than the node's clock. If none is, see "No telemetry" below. The end time is the oldest of the streaming GPUs' newest reading times.

**2. Declared jobs.** A job counts if its start time is known (not an estimate) and not after the end time; it has not ended, or ended less than 60 s before the end time; and it has a declared type. A job without a declared type is ignored, so its GPUs are covered by no job.

**3. The grid.** 600 slots, 0.1 s apart; the last is the end time rounded to the nearest 0.1 s. Each power reading goes in the slot nearest its time (an exact half goes to the even slot number). Readings outside the 600 slots are dropped; if two readings of one GPU land in one slot, the later one in the list wins. Keep only the slots in which every streaming GPU has a reading: the **kept slots**, in time order.

**4. Coverage.** The number of kept slots times 0.1 s. Under 30 s, the verdict is `warming_up` and no rule is judged.

**5. Quantities the rules share.**

- A GPU's mean power: the average of its readings over the kept slots.
- A GPU's **busy floor**: **300 W** if its model name contains `B200` or `B300` (ignoring case), otherwise **200 W**. A GPU with no known model name gets 200 W.
- A job's GPUs are the ones it was given that are streaming; a job with none is left out. A job's mean power: the average of all its GPUs' readings over the kept slots.
- A job is **busy** if its mean power is over the highest busy floor among its GPUs.
- A job's value for a once-a-second counter:
  1. For each of its GPUs, take that counter's readings timed later than the end time minus 60 s and no later than the end time plus 1 s.
  2. A GPU with at least **15** such readings contributes their median.
  3. The job's value is the average of those medians. If no GPU contributed, the job has no value.

**6. Verdict.** `flagged` if any rule below is broken by anything, otherwise `compliant`.

## Rule 1: no busy GPU outside a declared job

For each streaming GPU that no counted job was given: if its mean power is over its busy floor, that GPU breaks the rule.

## Rule 2: a job declared inference does not look like training

A job declared `inference` breaks the rule if its value for tensor-core utilisation exists and is over **45%**.

(Until 2026-11-11 a second test, "power spread", could also break this rule. It was removed: it fired on honest serving under a steady load.)

## Rule 3: separate jobs do not send bulk data to each other

One test. Jobs declared `pretraining` never break the rule. A job with no network-sent value cannot break it, so a box whose collector reports no network readings is not judged by this rule.

**Test 1, network.** Each job declared `inference` that has a network-sent value over **0.001 GB/s** breaks the rule, alone. The job does not have to be busy, and no second job is needed.

A job's network-sent value:

1. List the boxes its GPUs are on. If that is not exactly one box, the job has no value (the counter cannot tell traffic between a job's own boxes from traffic that leaves them).
2. Take that box's network-sent readings timed later than the end time minus 60 s and no later than the end time plus 1 s. With fewer than **15** such readings the job has no value.
3. The value is the **mean** of those readings (not the median: a hand-off moves in bursts). A box's traffic counts for every single-box job on it.

**Test 2, PCIe: switched off since 2026-11-11** (in the code: `PCIE_TEST_ON = False`). When on: for each pair of jobs that are both busy, each of the two that is declared `inference`, has a PCIe-sent value, and has that value over **0.35 GB/s** is an offender, and a pair with at least one offender breaks the rule.

A job's PCIe-sent value is its value for the PCIe sent counter, as defined in the shared steps.

## Violation periods

The page shows periods, not single evaluations. Each broken rule has a subject: for rule 1 the box (all its offending GPUs share one period), for rule 2 the job, for rule 3 the pair of jobs.

- When a rule breaks for a subject with no open period, open one, starting at the end time minus 10 s.
- Each time it breaks again, move the period's end to that evaluation's end time and add any new GPUs or jobs.
- At the first evaluation in which it does not break, close the period.
- A `warming_up` evaluation changes nothing.
- **No telemetry:** if no GPU is streaming, close every open period.

## What the rules do not do

- Nothing checks that a job declared `pretraining` is really training.
- Only power, tensor-core utilisation and PCIe sent are read.
- Flag 2 of rule 2 is also raised by honest serving under a heavy, steady load.
- One patchy GPU reduces coverage for every GPU, because a slot is kept only when all of them have a reading.
